"""MINER batch sampler for TVRID/LTCC (Algorithm 1 from the paper).

For LTCC, this implements the paper's exact two-level sampling procedure:
camera pair -> candidate PIDs -> soft hard-negative PIDs -> stochastic
cross-camera, cross-clothing positive pair -> exact image groups.

Yields flat index lists structured as [q_indices..., g_indices...] for use
with PyTorch DataLoader's batch_sampler parameter.
"""
from __future__ import annotations

import json
import os.path as osp
import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler


def _load_embeddings(path: str) -> Dict[str, torch.Tensor]:
    """Load pre-computed embeddings from a JSON file.

    Expected format: {"image_path": [float, ...], ...}
    Returns dict mapping normalized path suffix -> L2-normalized 1D tensor.
    Path keys are normalized to the portion after 'train/' so embeddings
    generated on one machine match datasets loaded on another.
    """
    with open(path, "r") as f:
        data = json.load(f)
    result = {}
    for k, v in data.items():
        t = torch.tensor(v, dtype=torch.float32)
        t = F.normalize(t.unsqueeze(0), p=2, dim=1).squeeze(0)
        result[_path_suffix(k)] = t
    return result


def _path_suffix(path: str) -> str:
    """Extract the path suffix after 'train/' for machine-independent matching."""
    path = path.replace("\\", "/")
    marker = "/train/"
    idx = path.rfind(marker)
    if idx >= 0:
        return path[idx + len(marker):]
    return osp.basename(path)


class MINERBatchSampler(Sampler[List[int]]):
    """BatchSampler implementing MINER Algorithm 1.

    Each iteration yields a flat list of dataset indices structured as:
        [q_idx_0, ..., q_idx_{nq_total-1}, g_idx_0, ..., g_idx_{ng_total-1}]

    where nq_total = num_ids * nq, ng_total = num_ids * ng.
    The training loop splits at nq_total to separate query/gallery.

    Args:
        frame_dataset: TVRIDFrameDataset with index mappings.
        num_ids: number of identities per batch (m in the paper).
        nq: query frames per identity.
        ng: gallery frames per identity.
        embeddings_path: JSON file with pre-computed embeddings for centroids.
        mining_temperature: softmax temperature for hard ID sampling.
        seed: random seed for reproducibility.
    """

    def __init__(
        self,
        frame_dataset,
        num_ids: int,
        nq: int,
        ng: int,
        embeddings_path: Optional[str] = None,
        mining_temperature: float = 2.0,
        seed: int = 1234,
    ):
        self.dataset = frame_dataset
        self.num_ids = num_ids
        self.nq = nq
        self.ng = ng
        self.mining_temperature = mining_temperature
        self._seed = seed
        self._epoch = 0

        # Centroid state
        self._centroid_matrix: Optional[torch.Tensor] = None
        self._centroid_pids: List[int] = []
        self._cam_centroids: Dict[int, Dict[int, torch.Tensor]] = {}
        self._pid_cam_cloth_centroids: Dict[Tuple[int, int, int], torch.Tensor] = {}
        self._candidate_pids_by_pair: Dict[Tuple[int, int], List[int]] = {}
        self._camera_pairs: List[Tuple[int, int]] = []
        self._exact_ltcc_sampling = bool(
            getattr(frame_dataset, "has_clothing_annotations", False)
        )

        if embeddings_path and osp.isfile(embeddings_path):
            print(f"Loading MINER embeddings from: {embeddings_path}")
            emb_map = _load_embeddings(embeddings_path)
            self._build_centroids(emb_map)
        else:
            print("No embeddings path — using random sampling (no MINER mining)")

        if self._exact_ltcc_sampling:
            self._build_ltcc_candidate_pairs()
            if self._camera_pairs:
                print(
                    "  MINER LTCC sampler: "
                    f"{len(self._camera_pairs)} ordered camera pairs, "
                    f"{sum(len(v) for v in self._candidate_pids_by_pair.values())} "
                    "pair-candidate entries"
                )
            else:
                print("  Warning: no valid LTCC camera pairs; using fallback sampling")

        # Batches per epoch
        denom = num_ids * nq
        self._length = max(1, len(frame_dataset) // max(denom, 1))

    @property
    def nq_per_batch(self) -> int:
        """Total query frames per batch = num_ids * nq."""
        return self.num_ids * self.nq

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self._length

    def _build_centroids(self, emb_map: Dict[str, torch.Tensor]) -> None:
        """Compute PID and appearance/clothing centroids from embeddings."""
        pid_embs: Dict[int, List[torch.Tensor]] = defaultdict(list)
        pid_cam_embs: Dict[int, Dict[int, List[torch.Tensor]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for idx in range(len(self.dataset)):
            path = _path_suffix(self.dataset._paths[idx])
            pid = self.dataset.get_pid(idx)
            camid = self.dataset.get_camid(idx)
            emb = emb_map.get(path)
            if emb is None:
                continue
            pid_embs[pid].append(emb)
            pid_cam_embs[pid][camid].append(emb)

            if self._exact_ltcc_sampling:
                clothid = self.dataset.get_clothing_id(idx)
                key = (pid, camid, clothid)
                # Accumulate first; normalization happens after averaging.
                self._pid_cam_cloth_centroids.setdefault(key, []).append(emb)

        centroid_list = []
        centroid_pids = []
        for pid in self.dataset.valid_pids:
            if pid not in pid_embs:
                continue
            embs = torch.stack(pid_embs[pid], dim=0)
            centroid = F.normalize(embs.mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0)
            centroid_list.append(centroid)
            centroid_pids.append(pid)

        if centroid_list:
            self._centroid_matrix = torch.stack(centroid_list, dim=0)
            self._centroid_pids = centroid_pids
            print(f"  MINER centroids: {len(centroid_pids)} PIDs, dim={self._centroid_matrix.shape[1]}")
        else:
            print("  Warning: no embeddings matched training images")

        for pid in self.dataset.valid_pids:
            if pid not in pid_cam_embs:
                continue
            cam_centroids = {}
            for camid, embs in pid_cam_embs[pid].items():
                e = torch.stack(embs, dim=0)
                cam_centroids[camid] = F.normalize(
                    e.mean(dim=0, keepdim=True), p=2, dim=1
                ).squeeze(0)
            if len(cam_centroids) >= 2:
                self._cam_centroids[pid] = cam_centroids

        if self._exact_ltcc_sampling:
            for key, embs in list(self._pid_cam_cloth_centroids.items()):
                e = torch.stack(embs, dim=0)
                self._pid_cam_cloth_centroids[key] = F.normalize(
                    e.mean(dim=0, keepdim=True), p=2, dim=1
                ).squeeze(0)

    def _build_ltcc_candidate_pairs(self) -> None:
        """Build Eq. (2) candidate identities for every ordered camera pair."""
        all_cameras = sorted({cam for cams in self.dataset.pid_to_cams.values() for cam in cams})
        for cam_q in all_cameras:
            for cam_g in all_cameras:
                if cam_q == cam_g:
                    continue
                candidates: List[int] = []
                for pid in self.dataset.valid_pids:
                    q_clothes = self.dataset.pid_cam_to_clothes.get((pid, cam_q), set())
                    g_clothes = self.dataset.pid_cam_to_clothes.get((pid, cam_g), set())
                    if q_clothes and g_clothes and any(aq != ag for aq in q_clothes for ag in g_clothes):
                        candidates.append(pid)
                if len(candidates) >= self.num_ids:
                    pair = (cam_q, cam_g)
                    self._candidate_pids_by_pair[pair] = candidates
                    self._camera_pairs.append(pair)

    @staticmethod
    def _weighted_sample_without_replacement(
        items: List[int], weights: List[float], k: int, rng: random.Random
    ) -> List[int]:
        """Sample ``k`` distinct items using a local, reproducible RNG."""
        remaining_items = list(items)
        remaining_weights = [max(float(w), 0.0) for w in weights]
        result: List[int] = []
        for _ in range(min(k, len(remaining_items))):
            total = sum(remaining_weights)
            if total <= 0.0:
                pick = rng.randrange(len(remaining_items))
            else:
                threshold = rng.random() * total
                cumulative = 0.0
                pick = len(remaining_items) - 1
                for i, weight in enumerate(remaining_weights):
                    cumulative += weight
                    if cumulative >= threshold:
                        pick = i
                        break
            result.append(remaining_items.pop(pick))
            remaining_weights.pop(pick)
        return result

    def _soft_sample_ids(self, candidates: List[int], rng: random.Random) -> List[int]:
        """Paper Eq. (4): sample m candidate PIDs without replacement."""
        if len(candidates) <= self.num_ids:
            result = list(candidates)
            while result and len(result) < self.num_ids:
                result.append(rng.choice(result))
            return result
        if self._centroid_matrix is None:
            return rng.sample(candidates, self.num_ids)

        centroid_by_pid = {
            pid: self._centroid_matrix[i]
            for i, pid in enumerate(self._centroid_pids)
        }
        seed_pid = rng.choice(candidates)
        seed_centroid = centroid_by_pid.get(seed_pid)
        if seed_centroid is None:
            return rng.sample(candidates, self.num_ids)
        scored = []
        for pid in candidates:
            centroid = centroid_by_pid.get(pid)
            scored.append(float(torch.dot(centroid, seed_centroid).item()) if centroid is not None else 0.0)
        temperature = max(float(self.mining_temperature), 1e-6)
        max_score = max(scored)
        weights = [pow(2.718281828, (score - max_score) / temperature) for score in scored]
        return self._weighted_sample_without_replacement(candidates, weights, self.num_ids, rng)

    def _pick_hard_id_set(
        self, rng: random.Random, candidates: Optional[List[int]] = None
    ) -> List[int]:
        """Algorithm 1, Steps 3-5: sample PIDs from one camera-pair set."""
        k = self.num_ids
        candidates = list(candidates or (
            self._centroid_pids if self._centroid_matrix is not None else self.dataset.valid_pids
        ))

        if len(candidates) <= k:
            result = list(candidates)
            while len(result) < k:
                result.append(rng.choice(candidates))
            return result

        if self._centroid_matrix is None:
            return rng.sample(candidates, k)

        return self._soft_sample_ids(candidates, rng)

    def _pick_ltcc_clothing_pair(
        self, pid: int, cam_q: int, cam_g: int, rng: random.Random
    ) -> Optional[Tuple[int, int]]:
        """Paper Eqs. (5)-(8): stochastic soft-argmin clothing pair."""
        q_clothes = self.dataset.pid_cam_to_clothes.get((pid, cam_q), set())
        g_clothes = self.dataset.pid_cam_to_clothes.get((pid, cam_g), set())
        pairs = [(aq, ag) for aq in q_clothes for ag in g_clothes if aq != ag]
        if not pairs:
            return None

        centroid_pairs = []
        pair_sims = []
        for aq, ag in pairs:
            q_centroid = self._pid_cam_cloth_centroids.get((pid, cam_q, aq))
            g_centroid = self._pid_cam_cloth_centroids.get((pid, cam_g, ag))
            if q_centroid is None or g_centroid is None:
                continue
            centroid_pairs.append((aq, ag))
            pair_sims.append(float(torch.dot(q_centroid, g_centroid).item()))
        if not centroid_pairs:
            return rng.choice(pairs)

        temperature = max(float(self.mining_temperature), 1e-6)
        max_neg_sim = max(-sim for sim in pair_sims)
        weights = [pow(2.718281828, ((-sim) - max_neg_sim) / temperature) for sim in pair_sims]
        return self._weighted_sample_without_replacement(centroid_pairs, weights, 1, rng)[0]

    def _pick_hard_camera_pair(self, pid: int, rng: random.Random) -> Tuple[int, int]:
        """Algorithm 1, Step 5: select the most dissimilar appearance pair."""
        if pid not in self._cam_centroids or len(self._cam_centroids[pid]) < 2:
            return -1, -1

        cam_centroids = self._cam_centroids[pid]
        cams = list(cam_centroids.keys())

        best_pair = None
        best_sim = float("inf")
        for i, c1 in enumerate(cams):
            for c2 in cams[i + 1:]:
                sim = (cam_centroids[c1] @ cam_centroids[c2]).item()
                if sim < best_sim:
                    best_sim = sim
                    best_pair = (c1, c2)

        if best_pair is None:
            return -1, -1

        if rng.random() < 0.5:
            return best_pair
        return best_pair[1], best_pair[0]

    def _sample_indices_from_camera(self, pid: int, camid: int, n: int, rng: random.Random) -> List[int]:
        """Sample n frame indices from a (pid, camera) combination."""
        indices = self.dataset.pid_cam_to_indices.get((pid, camid), [])
        if not indices:
            return []
        if len(indices) >= n:
            return rng.sample(indices, n)
        return [rng.choice(indices) for _ in range(n)]

    def _sample_indices_from_clothing(
        self, pid: int, camid: int, clothid: int, n: int, rng: random.Random
    ) -> List[int]:
        """Sample frames from one exact LTCC ``(pid, camera, clothing)`` group."""
        indices = self.dataset.pid_cam_cloth_to_indices.get((pid, camid, clothid), [])
        if not indices:
            return []
        if len(indices) >= n:
            return rng.sample(indices, n)
        return [rng.choice(indices) for _ in range(n)]

    def _sample_flat(self, pid: int, nq: int, ng: int, rng: random.Random) -> Tuple[List[int], List[int]]:
        """Fallback: sample Q/G indices from flat per-PID pool."""
        indices = self.dataset.pid_to_indices.get(pid, [])
        if not indices:
            return [], []
        if len(indices) >= nq + ng:
            sampled = rng.sample(indices, nq + ng)
            return sampled[:nq], sampled[nq:]
        q = [rng.choice(indices) for _ in range(nq)]
        g = [rng.choice(indices) for _ in range(ng)]
        return q, g

    def _sample_qg_for_pid(self, pid: int, nq: int, ng: int, rng: random.Random) -> Tuple[List[int], List[int]]:
        """Sample Q/G indices for a PID using a hard appearance pair."""
        cam_q, cam_g = self._pick_hard_camera_pair(pid, rng)
        if cam_q == -1:
            return self._sample_flat(pid, nq, ng, rng)

        q_indices = self._sample_indices_from_camera(pid, cam_q, nq, rng)
        g_indices = self._sample_indices_from_camera(pid, cam_g, ng, rng)

        if not q_indices or not g_indices:
            return self._sample_flat(pid, nq, ng, rng)

        return q_indices, g_indices

    def _sample_ltcc_qg_for_pid(
        self, pid: int, cam_q: int, cam_g: int, nq: int, ng: int, rng: random.Random
    ) -> Tuple[List[int], List[int]]:
        pair = self._pick_ltcc_clothing_pair(pid, cam_q, cam_g, rng)
        if pair is None:
            return [], []
        cloth_q, cloth_g = pair
        q_indices = self._sample_indices_from_clothing(pid, cam_q, cloth_q, nq, rng)
        g_indices = self._sample_indices_from_clothing(pid, cam_g, cloth_g, ng, rng)
        return q_indices, g_indices

    def __iter__(self) -> Iterator[List[int]]:
        for batch_idx in range(self._length):
            rng = random.Random(self._seed + batch_idx + self._epoch * self._length)

            if self._exact_ltcc_sampling and self._camera_pairs:
                cam_q, cam_g = rng.choice(self._camera_pairs)
                candidates = self._candidate_pids_by_pair[(cam_q, cam_g)]
                chosen_pids = self._pick_hard_id_set(rng, candidates)
            else:
                cam_q, cam_g = -1, -1
                chosen_pids = self._pick_hard_id_set(rng)

            q_all: List[int] = []
            g_all: List[int] = []

            for pid in chosen_pids:
                if self._exact_ltcc_sampling and cam_q != -1:
                    q_idx, g_idx = self._sample_ltcc_qg_for_pid(
                        pid, cam_q, cam_g, self.nq, self.ng, rng
                    )
                else:
                    q_idx, g_idx = self._sample_qg_for_pid(pid, self.nq, self.ng, rng)
                if not q_idx or not g_idx:
                    # Fallback: substitute a random valid PID to keep batch size fixed
                    fallback = rng.choice(self.dataset.valid_pids)
                    q_idx, g_idx = self._sample_flat(fallback, self.nq, self.ng, rng)
                q_all.extend(q_idx)
                g_all.extend(g_idx)

            # Structured: [all query indices, then all gallery indices]
            yield q_all + g_all
