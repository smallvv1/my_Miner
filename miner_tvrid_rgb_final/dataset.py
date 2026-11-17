"""Dataset adapters for TVRID RGB, LTCC, and PRCC clothes-changing ReID.

The adapters provide:
  - Training frames with PID labels and camera IDs
  - LTCC clothing IDs as MINER appearance factors
  - PRCC camera/clothing IDs as MINER appearance factors
  - Test passages or frame splits for ranking/evaluation
  - Frame listing and evenly-spaced selection utilities
"""

from __future__ import annotations

import csv
import os
import os.path as osp
from typing import Dict, List, Tuple

import numpy as np

from config import TVRID_NUM_CAMERAS

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# LTCC uses 14 clothing IDs and 12 camera IDs in its official split.  The
# sampler uses ``appearance_id`` (clothing) for LTCC, while ``camid`` remains
# the actual camera in the dataset metadata.
LTCC_NUM_CLOTHES = 14
LTCC_NUM_CAMERAS = 12

# PRCC has three test/training camera views.  Cameras A and B use the same
# outfit while camera C contains the changed-outfit view.
PRCC_NUM_CLOTHES = 2
PRCC_NUM_CAMERAS = 3

# Camera name -> 0-based camera id
CAM_NAME_TO_ID = {
    "flat": 0,
    "upward": 1,
    "downward": 2,
    "upsidedown": 3,
    "upsideDown": 3,
}


def _list_frames(passage_dir: str) -> List[str]:
    """Return sorted absolute paths for RGB frames in a passage dir."""
    if not osp.isdir(passage_dir):
        return []
    return sorted(
        osp.join(passage_dir, f)
        for f in os.listdir(passage_dir)
        if "_RGB" in f and osp.splitext(f)[1].lower() in _IMG_EXTS
    )


def _list_images(image_dir: str) -> List[str]:
    """Return sorted image paths from a flat image directory."""
    if not osp.isdir(image_dir):
        return []
    return sorted(
        osp.join(image_dir, name)
        for name in os.listdir(image_dir)
        if osp.splitext(name)[1].lower() in _IMG_EXTS
    )


def _parse_ltcc_filename(path: str) -> Tuple[int, int, int, int]:
    """Parse ``pid_clothid_camerid_frameid.png`` from an LTCC filename."""
    stem = osp.splitext(osp.basename(path))[0]
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected LTCC filename: {path}")

    pid = int(parts[0])
    cloth_id = int(parts[1])
    camera_token = parts[2].lower()
    camera_id = int(camera_token[1:] if camera_token.startswith("c") else camera_token)
    frame_id = int(parts[3])
    return pid, cloth_id, camera_id, frame_id


def _select_frames_evenly(paths: List[str], n: int = 5) -> List[str]:
    """Return *n* evenly-spaced frame paths from *paths*."""
    if not paths:
        return []
    if n == 1:
        return [paths[len(paths) // 2]]
    if len(paths) <= n:
        return list(paths)
    idxs = np.linspace(0, len(paths) - 1, num=n, dtype=int).tolist()
    return [paths[i] for i in idxs]


class TVRID:
    """TVRID Top-View Re-IDentification dataset (RGB only).

    Reads TVRID data and populates ``self.train`` with per-frame dicts:
    ``{image_path, pid, camid, gallery_id}``.

    The TVRID dataset contains top-view surveillance passages from 4 camera
    angles (flat, upward, downward, upsideDown). Each passage is a sequence
    of RGB frames for one person crossing a corridor.
    """

    dataset_dir = "DB_extracted"

    def __init__(
        self,
        root: str = "data",
        val_ratio: float = 0.0,
        seed: int = 1234,
        eval_frames_per_passage: int = 5,
    ):
        self.root = root
        self.dataset_dir_abs = osp.join(root, self.dataset_dir)

        train_csv = osp.join(self.dataset_dir_abs, "train_labels.csv")
        test_csv = osp.join(self.dataset_dir_abs, "public_test_labels.csv")

        if not osp.isfile(train_csv):
            raise FileNotFoundError(f"Training CSV not found: {train_csv}")

        # Parse training CSV
        all_rows = self._read_csv(train_csv)
        person_ids = sorted(set(r["person_id"] for r in all_rows))

        # Train / val split by identity
        rng = np.random.RandomState(seed)
        shuffled_pids = list(person_ids)
        rng.shuffle(shuffled_pids)
        n_val = max(0 if val_ratio == 0 else 1, int(len(shuffled_pids) * val_ratio))
        train_pids = set(shuffled_pids[: len(shuffled_pids) - n_val])

        # Contiguous relabel: train PIDs -> 0..N-1
        train_pid2label = {
            pid: idx for idx, pid in enumerate(sorted(train_pids))
        }

        train_rows = [r for r in all_rows if r["person_id"] in train_pids]

        # Explode train passages into individual frames
        self.train: List[dict] = []
        self.passage_metadata: Dict[str, dict] = {}
        train_base = osp.join(self.dataset_dir_abs, "train")

        for row in train_rows:
            pid_label = train_pid2label[row["person_id"]]
            cam_name = row["cam_name"].strip().lower()
            camid = CAM_NAME_TO_ID.get(cam_name, 0)
            passage_dir = osp.join(train_base, row["path"].replace("\\", "/"))
            frames = _list_frames(passage_dir)
            if not frames:
                continue

            gid = str(row["gallery_id"])
            self.passage_metadata[gid] = {
                "person_id": row["person_id"],
                "cam_name": cam_name,
                "passage_name": row.get("passage_name", ""),
                "cam_id": camid,
            }
            for frame_path in frames:
                self.train.append(
                    {
                        "image_path": frame_path,
                        "pid": pid_label,
                        "camid": camid,
                        "gallery_id": gid,
                    }
                )

        self.num_train_pids: int = len(train_pids)
        self.num_train_imgs: int = len(self.train)
        self.num_camera: int = TVRID_NUM_CAMERAS

        # Parse test CSV (query = gallery = every passage)
        self.query: List[dict] = []
        self.gallery: List[dict] = []

        if osp.isfile(test_csv):
            test_rows = self._read_csv(test_csv)
            test_base = osp.join(self.dataset_dir_abs, "test_public")
            for row in test_rows:
                passage_dir = osp.join(test_base, row["path"].replace("\\", "/"))
                frames = _list_frames(passage_dir)
                selected = _select_frames_evenly(frames, eval_frames_per_passage)
                if not selected:
                    continue

                cam_name = self._infer_cam_name(row["path"])
                camid = CAM_NAME_TO_ID.get(cam_name, 0)
                gid = str(row["gallery_id"])
                for frame_path in selected:
                    entry = {
                        "image_path": frame_path,
                        "pid": -1,
                        "camid": camid,
                        "gallery_id": gid,
                    }
                    self.query.append(entry)
                    self.gallery.append(entry)

        print(
            f"=> TVRID (rgb): "
            f"{self.num_train_pids} train IDs, "
            f"{self.num_train_imgs} train frames, "
            f"{len(self.query)} test frames"
        )

    @staticmethod
    def _read_csv(csv_path: str) -> List[dict]:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f)]

    @staticmethod
    def _infer_cam_name(path: str) -> str:
        for part in path.replace("\\", "/").split("/"):
            lower = part.strip().lower()
            if lower in CAM_NAME_TO_ID:
                return lower
        return "flat"


class LTCC:
    """LTCC clothes-changing ReID dataset.

    LTCC stores one frame per file using the naming convention
    ``pid_clothing-camera_frame.png`` (for example,
    ``000_1_c10_001003.png``).  MINER's hard-positive sampler needs an
    appearance factor, so each training sample exposes its clothing ID as
    ``appearance_id`` while retaining the real camera in ``camid``.

    ``root`` is the parent directory containing ``LTCC_ReID/``.  The
    official train split is read from the extracted ``train`` directory;
    the ``info`` files are kept by the dataset release for protocol
    bookkeeping and are not needed to enumerate the same 77 training IDs.
    """

    dataset_dir = "LTCC_ReID"

    def __init__(
        self,
        root: str = "data",
        val_ratio: float = 0.0,
        seed: int = 1234,
        eval_frames_per_passage: int = 5,
    ):
        del eval_frames_per_passage  # LTCC evaluation uses individual frames.

        self.root = root
        self.dataset_dir_abs = (
            root if osp.basename(osp.normpath(root)) == self.dataset_dir
            else osp.join(root, self.dataset_dir)
        )
        train_dir = osp.join(self.dataset_dir_abs, "train")
        train_paths = _list_images(train_dir)
        if not train_paths:
            raise FileNotFoundError(
                f"LTCC training images not found: {train_dir}"
            )

        parsed = []
        for path in train_paths:
            try:
                parsed.append((path, _parse_ltcc_filename(path)))
            except ValueError as exc:
                raise ValueError(f"Cannot parse LTCC training image: {path}") from exc

        person_ids = sorted({metadata[0] for _, metadata in parsed})
        rng = np.random.RandomState(seed)
        shuffled_pids = list(person_ids)
        rng.shuffle(shuffled_pids)
        n_val = max(0 if val_ratio == 0 else 1, int(len(shuffled_pids) * val_ratio))
        train_pids = set(shuffled_pids[: len(shuffled_pids) - n_val])
        pid2label = {pid: idx for idx, pid in enumerate(sorted(train_pids))}

        self.train: List[dict] = []
        for path, (pid, cloth_id, camera_id, _frame_id) in parsed:
            if pid not in train_pids:
                continue
            self.train.append(
                {
                    "image_path": path,
                    "pid": pid2label[pid],
                    "camid": camera_id - 1,
                    "appearance_id": cloth_id - 1,
                    "clothid": cloth_id,
                    "camera_id": camera_id - 1,
                    "gallery_id": osp.splitext(osp.basename(path))[0],
                }
            )

        self.num_train_pids = len(train_pids)
        self.num_train_imgs = len(self.train)
        # ``num_camera`` is consumed by the DIFFER camera embedding.  It must
        # describe real cameras, not the clothing cardinality used by MINER's
        # hard-positive sampler.
        self.num_camera = max(
            (entry["camera_id"] for entry in self.train),
            default=LTCC_NUM_CAMERAS - 1,
        ) + 1

        self.query = self._read_eval_split("query")
        self.gallery = self._read_eval_split("test")

        print(
            f"=> LTCC: {self.num_train_pids} train IDs, "
            f"{self.num_train_imgs} train frames, "
            f"{len(self.query)} query frames, {len(self.gallery)} gallery frames"
        )

    def _read_eval_split(self, split: str) -> List[dict]:
        entries = []
        for path in _list_images(osp.join(self.dataset_dir_abs, split)):
            pid, cloth_id, camera_id, _frame_id = _parse_ltcc_filename(path)
            entries.append(
                {
                    "image_path": path,
                    "pid": pid,
                    "camid": camera_id - 1,
                    "appearance_id": cloth_id - 1,
                    "clothid": cloth_id,
                    "camera_id": camera_id - 1,
                    "gallery_id": osp.splitext(osp.basename(path))[0],
                }
            )
        return entries


class PRCC:
    """PRCC clothes-changing ReID training split.

    The extracted RGB training layout is ``prcc/rgb/train/<pid>/*.jpg``.
    Camera identity is encoded by the filename prefix (``A_``, ``B_``, or
    ``C_``).  A and B share clothing ID 0, while C is clothing ID 1, which
    lets the MINER sampler select cross-clothing camera pairs.
    """

    dataset_dir = "prcc/rgb"

    def __init__(
        self,
        root: str = "data",
        val_ratio: float = 0.0,
        seed: int = 1234,
        eval_frames_per_passage: int = 5,
    ):
        del val_ratio, seed, eval_frames_per_passage

        self.root = root
        self.dataset_dir_abs = osp.join(root, self.dataset_dir)
        train_dir = osp.join(self.dataset_dir_abs, "train")
        if not osp.isdir(train_dir):
            raise FileNotFoundError(f"PRCC training directory not found: {train_dir}")

        pid_dir_names = sorted(
            (
                name for name in os.listdir(train_dir)
                if osp.isdir(osp.join(train_dir, name))
            ),
            key=int,
        )
        person_ids = [int(name) for name in pid_dir_names]
        pid2label = {pid: idx for idx, pid in enumerate(person_ids)}

        self.train: List[dict] = []
        for pid_dir_name, pid in zip(pid_dir_names, person_ids):
            # Preserve the directory spelling (e.g. ``001``) instead of
            # reconstructing it from the integer identity.
            pid_dir = osp.join(train_dir, pid_dir_name)
            for path in _list_images(pid_dir):
                camera_name = osp.basename(path).split("_", 1)[0].upper()
                if camera_name not in {"A", "B", "C"}:
                    raise ValueError(f"Cannot parse PRCC camera from filename: {path}")
                camera_id = {"A": 0, "B": 1, "C": 2}[camera_name]
                cloth_id = 0 if camera_name in {"A", "B"} else 1
                self.train.append(
                    {
                        "image_path": path,
                        "pid": pid2label[pid],
                        "camid": camera_id,
                        "camera_id": camera_id,
                        "appearance_id": cloth_id,
                        "clothid": cloth_id,
                        "gallery_id": osp.splitext(osp.basename(path))[0],
                    }
                )

        if not self.train:
            raise FileNotFoundError(f"No PRCC training images found under: {train_dir}")

        self.num_train_pids = len(person_ids)
        self.num_train_imgs = len(self.train)
        self.num_camera = PRCC_NUM_CAMERAS
        self.query: List[dict] = []
        self.gallery: List[dict] = []

        print(
            f"=> PRCC: {self.num_train_pids} train IDs, "
            f"{self.num_train_imgs} train frames, {self.num_camera} cameras"
        )


def build_dataset(
    dataset_name: str,
    root: str,
    val_ratio: float = 0.0,
    seed: int = 1234,
    eval_frames_per_passage: int = 5,
):
    """Build one of the dataset adapters shipped with this release."""
    name = dataset_name.lower()
    if name in {"ltcc", "ltcc_reid"}:
        return LTCC(
            root=root,
            val_ratio=val_ratio,
            seed=seed,
            eval_frames_per_passage=eval_frames_per_passage,
        )
    if name in {"tvrid", "tvrid_rgb"}:
        return TVRID(
            root=root,
            val_ratio=val_ratio,
            seed=seed,
            eval_frames_per_passage=eval_frames_per_passage,
        )
    if name in {"prcc", "prcc_rgb"}:
        return PRCC(
            root=root,
            val_ratio=val_ratio,
            seed=seed,
            eval_frames_per_passage=eval_frames_per_passage,
        )
    raise ValueError(
        f"Unsupported dataset {dataset_name!r}; expected 'ltcc', 'prcc', or 'tvrid_rgb'"
    )
