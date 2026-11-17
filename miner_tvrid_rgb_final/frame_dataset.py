"""Flat frame-level Dataset for TVRID and LTCC training images.

Each ``__getitem__`` returns ``(image, pid, camera_id)``.  LTCC keeps the
clothing annotation in separate mappings so MINER can sample the exact
``(pid, camera, clothing)`` groups required by Algorithm 1 in the paper.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import ppu_runtime  # noqa: F401  # register PPU torchvision schemas first
from torchvision.transforms import ToTensor


class TVRIDFrameDataset(Dataset):
    """Map-style dataset returning single TVRID frames.

    Builds index mappings from a TVRID dataset object:
      - idx -> (image_path, pid, appearance_id)
      - pid -> [indices]
      - (pid, appearance_id) -> [indices]

    Args:
        tvrid_dataset: TVRID object with .train list of dicts.
        transform: torchvision transform to apply to each image.
    """

    def __init__(self, tvrid_dataset, transform=None):
        self.transform = transform
        self._paths: List[str] = []
        self._pids: List[int] = []
        self._camids: List[int] = []
        self._appearance_ids: List[int] = []
        self._clothes: List[int] = []

        self.pid_to_indices: Dict[int, List[int]] = defaultdict(list)
        self.pid_cam_to_indices: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        self.pid_cam_cloth_to_indices: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
        self.pid_cam_to_clothes: Dict[Tuple[int, int], set] = defaultdict(set)
        self.pid_to_cams: Dict[int, set] = defaultdict(set)

        # LTCC provides both camera and clothing labels.  TVRID only has a
        # camera/appearance factor, so it continues through the legacy path.
        self.has_clothing_annotations = any("clothid" in entry for entry in tvrid_dataset.train)

        for idx, entry in enumerate(tvrid_dataset.train):
            path = entry["image_path"]
            pid = entry["pid"]
            camera_id = entry.get("camera_id", entry.get("camid", 0))
            appearance_id = entry.get("appearance_id", camera_id)
            clothing_id = entry.get("clothid", appearance_id)

            self._paths.append(path)
            self._pids.append(pid)
            self._camids.append(camera_id)
            self._appearance_ids.append(appearance_id)
            self._clothes.append(clothing_id)

            self.pid_to_indices[pid].append(idx)
            self.pid_cam_to_indices[(pid, camera_id)].append(idx)
            self.pid_cam_cloth_to_indices[(pid, camera_id, clothing_id)].append(idx)
            self.pid_cam_to_clothes[(pid, camera_id)].add(clothing_id)
            self.pid_to_cams[pid].add(camera_id)

        self.valid_pids: List[int] = [
            pid for pid, idxs in self.pid_to_indices.items() if len(idxs) >= 2
        ]

    def __len__(self) -> int:
        return len(self._paths)

    def get_pid(self, idx: int) -> int:
        return self._pids[idx]

    def get_camid(self, idx: int) -> int:
        return self._camids[idx]

    def get_appearance_id(self, idx: int) -> int:
        """Return the appearance factor (clothing for LTCC, camera for TVRID)."""
        return self._appearance_ids[idx]

    def get_clothing_id(self, idx: int) -> int:
        """Return the clothing ID, or the legacy appearance factor."""
        return self._clothes[idx]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        img = Image.open(self._paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = ToTensor()(img)
        return img, self._pids[idx], self._camids[idx]
