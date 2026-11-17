"""Evaluate an LTCC checkpoint under standard and clothes-changing protocols.

The MINER paper reports the official LTCC clothes-changing setting.  This
script prints both protocols to make protocol mismatches visible:

  standard: remove same PID + same camera
  cloth-changing: remove same PID + same camera OR same clothing
  official cloth-changing subset: query only the IDs in info/
  cloth-change_id_test.txt, with the same clothing-aware filtering

Run through ``run_ppu.sh`` for the local PPU runtime.
"""
from __future__ import annotations

import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import ppu_runtime  # noqa: F401  # register PPU torchvision schemas first
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from config import IMAGENET_MEAN, IMAGENET_STD, TVRID_DEFAULT_NUM_CLASSES
from dataset import build_dataset
from evaluation import eval_func
from model import build_model
from utils import load_checkpoint


class _ImageDataset(Dataset):
    def __init__(self, entries, transform):
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image = self.transform(Image.open(entry["image_path"]).convert("RGB"))
        return image, idx, entry["camid"]


def _encode(model, entries, transform, device, batch_size, num_workers):
    loader = DataLoader(
        _ImageDataset(entries, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    output = torch.empty((len(entries), 1024), dtype=torch.float32)
    with torch.no_grad():
        for images, indices, camera_ids in loader:
            with torch.amp.autocast("cuda"):
                feats = model(
                    images.to(device),
                    camera_id=camera_ids.to(device, dtype=torch.long),
                )
            feats = F.normalize(feats.float(), dim=1).cpu()
            output[indices] = feats
    return output.numpy()


def _metrics(name, q_entries, g_entries, q_feat, g_feat, cloth_changing=False):
    q_pids = np.asarray([x["pid"] for x in q_entries])
    g_pids = np.asarray([x["pid"] for x in g_entries])
    q_cams = np.asarray([x["camid"] for x in q_entries])
    g_cams = np.asarray([x["camid"] for x in g_entries])
    q_clothes = np.asarray([x["clothid"] for x in q_entries])
    g_clothes = np.asarray([x["clothid"] for x in g_entries])
    dist = 1.0 - q_feat @ g_feat.T
    cmc, mAP = eval_func(
        dist,
        q_pids,
        g_pids,
        q_cams,
        g_cams,
        max_rank=10,
        q_clothids=q_clothes,
        g_clothids=g_clothes,
        cloth_changing=cloth_changing,
    )
    print(
        f"{name}: mAP={mAP * 100:.4f}% "
        f"Rank-1={cmc[0] * 100:.4f}% "
        f"Rank-5={cmc[4] * 100:.4f}% "
        f"Rank-10={cmc[9] * 100:.4f}%"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="/home/data")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--camera-xishu",
        type=float,
        default=3.0,
        help="Camera embedding coefficient; use 0 for cross-dataset transfer.",
    )
    args = parser.parse_args()

    dataset = build_dataset("ltcc", args.data_root, val_ratio=0.0, seed=1234)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    num_classes = state.get("head.weight", torch.empty(TVRID_DEFAULT_NUM_CLASSES)).shape[0]
    model = build_model(
        num_classes=num_classes,
        num_cameras=dataset.num_camera,
        camera_xishu=args.camera_xishu,
    )
    load_checkpoint(model, args.checkpoint)
    device = torch.device(args.device)
    model.to(device).eval()
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    q_feat = _encode(model, dataset.query, transform, device, args.batch_size, args.num_workers)
    g_feat = _encode(model, dataset.gallery, transform, device, args.batch_size, args.num_workers)
    _metrics("standard", dataset.query, dataset.gallery, q_feat, g_feat, cloth_changing=False)
    _metrics("cloth-changing/all-test-IDs", dataset.query, dataset.gallery, q_feat, g_feat, cloth_changing=True)

    subset_file = osp.join(dataset.dataset_dir_abs, "info", "cloth-change_id_test.txt")
    with open(subset_file, "r", encoding="utf-8") as f:
        subset_pids = {int(line.strip()) for line in f if line.strip()}
    q_subset_idx = [i for i, entry in enumerate(dataset.query) if entry["pid"] in subset_pids]
    q_subset = [dataset.query[i] for i in q_subset_idx]
    _metrics(
        "cloth-changing/official-subset",
        q_subset,
        dataset.gallery,
        q_feat[q_subset_idx],
        g_feat,
        cloth_changing=True,
    )


if __name__ == "__main__":
    main()
