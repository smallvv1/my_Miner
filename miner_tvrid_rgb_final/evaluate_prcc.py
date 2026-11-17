"""Evaluate a checkpoint on the PRCC test split.

The MINER paper uses the cross-clothes Camera A -> Camera C protocol for
PRCC.  We also report the same-clothes Camera B -> Camera A protocol as a
diagnostic.  All images in the corresponding camera folders are used, and no
re-ranking or test-set adaptation is applied.

Run through ``run_ppu.sh`` for the local PPU runtime.
"""
from __future__ import annotations

import argparse
import glob
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import ppu_runtime  # noqa: F401  # register PPU torchvision schemas first
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from config import IMAGENET_MEAN, IMAGENET_STD, TVRID_DEFAULT_NUM_CLASSES
from evaluation import eval_func
from model import build_model
from utils import load_checkpoint


CAMERA_IDS = {"A": 0, "B": 1, "C": 2}
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


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


def _collect_camera(data_root, camera):
    camera_dir = osp.join(data_root, "prcc", "rgb", "test", camera)
    if not osp.isdir(camera_dir):
        raise FileNotFoundError(f"PRCC camera directory does not exist: {camera_dir}")

    entries = []
    pid_dirs = sorted(
        (path for path in glob.glob(osp.join(camera_dir, "*")) if osp.isdir(path)),
        key=lambda path: int(osp.basename(path)),
    )
    for pid_dir in pid_dirs:
        pid = int(osp.basename(pid_dir))
        image_paths = []
        for pattern in IMAGE_EXTENSIONS:
            image_paths.extend(glob.glob(osp.join(pid_dir, pattern)))
        for image_path in sorted(set(image_paths)):
            entries.append(
                {
                    "image_path": image_path,
                    "pid": pid,
                    "camid": CAMERA_IDS[camera],
                }
            )
    if not entries:
        raise RuntimeError(f"No PRCC images found in {camera_dir}")
    return entries


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


def _metrics(name, q_entries, g_entries, q_feat, g_feat):
    q_pids = np.asarray([entry["pid"] for entry in q_entries])
    g_pids = np.asarray([entry["pid"] for entry in g_entries])
    q_cams = np.asarray([entry["camid"] for entry in q_entries])
    g_cams = np.asarray([entry["camid"] for entry in g_entries])
    dist = 1.0 - q_feat @ g_feat.T
    cmc, mAP = eval_func(
        dist,
        q_pids,
        g_pids,
        q_cams,
        g_cams,
        max_rank=10,
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera-xishu", type=float, default=3.0)
    args = parser.parse_args()

    camera_a = _collect_camera(args.data_root, "A")
    camera_b = _collect_camera(args.data_root, "B")
    camera_c = _collect_camera(args.data_root, "C")
    print(
        f"PRCC test images: A={len(camera_a)}, B={len(camera_b)}, "
        f"C={len(camera_c)}; identities={len({entry['pid'] for entry in camera_a})}"
    )

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {key.replace("module.", ""): value for key, value in state.items()}
    num_classes = state.get("head.weight", torch.empty(TVRID_DEFAULT_NUM_CLASSES)).shape[0]
    num_cameras = state.get("camera_embed", torch.empty((12, 1))).shape[0]
    model = build_model(
        num_classes=num_classes,
        num_cameras=max(int(num_cameras), len(CAMERA_IDS)),
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
    feat_a = _encode(model, camera_a, transform, device, args.batch_size, args.num_workers)
    feat_b = _encode(model, camera_b, transform, device, args.batch_size, args.num_workers)
    feat_c = _encode(model, camera_c, transform, device, args.batch_size, args.num_workers)

    # Main MINER protocol: cross-clothes Camera A -> Camera C.
    _metrics("cross-clothes/A->C", camera_a, camera_c, feat_a, feat_c)
    # Same-clothes diagnostic: Camera B -> Camera A.
    _metrics("same-clothes/B->A", camera_b, camera_a, feat_b, feat_a)


if __name__ == "__main__":
    main()
