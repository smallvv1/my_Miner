"""Generate per-image embeddings JSON for MINER sampler.

Loads a pretrained EVA02-Large checkpoint, runs inference on all TVRID RGB
training frames, and saves {image_path: [float, ...]} to JSON for centroid
computation in the MINER sampler (Algorithm 1).

Usage:
    cd release && python generate_embeddings.py \
        --checkpoint weights/eva02_l_bio_best.pth \
        --data-root ./data \
        --output weights/tvrid_embeddings.json
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image
import ppu_runtime  # noqa: F401  # register PPU torchvision schemas first
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from config import IMAGENET_MEAN, IMAGENET_STD, TVRID_DEFAULT_NUM_CLASSES
from model import build_model
from dataset import build_dataset
from utils import load_checkpoint


def get_args():
    parser = argparse.ArgumentParser(description="Generate MINER embeddings from checkpoint")
    parser.add_argument("--checkpoint", default="weights/eva02_l_bio_best.pth",
                        help="Path to .pth model checkpoint")
    parser.add_argument("--data-root", default="./data",
                        help="Dataset root (parent of LTCC_ReID/ or DB_extracted/)")
    parser.add_argument("--dataset", default="tvrid_rgb",
                        choices=["ltcc", "prcc", "tvrid_rgb"],
                        help="Dataset adapter to use")
    parser.add_argument("--output", default="weights/tvrid_embeddings.json",
                        help="Output JSON path")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--img-size", type=int, default=224, help="Image resize dimension")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for dataset split")
    parser.add_argument("--eval-frames", type=int, default=5, help="Frames per test passage")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = get_args()

    # Load dataset
    print(f"Loading {args.dataset} dataset from: {args.data_root}")
    dataset = build_dataset(
        dataset_name=args.dataset,
        root=args.data_root,
        val_ratio=0.0,  # embed all IDs, no validation holdout
        seed=args.seed,
        eval_frames_per_passage=args.eval_frames,
    )
    print(f"  {len(dataset.train)} training frames across {dataset.num_train_pids} PIDs")

    # Build model and load checkpoint
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    clean = {k.replace("module.", ""): v for k, v in state_dict.items()}
    num_classes = clean["head.weight"].shape[0] if "head.weight" in clean else TVRID_DEFAULT_NUM_CLASSES

    model = build_model(num_classes=num_classes, num_cameras=dataset.num_camera, camera_xishu=3)
    load_checkpoint(model, args.checkpoint)
    model.to(args.device)
    model.eval()

    transform = Compose([
        Resize((args.img_size, args.img_size)),
        ToTensor(),
        Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Run inference in batches
    print(f"\nGenerating embeddings (batch_size={args.batch_size})...")
    all_paths = [e["image_path"] for e in dataset.train]
    camera_by_path = {e["image_path"]: e["camid"] for e in dataset.train}
    embeddings = {}

    for i in tqdm(range(0, len(all_paths), args.batch_size), desc="Embedding"):
        batch_paths = all_paths[i : i + args.batch_size]
        imgs, valid_paths, valid_cameras = [], [], []
        for p in batch_paths:
            try:
                imgs.append(transform(Image.open(p).convert("RGB")))
                valid_paths.append(p)
                # The reference encoder in DIFFER is camera-conditioned.
                valid_cameras.append(camera_by_path[p])
            except Exception as e:
                print(f"  Warning: skipping {p}: {e}")

        if not imgs:
            continue

        batch = torch.stack(imgs).to(args.device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            cameras = torch.tensor(valid_cameras, dtype=torch.long, device=args.device)
            feats = model(batch, camera_id=cameras)

        feats = F.normalize(feats.float().cpu(), p=2, dim=1)
        for path, feat in zip(valid_paths, feats):
            embeddings[path] = feat.tolist()

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(embeddings, f)

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nSaved {len(embeddings)} embeddings to {args.output} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
