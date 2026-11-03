#!/usr/bin/env python3
"""Evaluate a checkpoint on the 12 SUIM-Hard degradation subsets."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.classes import CLASSES

SUBSETS = [
    "lowlight_mild", "lowlight_medium", "lowlight_severe",
    "turbid_mild", "turbid_medium", "turbid_severe",
    "colorcast_mild", "colorcast_medium", "colorcast_severe",
    "blur_mild", "blur_medium", "blur_severe",
]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def load_model(config_path, checkpoint_path, device):
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    model = DeepLabV3Plus(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, cfg


def evaluate_subset(model, img_dir, mask_dir, pairs, nclass, device):
    inter = np.zeros(nclass, dtype=np.float64)
    union = np.zeros(nclass, dtype=np.float64)
    with torch.no_grad():
        for img_rel, mask_rel in pairs:
            img_path = img_dir / Path(img_rel).name
            mask_path = mask_dir / Path(mask_rel).name
            if not img_path.exists() or not mask_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            mask = np.array(Image.open(mask_path), dtype=np.int64)
            arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            tensor = torch.from_numpy((arr - MEAN) / STD).unsqueeze(0).to(device)
            pred = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy()
            for c in range(nclass):
                p = pred == c
                g = mask == c
                inter[c] += np.logical_and(p, g).sum()
                union[c] += np.logical_or(p, g).sum()
    return inter / np.maximum(union, 1) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default="data/SUIM-Hard_test")
    ap.add_argument("--split", default="splits/suim/val.txt")
    ap.add_argument("--output", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(
            f"SUIM-Hard root '{data_root}' not found. Run scripts/generate_suim_hard.py "
            "after downloading SUIM (see DATASETS.md)."
        )
    split_path = Path(args.split)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file '{split_path}' not found.")
    pairs = [line.strip().split() for line in open(split_path) if line.strip()]

    device = torch.device(args.device)
    model, cfg = load_model(args.config, args.checkpoint, device)
    nclass = cfg["nclass"]
    names = CLASSES["suim"]

    results = {}
    for subset in SUBSETS:
        iou = evaluate_subset(
            model, data_root / "images" / subset, data_root / "masks" / subset, pairs, nclass, device
        )
        results[subset] = {
            "mIoU": round(float(iou.mean()), 2),
            **{name: round(float(v), 2) for name, v in zip(names, iou)},
        }
        print(f"{subset:20s} mIoU={results[subset]['mIoU']:.2f}")

    mious = [results[s]["mIoU"] for s in SUBSETS]
    results["Avg-Hard"] = round(float(np.mean(mious)), 2)
    print(f"{'Avg-Hard':20s} mIoU={results['Avg-Hard']:.2f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.output, "w"), indent=2)
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
