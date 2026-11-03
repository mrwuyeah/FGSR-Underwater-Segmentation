#!/usr/bin/env python3
"""Evaluate a checkpoint on the clean SUIM / DUT validation split."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # pretrained/ and splits/ are relative to the repo root

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.classes import CLASSES
from util.utils import intersectionAndUnion


def load_model(config_path, checkpoint_path, device):
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    model = DeepLabV3Plus(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
    model.to(device).eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/*.yaml")
    ap.add_argument("--checkpoint", required=True, help="checkpoint path")
    ap.add_argument("--dataset", default="suim", choices=["suim", "dut"])
    ap.add_argument("--data-root", default=None, help="default: data/SUIM or data/DUT")
    ap.add_argument("--output", default=None, help="optional JSON output path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data_root = args.data_root or f"data/{args.dataset.upper()}"
    if args.dataset == "dut":
        data_root = args.data_root or "data/DUT"
    if not (ROOT / data_root).exists():
        raise FileNotFoundError(
            f"Dataset root '{data_root}' not found. See DATASETS.md for the expected layout."
        )

    device = torch.device(args.device)
    model, cfg = load_model(args.config, args.checkpoint, device)
    valset = SemiDataset(args.dataset, data_root, "val")
    loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)

    inter = np.zeros(cfg["nclass"], dtype=np.float64)
    union = np.zeros(cfg["nclass"], dtype=np.float64)
    with torch.no_grad():
        for img, mask, _ in loader:
            pred = model(img.to(device)).argmax(dim=1).cpu().numpy()
            i, u, _ = intersectionAndUnion(pred, mask.numpy(), cfg["nclass"], 255)
            inter += i
            union += u

    iou = inter / (union + 1e-10) * 100.0
    miou = float(np.mean(iou))
    names = CLASSES[args.dataset]
    print(f"mIoU: {miou:.2f}")
    for name, value in zip(names, iou):
        print(f"  {name:28s} {value:.2f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump(
            {"mIoU": round(miou, 2), "per_class": {n: round(float(v), 2) for n, v in zip(names, iou)}},
            open(args.output, "w"),
            indent=2,
        )
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
