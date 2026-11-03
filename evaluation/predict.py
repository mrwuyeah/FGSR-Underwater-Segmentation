#!/usr/bin/env python3
"""Run inference with a trained checkpoint and save class-index PNG masks."""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", default=None, help="single image path")
    ap.add_argument("--list", default=None, help="text file with one image path per line")
    ap.add_argument("--output-dir", default="predictions")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not args.image and not args.list:
        ap.error("provide --image or --list")
    images = [args.image] if args.image else [l.strip() for l in open(args.list) if l.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, cfg = load_model(args.config, args.checkpoint, device)

    with torch.no_grad():
        for image_path in images:
            img = Image.open(image_path).convert("RGB")
            arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            tensor = torch.from_numpy((arr - MEAN) / STD).unsqueeze(0).to(device)
            pred = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            out_path = out_dir / (Path(image_path).stem + ".png")
            Image.fromarray(pred).save(out_path)
            print(f"{image_path} -> {out_path}  (classes: {sorted(set(pred.flatten().tolist()))})")

    print(f"model nclass={cfg['nclass']}; masks store class indices 0..{cfg['nclass']-1}")


if __name__ == "__main__":
    main()
