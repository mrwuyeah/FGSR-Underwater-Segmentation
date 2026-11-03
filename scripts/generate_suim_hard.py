#!/usr/bin/env python3
"""Generate SUIM-Hard: 12 degradation subsets of the SUIM validation split.

The clean SUIM validation images are read from ``data/SUIM`` and written to
``data/SUIM-Hard_test``. Ground-truth masks are copied unchanged from SUIM.

    python scripts/generate_suim_hard.py
"""

import cv2
import json
import numpy as np
import os
import shutil
import time
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
VAL_FILE = PROJ_ROOT / "splits/suim/val.txt"
DATA_ROOT = PROJ_ROOT / "data/SUIM"
OUT_ROOT = PROJ_ROOT / "data/SUIM-Hard_test"

def to_float(x): return x.astype(np.float32) / 255.0
def to_uint8(x): return np.clip(x * 255, 0, 255).astype(np.uint8)

PARAMS = {
    "lowlight_mild":   {"type": "lowlight",  "gamma": 1.4,  "alpha": 0.90},
    "lowlight_medium": {"type": "lowlight",  "gamma": 1.8,  "alpha": 0.75},
    "lowlight_severe": {"type": "lowlight",  "gamma": 2.3,  "alpha": 0.60},
    "turbid_mild":     {"type": "turbid",    "t": 0.75,  "A": [0.65, 0.85, 0.90]},
    "turbid_medium":   {"type": "turbid",    "t": 0.55,  "A": [0.65, 0.85, 0.90]},
    "turbid_severe":   {"type": "turbid",    "t": 0.35,  "A": [0.65, 0.85, 0.90]},
    "colorcast_mild":  {"type": "colorcast", "scale": [0.85, 1.05, 1.10]},
    "colorcast_medium":{"type": "colorcast", "scale": [0.65, 1.10, 1.25]},
    "colorcast_severe":{"type": "colorcast", "scale": [0.45, 1.15, 1.35]},
    "blur_mild":       {"type": "blur",      "k": 3,  "sigma": 0.8},
    "blur_medium":     {"type": "blur",      "k": 5,  "sigma": 1.4},
    "blur_severe":     {"type": "blur",      "k": 7,  "sigma": 2.0},
}

def main():
    if not VAL_FILE.exists():
        raise FileNotFoundError(
            f"Split file not found: {VAL_FILE}. Run from the repository root."
        )
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"SUIM root not found: {DATA_ROOT}. See DATASETS.md for the layout."
        )

    with open(VAL_FILE) as f:
        val_pairs = [line.strip().split() for line in f if line.strip()]

    print(f"Val images: {len(val_pairs)}")
    start = time.time()
    count = 0

    for sub_name, p in PARAMS.items():
        img_dir = OUT_ROOT / "images" / sub_name
        mask_dir = OUT_ROOT / "masks" / sub_name
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for img_rel, mask_rel in val_pairs:
            img = cv2.imread(os.path.join(DATA_ROOT, img_rel))
            if img is None:
                continue

            t = p["type"]
            if t == "lowlight":
                out = to_uint8(p["alpha"] * to_float(img) ** p["gamma"])
            elif t == "turbid":
                A = np.array([p["A"][2], p["A"][1], p["A"][0]], dtype=np.float32).reshape(1, 1, 3)
                out = to_uint8(to_float(img) * p["t"] + A * (1 - p["t"]))
            elif t == "colorcast":
                s = np.array([p["scale"][2], p["scale"][1], p["scale"][0]], dtype=np.float32).reshape(1, 1, 3)
                out = to_uint8(to_float(img) * s)
            elif t == "blur":
                out = cv2.GaussianBlur(img, (p["k"], p["k"]), p["sigma"])
            else:
                raise ValueError(f"Unknown degradation type: {t}")

            fname = Path(img_rel).name
            cv2.imwrite(str(img_dir / fname), out)

            # Copy mask
            mask_path = os.path.join(DATA_ROOT, mask_rel)
            if os.path.exists(mask_path):
                shutil.copy2(mask_path, mask_dir / Path(mask_rel).name)

            count += 1

        print(f"  {sub_name:20s}  done")

    # Write params
    with open(OUT_ROOT / "degradation_params.json", "w") as f:
        json.dump(PARAMS, f, indent=2)

    print(f"\nTotal: {count} images, {time.time() - start:.1f}s")
    print(f"Output: {OUT_ROOT.resolve()}")

if __name__ == "__main__":
    main()
