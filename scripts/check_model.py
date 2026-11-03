#!/usr/bin/env python3
"""Shape / range check for the FGSR and MSD modules.

This does not need the datasets or a GPU:

    python scripts/check_model.py --device cpu

Add ``--backward`` to also run a small backward pass. If
``pretrained/resnet50.pth`` exists, the full DeepLabV3+ model is built and
checked as well.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.fgsr_head import FGSR
from model.semseg.aspp_maa_hybrid import ASPPMAADecoder
from model.semseg.resnet_maa_decoder import ResNetMAADecoder


def make_feats(batch_size, height, width, device="cpu"):
    channels = [256, 512, 1024, 2048]
    strides = [4, 8, 16, 32]
    return [
        torch.randn(batch_size, c, height // s, width // s, device=device)
        for c, s in zip(channels, strides)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--nclass", type=int, default=6)
    parser.add_argument("--backward", action="store_true",
                        help="also run a small backward pass (64x64 input)")
    args = parser.parse_args()

    device = torch.device(args.device)
    feats = make_feats(args.batch_size, args.height, args.width, device=device)
    h, w = feats[0].shape[-2:]

    fgsr = FGSR(embed_dim=128, r_min=0.5).to(device).eval()
    msd = ASPPMAADecoder(num_classes=args.nclass, maa_dim=128).to(device).eval()
    maa = ResNetMAADecoder(num_classes=args.nclass, embed_dim=128).to(device).eval()

    with torch.no_grad():
        R = fgsr(feats)
        logits = msd(feats)
        logits_maa = maa(feats)

    assert R.shape == (args.batch_size, 1, h, w), R.shape
    assert R.min().item() >= 0.5 - 1e-6, R.min().item()
    assert R.max().item() <= 1.0 + 1e-6, R.max().item()
    print(f"FGSR: R {tuple(R.shape)}, range [{R.min().item():.3f}, {R.max().item():.3f}]")

    expected = (args.batch_size, args.nclass, h, w)
    assert logits.shape == expected, logits.shape
    print(f"MSD: logits {tuple(logits.shape)}, alpha={msd.alpha.item():.3f}")

    assert logits_maa.shape[0] == args.batch_size and logits_maa.shape[1] == args.nclass
    print(f"MAA decoder: logits {tuple(logits_maa.shape)}")

    if args.backward:
        # Use a small input so the check also runs on memory-constrained CPUs.
        small = make_feats(2, 64, 64, device=device)
        fgsr.train(); msd.train(); maa.train()
        R = fgsr(small)
        out = msd(small)
        out_maa = maa(small)
        (out.mean() + R.mean() + out_maa.mean()).backward()
        for name, module in [("FGSR", fgsr), ("MSD", msd), ("MAA", maa)]:
            assert all(p.grad is not None for p in module.parameters()), \
                f"{name}: some parameters received no gradient"
        print("Backward pass: OK")

    # --- full model (needs pretrained/resnet50.pth) ---
    backbone_path = ROOT / "pretrained" / "resnet50.pth"
    if backbone_path.exists():
        from model.semseg.deeplabv3plus import DeepLabV3Plus

        cfg = {
            "dataset": "suim",
            "nclass": args.nclass,
            "crop_size": [480, 640],
            "backbone": "resnet50",
            "replace_stride_with_dilation": [False, True, True],
            "dilations": [6, 12, 18],
            "use_fgsr": True,
            "r_min": 0.5,
            "use_hybrid_decoder": True,
        }
        model = DeepLabV3Plus(cfg).to(device).eval()
        x = torch.randn(1, 3, 480, 640, device=device)
        with torch.no_grad():
            out, rel = model(x, return_reliability=True)
        assert out.shape == (1, args.nclass, 480, 640), out.shape
        assert rel.shape == (1, 1, 480, 640), rel.shape
        print(f"Full model: logits {tuple(out.shape)}, R {tuple(rel.shape)}")
    else:
        print("Skipping full-model check: pretrained/resnet50.pth not found.")

    print("All checks passed.")


if __name__ == "__main__":
    main()
