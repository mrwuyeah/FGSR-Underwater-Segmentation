# FGSR-Underwater-Segmentation

Repository: <https://github.com/mrwuyeah/FGSR-Underwater-Segmentation>

Code release for the manuscript

> **Feature-Guided Pseudo-Label Reweighting for Degradation-Robust Semi-Supervised Underwater Semantic Segmentation**

submitted to the *Journal of Electronic Imaging*.

This repository contains the implementation of **FGSR** (Feature-Guided Spatial
Reweighting) and **MSD** (Multi-Scale auxiliary Decoder), together with the
UniMatch-based training pipeline, the U2PL cross-framework port, the SUIM/DUT
data splits, and the SUIM-Hard robustness evaluation protocol.

> **Naming note.** The paper and the code both use **FGSR** and **MSD**. The
> MSD decoder keeps the internal alias `MAA` / `hybrid` (config key
> `use_hybrid_decoder`; file `model/semseg/aspp_maa_hybrid.py`). Both names
> refer to the same decoder.

---

## Method at a glance

* **Baseline** — UniMatch weak-to-strong pseudo-label supervision on top of
  DeepLabV3+ (ResNet-50) with confidence threshold `tau`.
* **FGSR** — a train-time branch that projects the four ResNet-50 feature
  levels `c1`–`c4` to 128 channels, upsamples and concatenates them, and
  predicts a pixel-wise weight map `W in [r_min, 1]`. `W` multiplies the
  per-pixel unsupervised loss. At inference the branch is unused.
* **MSD** — a gated multi-scale residual branch added to the ASPP decoder.
  The decoder output is `A + alpha * M`, where `alpha` is a learnable scalar
  initialised to `0.1`.
* **SUIM-Hard** — 12 degraded copies of the SUIM validation set (low-light,
  turbidity, colour cast and blur at three severity levels) used only for
  post-hoc robustness evaluation.

Paper configurations:

| Setting | Config | Labeled split |
|---|---|---|
| UniMatch baseline (`tau = 0.9`) | `configs/suim_conf0.9.yaml` | `splits/suim/381` |
| UniMatch + FGSR | `configs/suim_fgsr.yaml` | `splits/suim/381` |
| UniMatch + MSD | `configs/suim_hybrid.yaml` | `splits/suim/381` |
| UniMatch + FGSR + MSD (full) | `configs/suim_hybrid_fgsr.yaml` | `splits/suim/381` |
| DUT-USEG baseline | `configs/dut_baseline.yaml` | `splits/dut/345` |
| DUT-USEG full | `configs/dut_hybrid_fgsr.yaml` | `splits/dut/345` |
| Fully supervised reference | `configs/suim_sup_full.yaml` | `splits/suim/full` |

Additional configs in `configs/` cover the label-ratio study (1/16, 1/8, 1/2),
the `r_min` sensitivity analysis, soft-confidence weighting, and a
feature-detach diagnostic retained for internal completeness. They are kept
because they are small and document the ablation settings.

---

## Repository layout

```
.
├── configs/              UniMatch experiment configs
├── dataset/              SemiDataset and augmentation transforms
├── model/
│   ├── backbone/         ResNet / Xception backbones
│   ├── semseg/
│   │   ├── deeplabv3plus.py       DeepLabV3+, model factory
│   │   ├── aspp_maa_hybrid.py     MSD ("ASPP-MAA hybrid") decoder
│   │   └── resnet_maa_decoder.py  alternative standalone MAA decoder
│   └── fgsr_head.py      FGSR weight-map branch (class FGSR)
├── util/                 metrics, logging, OHEM, distributed helpers
├── splits/               SUIM and DUT-USEG split lists (no images)
├── evaluation/           clean-data evaluation, SUIM-Hard evaluation, inference
├── scripts/              training launchers, SUIM-Hard generator, model check
├── u2pl/                 U2PL framework with the FGSR port
├── data/                 dataset placeholder (.gitkeep)
├── pretrained/           backbone placeholder (.gitkeep)
├── checkpoints/          output placeholder (.gitkeep)
├── licenses/             third-party licence texts
├── DATASETS.md           dataset acquisition and layout
├── requirements.txt
└── LICENSE
```

`unimatch.py` and `supervised.py` are the two UniMatch entry points; the U2PL
entry points are `u2pl/train_semi.py`, `u2pl/train_sup.py`, `u2pl/eval.py` and
`u2pl/infer.py`.

---

## Environment

```bash
conda create -n fgsr python=3.10 -y
conda activate fgsr
pip install -r requirements.txt
```

Install a PyTorch build matching your CUDA driver, for example:

```bash
pip install torch torchvision
```

The paper used `torch==1.12.1 / torchvision==0.13.1`; this release targets
PyTorch 2.x (the evaluation utilities pass `weights_only=False` to
`torch.load`). A single GPU is sufficient to run the code; the paper used
4 GPUs with `batch_size: 2` per GPU (total batch 8).

### Pretrained backbones

* **UniMatch experiments** — download the ResNet-50 ImageNet checkpoint from
  the UniMatch repository
  ([Google Drive](https://drive.google.com/file/d/1mqUrqFvTQ0k5QEotk4oiOFyP6B9dVZXS/view?usp=sharing))
  and place it at `pretrained/resnet50.pth`. The loader in
  `model/backbone/resnet.py` reads this path relative to the repository root.
* **U2PL experiments** — the U2PL ResNet-50 encoder reads
  `pretrained/resnet50_v1c.pth` (the ImageNet ResNet-50 checkpoint used by the
  U2PL release). This file is only needed for `u2pl/`.

Both files are excluded by `.gitignore`; the `pretrained/` directory ships
empty.

### Data

Download SUIM and DUT-USEG from their official sources and arrange them as
described in **[DATASETS.md](DATASETS.md)**. No images, masks, degraded images
or checkpoints are distributed with this repository.

---

## Training

All commands are run from the repository root.

### UniMatch (main framework)

```bash
# UniMatch baseline, tau = 0.9, SUIM 1/4 split (381 labeled)
bash scripts/train_suim.sh configs/suim_conf0.9.yaml 381 suim_baseline 4 29501

# FGSR only
bash scripts/train_suim.sh configs/suim_fgsr.yaml 381 suim_fgsr 4 29502

# MSD only
bash scripts/train_suim.sh configs/suim_hybrid.yaml 381 suim_msd 4 29503

# FGSR + MSD (full model)
bash scripts/train_suim.sh configs/suim_hybrid_fgsr.yaml 381 suim_full 4 29504
```

The launcher is equivalent to:

```bash
torchrun --nproc_per_node=4 --master_port=29504 unimatch.py \
  --config configs/suim_hybrid_fgsr.yaml \
  --labeled-id-path splits/suim/381/labeled.txt \
  --unlabeled-id-path splits/suim/381/unlabeled.txt \
  --save-path checkpoints/suim_full --port 29504
```

Use a different split directory for the label-ratio study, e.g.
`splits/suim/95`, `splits/suim/190`, `splits/suim/762`.

### DUT-USEG

```bash
bash scripts/train_dut.sh configs/dut_baseline.yaml dut_baseline 4 29505
bash scripts/train_dut.sh configs/dut_hybrid_fgsr.yaml dut_full 4 29506
```

### Fully supervised reference

```bash
bash scripts/train_suim_supervised.sh configs/suim_sup_full.yaml 4 29507
```

### U2PL (cross-framework)

```bash
bash scripts/train_u2pl_suim.sh u2pl/configs/suim_1_4_baseline.yaml 4 29508
bash scripts/train_u2pl_suim.sh u2pl/configs/suim_1_4_fgsr.yaml     4 29509
```

The U2PL port lives in `u2pl/u2pl/`; the FGSR branch for U2PL is
`u2pl/u2pl/models/fgsr_head.py` and is fused into the unsupervised loss in
`u2pl/u2pl/utils/loss_helper.py`.

### CPS

The CPS numbers in the paper were obtained with the public CPS codebase. The
CPS SUIM adaptation, configs and logs are **not** part of this release; see
`cps/README.md`.

---

## Evaluation

Clean SUIM / DUT validation:

```bash
python evaluation/eval_clean.py \
  --config configs/suim_hybrid_fgsr.yaml \
  --checkpoint checkpoints/suim_full/best.pth \
  --dataset suim \
  --output results/suim_full.json
```

For deployment-style inference the FGSR branch can be omitted: use
`configs/suim_hybrid.yaml` (MSD only) with the full checkpoint.
`eval_clean.py` loads with `strict=False`, so the unused FGSR weights are
ignored and inference pays no FGSR cost.

Single-image / image-list inference:

```bash
python evaluation/predict.py \
  --config configs/suim_hybrid.yaml \
  --checkpoint checkpoints/suim_full/best.pth \
  --image path/to/image.jpg \
  --output-dir predictions/
```

## SUIM-Hard robustness protocol

```bash
# Generate the 12 degraded validation subsets from data/SUIM
python scripts/generate_suim_hard.py

# Evaluate a checkpoint on all 12 subsets
python evaluation/eval_suim_hard.py \
  --config configs/suim_hybrid_fgsr.yaml \
  --checkpoint checkpoints/suim_full/best.pth \
  --data-root data/SUIM-Hard_test \
  --output results/suim_full_suimhard.json
```

The generator writes `data/SUIM-Hard_test/images/<subset>/` and
`data/SUIM-Hard_test/masks/<subset>/`. Ground-truth masks are copied from the
SUIM validation masks; no degraded images are committed.

## Quick model check

```bash
python scripts/check_model.py
```

This instantiates FGSR and MSD on random feature maps, checks output shapes and
the `[r_min, 1]` range, runs a backward pass, and (if
`pretrained/resnet50.pth` exists) builds the full DeepLabV3+ model.

---

## Licence and acknowledgements

This repository builds on:

* [UniMatch](https://github.com/LiheYoung/UniMatch) (MIT licence),
* [U2PL](https://github.com/Haochen-Wang409/U2PL) (Apache-2.0 licence),
* the CPS codebase (MIT licence, not included).

The corresponding licence texts are kept in `licenses/`. The root `LICENSE`
is the Apache-2.0 licence inherited from the project. Please retain the
upstream copyright notices when redistributing this code.

## Contact

Questions about the paper or code: `jjwu@whut.edu.cn`.
