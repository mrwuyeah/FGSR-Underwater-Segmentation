# U2PL + FGSR

This directory is the cross-framework port of FGSR to
[U2PL](https://github.com/Haochen-Wang409/U2PL) (Apache-2.0). The upstream
README is kept as `README_upstream.md`.

FGSR enters U2PL in three places:

| File | Role |
|---|---|
| `u2pl/models/fgsr_head.py` | FGSR weight-map branch (class `FGSR`) |
| `u2pl/models/model_helper.py` | builds the branch when `net.use_fgsr: True` |
| `u2pl/utils/loss_helper.py` | multiplies the unsupervised loss by the FGSR weight map (`fgsr_R`) |
| `train_semi.py` | requests the weight map and passes it to the loss |

## Data and configs

The two configs in `configs/` use paths relative to the config directory:

* `dataset.train/val.data_root: ../../data/SUIM`
* `dataset.train/val.data_list: ../splits/suim/...`
* `saver.snapshot_dir: ../../checkpoints/u2pl_1_4_*`

`train_semi.py` resolves all of these against the config directory, so the
model can be launched from either the repository root or `u2pl/`.

## Training

From the repository root:

```bash
bash scripts/train_u2pl_suim.sh u2pl/configs/suim_1_4_baseline.yaml 4 29508
bash scripts/train_u2pl_suim.sh u2pl/configs/suim_1_4_fgsr.yaml     4 29509
```

Or directly:

```bash
cd u2pl
torchrun --nproc_per_node=4 --master_port=29509 train_semi.py \
    --config configs/suim_1_4_fgsr.yaml --port 29509
```

The U2PL ResNet-50 encoder loads `pretrained/resnet50_v1c.pth` from the
repository root. See the main `README.md` and `DATASETS.md`.

## Evaluation / inference

`eval.py` and `infer.py` are the upstream U2PL entry points. Pass the same
config and the checkpoint written to `checkpoints/u2pl_1_4_fgsr/`:

```bash
cd u2pl
python eval.py --config configs/suim_1_4_fgsr.yaml \
    --model_path ../checkpoints/u2pl_1_4_fgsr/ckpt_best.pth
```
