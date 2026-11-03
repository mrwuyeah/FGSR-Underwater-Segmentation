# Datasets

This repository contains **only split lists**. Images, annotation masks,
degraded images and checkpoints are not redistributed. Download the data from
the official sources below and arrange it under `data/`.

---

## SUIM

**Official sources**

* Dataset page: <https://irvlab.cs.umn.edu/resources/suim-dataset>
* Project page: <https://irvlab.cs.umn.edu/image-segmentation/suim>
* GitHub: <https://github.com/IRVLab/SUIM>
* Paper: M. J. Islam et al., *Semantic Segmentation of Underwater Imagery:
  Dataset and Benchmark*, IROS 2020 (<https://arxiv.org/abs/2004.01241>)

**Expected layout**

```
data/SUIM/
├── images/
│   ├── training/     # 1,525 .jpg images
│   └── validation/   #   110 .jpg images
└── annotations/
    ├── training/     # 1,525 mask files, one per image
    └── validation/   #   110 mask files
```

The split files in `splits/suim/` store paths relative to `data/SUIM/`, e.g.

```
images/training/d_r_297_.jpg annotations/training/d_r_297_.png
```

**Label format and six-class mapping**

`dataset/semi.py` reads masks with `PIL.Image.open(...)` and uses the raw pixel
value as the class index. Masks must therefore be single-channel palette PNGs
with values `0`–`5`:

| Index | Class | Official SUIM 3-bit code |
|:---:|---|---|
| 0 | BW — background water | `000` (black) |
| 1 | HD — human divers | `001` (blue) |
| 2 | WR — wrecks / ruins | `011` (cyan) |
| 3 | RO — robots / AUVs | `100` (red) |
| 4 | RI — reefs / invertebrates | `101` (pink) |
| 5 | FV — fish / vertebrates | `110` (yellow) |

The official SUIM release also contains **PF** (`010`, aquatic plants) and
**SR** (`111`, sea-floor / rocks). These two classes are outside the six-class
protocol used in the paper and are not evaluated here. The official masks are
RGB-coloured; they must be converted to palette-index PNGs using the mapping
above before training. The code does not perform this conversion and does not
support masks containing other label values.

---

## DUT-USEG

**Official sources**

* GitHub: <https://github.com/baxiyi/DUT-USEG>
* Paper: Z. Ma et al., *An Underwater Image Semantic Segmentation Method
  Focusing on Boundaries and a Real Underwater Scene Semantic Segmentation
  Dataset*, ChinaMM 2021 (<https://arxiv.org/abs/2108.11727>)

The download link is maintained in the official repository README (currently
a Baidu Cloud link). Follow the usage terms stated there.

**Expected layout**

```
data/DUT/
├── images/
│   ├── training/
│   └── validation/     # 107 images used by splits/dut/val.txt
└── annotations/
    ├── training/
    └── validation/
```

**Label format**

Masks are single-channel palette PNGs with values `0`–`4`:

| Index | Class |
|:---:|---|
| 0 | BW — background water |
| 1 | SC — sea cucumber |
| 2 | SU — sea urchin |
| 3 | SL — scallop |
| 4 | SF — starfish |

This ordering is the one used by `util/classes.py` (`dut` entry) and by the
paper's per-class DUT table.

---

## Splits

| Path | Content |
|---|---|
| `splits/suim/95/` | 1/16 supervision: 95 labeled, 1,430 unlabeled |
| `splits/suim/190/` | 1/8 supervision: 190 labeled, 1,335 unlabeled |
| `splits/suim/381/` | 1/4 supervision: 381 labeled, 1,144 unlabeled |
| `splits/suim/762/` | 1/2 supervision: 762 labeled, 763 unlabeled |
| `splits/suim/full/labeled.txt` | all 1,525 training images |
| `splits/suim/val.txt` | 110 validation images |
| `splits/dut/345/` | 345 labeled, 1,035 unlabeled |
| `splits/dut/val.txt` | 107 validation images |

Each line contains an image path and the corresponding mask path, separated by
a space. Paths are relative to the dataset root (`data/SUIM` or `data/DUT`).

---

## SUIM-Hard

SUIM-Hard is generated locally from the clean SUIM validation split:

```bash
python scripts/generate_suim_hard.py
```

The script applies four degradations at three severity levels (12 subsets in
total) and writes:

```
data/SUIM-Hard_test/
├── degradation_params.json
├── images/
│   ├── lowlight_mild/ ... blur_severe/
└── masks/
    ├── lowlight_mild/ ... blur_severe/
```

Ground-truth masks are copied unchanged from `data/SUIM/annotations/validation/`.
The generated images are derived from the SUIM release and are subject to the
original dataset's terms; they are not included in this repository.
