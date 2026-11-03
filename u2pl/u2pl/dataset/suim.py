import copy
import math
import os
import os.path
import random

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from . import augmentation as psp_trsform
from .base import BaseDataset


class suim_dset(BaseDataset):
    def __init__(self, data_root, data_list, trs_form, seed, n_sup, split="val"):
        super(suim_dset, self).__init__(data_list)
        self.data_root = data_root
        self.transform = trs_form
        random.seed(seed)
        if len(self.list_sample) >= n_sup and split == "train":
            self.list_sample_new = random.sample(self.list_sample, n_sup)
        elif len(self.list_sample) < n_sup and split == "train":
            num_repeat = math.ceil(n_sup / len(self.list_sample))
            self.list_sample = self.list_sample * num_repeat
            self.list_sample_new = random.sample(self.list_sample, n_sup)
        else:
            self.list_sample_new = self.list_sample

    def label_loader(self, path):
        """Load label as palette-mode PIL, preserving palette indices (0-5)."""
        with open(path, "rb") as f:
            img = Image.open(f)
            img.load()  # force load while file handle is open
            return img

    def __getitem__(self, index):
        image_path = os.path.join(self.data_root, self.list_sample_new[index][0])
        label_path = os.path.join(self.data_root, self.list_sample_new[index][1])
        image = self.img_loader(image_path, "RGB")
        label = self.label_loader(label_path)
        image, label = self.transform(image, label)
        return image[0], label[0, 0].long()

    def __len__(self):
        return len(self.list_sample_new)


def build_transfrom(cfg):
    trs_form = []
    mean, std, ignore_label = cfg["mean"], cfg["std"], cfg["ignore_label"]

    # ── 阶段1: 像素值变换 (0-255 空间) ──
    trs_form.append(psp_trsform.ToTensor())
    if cfg.get("color_jitter", {}).get("enable", False):
        trs_form.append(psp_trsform.PhotoMetricDistortion(
            brightness_delta=cfg["color_jitter"].get("brightness_delta", 32),
            contrast_range=cfg["color_jitter"].get("contrast_range", (0.5, 1.5)),
            saturation_range=cfg["color_jitter"].get("saturation_range", (0.5, 1.5)),
            hue_delta=cfg["color_jitter"].get("hue_delta", 18),
        ))

    # ── 阶段2: 空间变换 ──
    if cfg.get("resize", False):
        trs_form.append(psp_trsform.Resize(cfg["resize"]))
    if cfg.get("rand_resize", False):
        trs_form.append(psp_trsform.RandResize(cfg["rand_resize"]))
    if cfg.get("rand_rotation", False):
        rand_rotation = cfg["rand_rotation"]
        trs_form.append(
            psp_trsform.RandRotate(rand_rotation, ignore_label=ignore_label)
        )
    if cfg.get("GaussianBlur", False) and cfg["GaussianBlur"]:
        trs_form.append(psp_trsform.RandomGaussianBlur())
    if cfg.get("flip", False) and cfg.get("flip"):
        trs_form.append(psp_trsform.RandomHorizontalFlip())
    if cfg.get("crop", False):
        crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
        trs_form.append(
            psp_trsform.Crop(crop_size, crop_type=crop_type, ignore_label=ignore_label)
        )
    if cfg.get("cutout", False):
        n_holes, length = cfg["cutout"]["n_holes"], cfg["cutout"]["length"]
        trs_form.append(psp_trsform.Cutout(n_holes=n_holes, length=length))
    if cfg.get("cutmix", False):
        n_holes, prop_range = cfg["cutmix"]["n_holes"], cfg["cutmix"]["prop_range"]
        trs_form.append(psp_trsform.Cutmix(prop_range=prop_range, n_holes=n_holes))

    # ── 阶段3: 归一化 (保持在最后, 与 mmseg 惯例一致) ──
    trs_form.append(psp_trsform.Normalize(mean=mean, std=std))

    return psp_trsform.Compose(trs_form)


def build_suimloader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]
    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = cfg.get("n_sup", 1525)

    trs_form = build_transfrom(cfg)
    dset = suim_dset(cfg["data_root"], cfg["data_list"], trs_form, seed, n_sup, split)

    sample = DistributedSampler(dset)
    drop_last = True if split == "train" else False
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sample,
        shuffle=False,
        pin_memory=False,
        drop_last=drop_last,
    )
    return loader


def build_suim_semi_loader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]
    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = 1525 - cfg.get("n_sup", 1525)

    trs_form = build_transfrom(cfg)
    trs_form_unsup = build_transfrom(cfg)
    dset = suim_dset(cfg["data_root"], cfg["data_list"], trs_form, seed, n_sup, split)

    if split == "val":
        sample = DistributedSampler(dset)
        loader = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample,
            shuffle=False,
            pin_memory=True,
        )
        return loader
    else:
        data_list_unsup = cfg["data_list"].replace("labeled.txt", "unlabeled.txt")
        dset_unsup = suim_dset(
            cfg["data_root"], data_list_unsup, trs_form_unsup, seed, n_sup, split
        )

        sample_sup = DistributedSampler(dset)
        loader_sup = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample_sup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        sample_unsup = DistributedSampler(dset_unsup)
        loader_unsup = DataLoader(
            dset_unsup,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample_unsup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
        return loader_sup, loader_unsup
