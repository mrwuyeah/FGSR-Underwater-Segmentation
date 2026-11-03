import random

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms


def crop(img, mask, size, ignore_value=255):
    w, h = img.size
    if isinstance(size, (tuple, list)):
        crop_h, crop_w = size
    else:
        crop_h = crop_w = size
    padw = crop_w - w if w < crop_w else 0
    padh = crop_h - h if h < crop_h else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    x = random.randint(0, w - crop_w)
    y = random.randint(0, h - crop_h)
    img = img.crop((x, y, x + crop_w, y + crop_h))
    mask = mask.crop((x, y, x + crop_w, y + crop_h))

    return img, mask


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def resize(img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    if isinstance(img_size, (tuple, list)):
        h, w = img_size
    else:
        h = w = img_size
    mask = torch.zeros(h, w)
    if random.random() > p:
        return mask

    area = np.random.uniform(size_min, size_max) * h * w
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(area / ratio))
        cutmix_h = int(np.sqrt(area * ratio))
        x = np.random.randint(0, max(w - cutmix_w + 1, 1))
        y = np.random.randint(0, max(h - cutmix_h + 1, 1))

        if x + cutmix_w <= w and y + cutmix_h <= h:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask
