"""Shared utilities: data loading, model construction, augmentation, metrics."""
import importlib.util
import json
import math
import os
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = "/home/claude/exp"
DATA = f"{ROOT}/data"
OUT = f"{ROOT}/out"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
WEIGHTS = f"{DATA}/w/d_li14/pretrained/mobilenetv2-c5e733a8.pth"
ARCH = f"{DATA}/w/d_li14/models/imagenet/mobilenetv2.py"


def load_images(res):
    """Return uint8 array [N,3,res,res], labels [N], file list. Cached on disk."""
    cache = f"{DATA}/cache_{res}.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return z["x"], z["y"], list(z["f"])
    ex = f"{DATA}/trashnet"
    if not os.path.isdir(ex):
        os.makedirs(ex)
        with zipfile.ZipFile(f"{DATA}/dataset-resized.zip") as zf:
            for n in zf.namelist():
                if n.lower().endswith(".jpg") and ".." not in n and "__MACOSX" not in n and not os.path.basename(n).startswith("._"):
                    zf.extract(n, ex)
    xs, ys, fs = [], [], []
    for ci, c in enumerate(CLASSES):
        d = None
        for root, _, files in os.walk(ex):
            if os.path.basename(root) == c and "__MACOSX" not in root:
                d = root
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(".jpg") and not f.startswith("._"):
                im = Image.open(os.path.join(d, f)).convert("RGB").resize((res, res), Image.BILINEAR)
                xs.append(np.asarray(im).transpose(2, 0, 1))
                ys.append(ci)
                fs.append(f"{c}/{f}")
    x = np.stack(xs).astype(np.uint8)
    y = np.array(ys, dtype=np.int64)
    np.savez(cache, x=x, y=y, f=np.array(fs))
    return x, y, fs


def build_backbone():
    spec = importlib.util.spec_from_file_location("mnv2", ARCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.mobilenetv2()
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    m.load_state_dict(sd, strict=True)
    return m


# features[0]=stem, features[1..17]=blocks 0..16.  Keras "top 30 layers" ~ blocks 14-16 + final 1x1 conv
SPLIT_AT = 15  # lower = features[:15] (frozen always), top = features[15:] + conv


class Top(nn.Module):
    """Trainable part: last 3 inverted-residual blocks + 1x1 conv + GAP + head."""

    def __init__(self, base):
        super().__init__()
        self.blocks = nn.Sequential(*[base.features[i] for i in range(SPLIT_AT, len(base.features))])
        self.conv = base.conv
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Linear(1280, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 6))

    def pooled(self, z):
        return self.pool(self.conv(self.blocks(z))).flatten(1)

    def forward(self, z):
        return self.head(self.pooled(z))

    def set_mode(self, train_head=True):
        self.eval()  # BN always in inference mode (frozen statistics)
        self.head.train(train_head)


def lower_of(base):
    return nn.Sequential(*[base.features[i] for i in range(SPLIT_AT)]).eval()


def normalize(x_uint8_or_float):
    x = x_uint8_or_float
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    return (x - MEAN) / STD


def augment(x, g, rot=20.0, shift=0.10, zoom=0.15):
    """Random affine + horizontal flip on float images in [0,1]. Parameters are the ones reported in the paper."""
    B = x.size(0)
    ang = (torch.rand(B, generator=g) * 2 - 1) * math.radians(rot)
    sc = 1.0 / (1 + (torch.rand(B, generator=g) * 2 - 1) * zoom)
    tx = (torch.rand(B, generator=g) * 2 - 1) * shift * 2
    ty = (torch.rand(B, generator=g) * 2 - 1) * shift * 2
    flip = (torch.rand(B, generator=g) < 0.5).float() * 2 - 1
    c, s = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(B, 2, 3)
    theta[:, 0, 0] = c * sc * flip
    theta[:, 0, 1] = -s * sc
    theta[:, 1, 0] = s * sc * flip
    theta[:, 1, 1] = c * sc
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def metrics(y, prob):
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support, roc_auc_score)
    pred = prob.argmax(1)
    p, r, f, sup = precision_recall_fscore_support(y, pred, labels=range(6), zero_division=0)
    onehot = np.eye(6)[y]
    return dict(
        acc=float(accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, average="macro")),
        weighted_f1=float(f1_score(y, pred, average="weighted")),
        macro_p=float(p.mean()), macro_r=float(r.mean()),
        trash_f1=float(f[5]), trash_recall=float(r[5]), trash_precision=float(p[5]),
        auc_macro=float(roc_auc_score(onehot, prob, average="macro")),
        per_class=dict(precision=p.tolist(), recall=r.tolist(), f1=f.tolist(), support=sup.tolist()),
        confusion=confusion_matrix(y, pred, labels=range(6)).tolist(),
    )
