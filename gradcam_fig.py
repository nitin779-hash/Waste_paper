"""Grad-CAM + error-analysis figure (300 dpi) for the seed-0 class-weighted model on its held-out test split."""
import json, sys
import numpy as np
import torch, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/claude/exp/code")
from common import *

torch.set_num_threads(2)
x, y, files = load_images(224)
TE = np.load(f"{OUT}/testidx_seed0.npy")
base = build_backbone(); lower = lower_of(base); top = Top(base)
top.load_state_dict(torch.load(f"{OUT}/top_seed0_cw.pt")); top.set_mode(False)

def cam(i):
    xi = normalize(torch.from_numpy(x[i:i + 1]))
    with torch.no_grad():
        z = lower(xi)
    z.requires_grad_(False)
    feat = top.conv(top.blocks(z)).detach().requires_grad_(True)
    logits = top.head(top.pool(feat).flatten(1))
    pred = int(logits.argmax(1))
    logits[0, pred].backward()
    w = feat.grad.mean((2, 3), keepdim=True)
    c = F.relu((w * feat).sum(1, keepdim=True))
    c = F.interpolate(c, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    c = (c / (c.max() + 1e-8)).detach().numpy()
    return pred, c

preds = {}
for i in TE:
    p, _ = cam(i); preds[i] = p
want = [(3, 0), (1, 4), (4, 1), (5, 3), (0, 3), (2, 1)]  # (true, predicted) pairs of interest
shown = []
for t, p in want:
    cand = [i for i in TE if y[i] == t and preds[i] == p]
    if cand:
        shown.append(cand[0])
correct = [next(i for i in TE if y[i] == c and preds[i] == c) for c in (3, 1)]
items = shown[:6] + correct
fig, ax = plt.subplots(2, len(items), figsize=(1.9 * len(items), 4.3), dpi=300)
for k, i in enumerate(items):
    p, c = cam(i)
    ax[0, k].imshow(x[i].transpose(1, 2, 0)); ax[0, k].axis("off")
    ax[0, k].set_title(f"true: {CLASSES[y[i]]}\npred: {CLASSES[p]}", fontsize=6.5, color="green" if p == y[i] else "firebrick")
    ax[1, k].imshow(x[i].transpose(1, 2, 0)); ax[1, k].imshow(c, cmap="jet", alpha=0.45); ax[1, k].axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_gradcam_300dpi.png", dpi=300, bbox_inches="tight")
print("saved", len(items), "examples")
