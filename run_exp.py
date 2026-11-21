"""Leakage-free multi-seed ablation of two-phase MobileNetV2 transfer learning on TrashNet.

Protocol (fixed before looking at any test result):
  * per seed: stratified 70/15/15 train/val/test split (different split for every seed)
  * Phase 1: head only, frozen backbone, un-augmented cached features, Adam 1e-3, 10 epochs, batch 32
  * Phase 2: unfreeze blocks 14-16 + final 1x1 conv (~ top-30 Keras layers), Adam lr2, E2 epochs, batch 32,
             augmentation re-drawn every epoch, BatchNorm statistics frozen
  * NO early stopping / checkpoint selection: the last epoch is evaluated once on val and test
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CLASSES, OUT, Top, augment, build_backbone, load_images, lower_of, metrics, normalize)

ap = argparse.ArgumentParser()
ap.add_argument("--res", type=int, default=224)
ap.add_argument("--e1", type=int, default=10)
ap.add_argument("--e2", type=int, default=10)
ap.add_argument("--lr2", type=float, default=1e-5)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--variants", default="ce,cw,focal,os")
ap.add_argument("--tag", default="main")
ap.add_argument("--save_models", default="0")
args = ap.parse_args()
torch.set_num_threads(2)
os.makedirs(OUT, exist_ok=True)
resf = f"{OUT}/results_{args.tag}.json"
R = json.load(open(resf)) if os.path.exists(resf) else {}


def save():
    json.dump(R, open(resf, "w"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


x, y, files = load_images(args.res)
N = len(y)
log("images", x.shape, "class counts", np.bincount(y).tolist())
xt = torch.from_numpy(x)
yt = torch.from_numpy(y)

# ---- global un-augmented caches (frozen lower part + pooled 1280-d features) ----
lower = lower_of(build_backbone())
ccache = f"{OUT}/lower_clean_{args.res}.pt"
if os.path.exists(ccache):
    lower_clean = torch.load(ccache)
else:
    outs = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N, 64):
            outs.append(lower(normalize(xt[i:i + 64])).half())
    lower_clean = torch.cat(outs)
    torch.save(lower_clean, ccache)
    log("lower_clean", tuple(lower_clean.shape), f"{time.time() - t0:.0f}s")
with torch.no_grad():
    t0 = Top(build_backbone()).eval()
    pooled_clean = torch.cat([t0.pooled(lower_clean[i:i + 128].float()) for i in range(0, N, 128)])
del t0


def split(seed):
    idx = np.arange(N)
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=seed)
    return np.sort(tr), np.sort(va), np.sort(te)


def probs_head(top, feats_pooled):
    top.head.eval()
    with torch.no_grad():
        return F.softmax(top.head(feats_pooled), 1).numpy()


def probs_top(top, z):
    top.set_mode(False)
    with torch.no_grad():
        return torch.cat([F.softmax(top(z[i:i + 128].float()), 1) for i in range(0, len(z), 128)]).numpy()


def train_p1(seed, weights):
    torch.manual_seed(seed)
    top = Top(build_backbone())
    opt = torch.optim.Adam(top.head.parameters(), lr=1e-3)
    tr = TR
    g = torch.Generator().manual_seed(seed)
    for ep in range(args.e1):
        top.head.train()
        perm = tr[torch.randperm(len(tr), generator=g).numpy()]
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            loss = F.cross_entropy(top.head(pooled_clean[b]), yt[b], weight=weights)
            opt.zero_grad(); loss.backward(); opt.step()
    return top


def run_p2(seed, variant, p1_head_state, aug_cache, cw):
    torch.manual_seed(seed + 17)
    top = Top(build_backbone())
    top.head.load_state_dict(p1_head_state)
    params = [p for n, p in top.named_parameters()]
    opt = torch.optim.Adam(params, lr=args.lr2)
    g = torch.Generator().manual_seed(seed * 7 + 1)
    ntr = len(TR)
    ytr = yt[TR]
    samp_p = (1.0 / torch.bincount(ytr, minlength=6).float())[ytr]
    curve = []
    t0 = time.time()
    for ep in range(args.e2):
        top.set_mode(True)
        if variant == "os":
            order = torch.multinomial(samp_p, ntr, replacement=True, generator=g)
        else:
            order = torch.randperm(ntr, generator=g)
        tot = 0.0
        for i in range(0, ntr, 32):
            b = order[i:i + 32]
            z = aug_cache[ep][b].float()
            logits = top(z)
            yb = ytr[b]
            if variant == "cw":
                loss = F.cross_entropy(logits, yb, weight=cw)
            elif variant == "focal":
                ce = F.cross_entropy(logits, yb, reduction="none")
                loss = (((1 - torch.exp(-ce)) ** 2) * ce).mean()
            else:
                loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        pv = probs_top(top, lower_clean[VA])
        mv = metrics(y[VA], pv)
        curve.append(dict(epoch=ep + 1, train_loss=tot / ntr, val_acc=mv["acc"], val_macro_f1=mv["macro_f1"]))
        log(f"  seed{seed} {variant} ep{ep + 1}/{args.e2} loss {tot / ntr:.3f} val_acc {mv['acc']:.3f} "
            f"val_mF1 {mv['macro_f1']:.3f} ({time.time() - t0:.0f}s)")
    return top, curve


for seed in [int(s) for s in args.seeds.split(",")]:
    TR, VA, TE = split(seed)
    json.dump(dict(train=[files[i] for i in TR], val=[files[i] for i in VA], test=[files[i] for i in TE]),
              open(f"{OUT}/split_seed{seed}.json", "w"))
    cnt = {k: np.bincount(y[v], minlength=6).tolist() for k, v in dict(train=TR, val=VA, test=TE).items()}
    R.setdefault(str(seed), {})["split_counts"] = cnt
    log(f"seed {seed}: train/val/test = {len(TR)}/{len(VA)}/{len(TE)}", cnt)
    TR_t = torch.from_numpy(TR)
    ntr_c = np.bincount(y[TR], minlength=6)
    cw = torch.tensor(len(TR) / (6 * ntr_c), dtype=torch.float32)  # inverse-frequency ("balanced") weights
    R[str(seed)]["class_weights"] = cw.tolist()

    # ---- Phase 1 (head only) : baseline (CE) and head-only + class weights ----
    done = R[str(seed)]
    if "p1_ce" not in done:
        top_ce = train_p1(seed, None)
        top_cw = train_p1(seed, cw)
        for name, t in (("p1_ce", top_ce), ("p1_cw", top_cw)):
            R[str(seed)][name] = dict(val=metrics(y[VA], probs_head(t, pooled_clean[VA])),
                                      test=metrics(y[TE], probs_head(t, pooled_clean[TE])))
            log(f"seed{seed} {name}: val acc {R[str(seed)][name]['val']['acc']:.3f}  "
                f"test acc {R[str(seed)][name]['test']['acc']:.3f}  test mF1 {R[str(seed)][name]['test']['macro_f1']:.3f}")
        torch.save(top_ce.head.state_dict(), f"{OUT}/p1head_seed{seed}.pt")
        save()
    p1_state = torch.load(f"{OUT}/p1head_seed{seed}.pt")

    todo = [v for v in args.variants.split(",") if f"p2_{v}" not in R[str(seed)]]
    if not todo:
        continue
    # ---- augmented lower-feature caches (shared by all Phase-2 variants of this seed) ----
    t0 = time.time()
    aug_cache = []
    with torch.no_grad():
        for ep in range(args.e2):
            g = torch.Generator().manual_seed(seed * 1000 + ep)
            outs = []
            for i in range(0, len(TR), 64):
                xb = xt[TR[i:i + 64]].float() / 255.0
                outs.append(lower(normalize(augment(xb, g))).half())
            aug_cache.append(torch.cat(outs))
    log(f"seed{seed}: augmented caches built in {time.time() - t0:.0f}s")
    for v in todo:
        top, curve = run_p2(seed, v, p1_state, aug_cache, cw)
        R[str(seed)][f"p2_{v}"] = dict(
            val=metrics(y[VA], probs_top(top, lower_clean[VA])),
            test=metrics(y[TE], probs_top(top, lower_clean[TE])), curve=curve)
        pt = probs_top(top, lower_clean[TE])
        np.save(f"{OUT}/testprobs_seed{seed}_{v}.npy", pt)
        if str(seed) in args.save_models.split(","):
            torch.save(top.state_dict(), f"{OUT}/top_seed{seed}_{v}.pt")
        m = R[str(seed)][f"p2_{v}"]["test"]
        log(f"seed{seed} p2_{v}: TEST acc {m['acc']:.3f} mF1 {m['macro_f1']:.3f} trashF1 {m['trash_f1']:.3f} auc {m['auc_macro']:.3f}")
        save()
    np.save(f"{OUT}/testlabels_seed{seed}.npy", y[TE])
    np.save(f"{OUT}/testidx_seed{seed}.npy", TE)
log("DONE")
