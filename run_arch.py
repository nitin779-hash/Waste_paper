"""Head-only vs two-phase fine-tuning for other lightweight backbones (timm), same protocol as run_exp.py (CE loss only)."""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, DATA, MEAN, STD, augment, load_images, metrics
ap = argparse.ArgumentParser()
ap.add_argument("--arch", required=True)  # mobilenetv3_large_100 | efficientnet_b0
ap.add_argument("--seeds", default="0,1,2,3,4"); ap.add_argument("--e1", type=int, default=10); ap.add_argument("--e2", type=int, default=15)
ap.add_argument("--lr2", type=float, default=1e-4); ap.add_argument("--split_at", type=int, default=5)
args = ap.parse_args(); torch.set_num_threads(2)
W = {"mobilenetv3_large_100": f"{DATA}/w2/mobilenetv3_large_100_ra-f55367f5.pth", "efficientnet_b0": f"{DATA}/w2/efficientnet_b0_ra-3dd342df.pth"}
resf = f"{OUT}/results_arch_{args.arch}.json"; R = json.load(open(resf)) if os.path.exists(resf) else {}
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

def build():
    m = timm.create_model(args.arch, pretrained=True, pretrained_cfg_overlay=dict(file=W[args.arch]), num_classes=1000)
    return m
class Lower(nn.Module):
    def __init__(s, m):
        super().__init__(); s.stem = nn.Sequential(m.conv_stem, m.bn1); s.blocks = nn.Sequential(*list(m.blocks)[:args.split_at])
    def forward(s, x): return s.blocks(s.stem(x))
class Top(nn.Module):
    def __init__(s, m):
        super().__init__(); s.blocks = nn.Sequential(*list(m.blocks)[args.split_at:])
        s.neck = nn.Sequential(m.conv_head, getattr(m, "bn2", nn.Identity()), getattr(m, "act2", nn.Identity()))
        s.pool = nn.AdaptiveAvgPool2d(1); s.head = nn.Sequential(nn.Linear(1280, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 6))
    def pooled(s, z): return s.pool(s.neck(s.blocks(z))).flatten(1)
    def forward(s, z): return s.head(s.pooled(z))
    def set_mode(s, th=True): s.eval(); s.head.train(th)
norm = lambda x: (x.float() / 255.0 - MEAN) / STD if x.dtype == torch.uint8 else (x - MEAN) / STD
x, y, files = load_images(224); N = len(y); xt = torch.from_numpy(x); yt = torch.from_numpy(y)
m0 = build(); lower = Lower(m0).eval()
with torch.no_grad(): lc = torch.cat([lower(norm(xt[i:i + 64])).half() for i in range(0, N, 64)])
t0 = Top(build()).eval()
with torch.no_grad(): pc = torch.cat([t0.pooled(lc[i:i + 128].float()) for i in range(0, N, 128)])
nparams = sum(p.numel() for p in m0.parameters()); log(args.arch, "params", nparams, "lower feats", tuple(lc.shape))
def probs_top(top, z):
    top.set_mode(False)
    with torch.no_grad(): return torch.cat([F.softmax(top(z[i:i + 128].float()), 1) for i in range(0, len(z), 128)]).numpy()
for seed in [int(s) for s in args.seeds.split(",")]:
    if str(seed) in R and "p2_ce" in R[str(seed)]: continue
    idx = np.arange(N); tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=seed); TR, VA, TE = np.sort(tr), np.sort(va), np.sort(te)
    R.setdefault(str(seed), {}); torch.manual_seed(seed)
    top = Top(build()); opt = torch.optim.Adam(top.head.parameters(), lr=1e-3); g = torch.Generator().manual_seed(seed)
    for ep in range(args.e1):
        top.head.train(); perm = TR[torch.randperm(len(TR), generator=g).numpy()]
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]; loss = F.cross_entropy(top.head(pc[b]), yt[b]); opt.zero_grad(); loss.backward(); opt.step()
    top.head.eval()
    with torch.no_grad(): ph = lambda ix: F.softmax(top.head(pc[ix]), 1).detach().numpy()
    R[str(seed)]["p1_ce"] = dict(val=metrics(y[VA], ph(VA)), test=metrics(y[TE], ph(TE)))
    log(f"seed{seed} p1_ce test acc {R[str(seed)]['p1_ce']['test']['acc']:.3f}")
    hs = top.head.state_dict()
    t0_ = time.time(); caches = []
    with torch.no_grad():
        for ep in range(args.e2):
            gg = torch.Generator().manual_seed(seed * 1000 + ep)
            caches.append(torch.cat([lower(norm(augment(xt[TR[i:i + 64]].float() / 255.0, gg))).half() for i in range(0, len(TR), 64)]))
    log(f"seed{seed} aug caches {time.time() - t0_:.0f}s")
    torch.manual_seed(seed + 17); top = Top(build()); top.head.load_state_dict(hs)
    opt = torch.optim.Adam(top.parameters(), lr=args.lr2); g = torch.Generator().manual_seed(seed * 7 + 1); ytr = yt[TR]; curve = []
    for ep in range(args.e2):
        top.set_mode(True); order = torch.randperm(len(TR), generator=g); tot = 0.0
        for i in range(0, len(TR), 32):
            b = order[i:i + 32]; loss = F.cross_entropy(top(caches[ep][b].float()), ytr[b]); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        mv = metrics(y[VA], probs_top(top, lc[VA])); curve.append(dict(epoch=ep + 1, train_loss=tot / len(TR), val_acc=mv["acc"]))
    R[str(seed)]["p2_ce"] = dict(val=mv, test=metrics(y[TE], probs_top(top, lc[TE])), curve=curve)
    np.save(f"{OUT}/testprobs_arch_{args.arch}_seed{seed}.npy", probs_top(top, lc[TE]))
    log(f"seed{seed} p2_ce TEST acc {R[str(seed)]['p2_ce']['test']['acc']:.3f} mF1 {R[str(seed)]['p2_ce']['test']['macro_f1']:.3f}")
    R["_params"] = nparams; json.dump(R, open(resf, "w"))
log("DONE")
