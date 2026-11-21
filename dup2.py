import json, sys, numpy as np
sys.path.insert(0, "/home/claude/exp/code")
from common import *
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
z = np.load(f"{OUT}/dup_matrices.npz"); Cs = z["Cs"].astype(np.float32)
y = np.load(f"{DATA}/cache_224.npz")["y"]; N = len(y)
same = y[:, None] == y[None, :]
res = {}
for thr in (0.80, 0.85, 0.90):
    A = (Cs >= thr) & same
    nc, lab = connected_components(coo_matrix(A), directed=False)
    sizes = np.bincount(lab)
    res[thr] = dict(groups=int(nc), in_multi=int((sizes[lab] > 1).sum()), largest=int(sizes.max()))
    print(thr, res[thr])
    if thr == 0.80: json.dump(lab.tolist(), open(f"{OUT}/groups_cos080.json", "w"))
# contamination of random splits
R = json.load(open(f"{OUT}/results_main.json"))
out = {}
for thr in (0.80, 0.85, 0.90):
    rows = []
    for s in range(5):
        sp = json.load(open(f"{OUT}/split_seed{s}.json")); files = list(np.load(f"{DATA}/cache_224.npz")["f"])
        idx = {f: i for i, f in enumerate(files)}
        tr = np.array([idx[f] for f in sp["train"]]); te = np.array([idx[f] for f in sp["test"]])
        nn = Cs[np.ix_(te, tr)].max(1)
        contam = nn >= thr
        pr = np.load(f"{OUT}/testprobs_main_seed{s}_ftc.npy"); ok = pr.argmax(1) == y[te]
        rows.append((contam.sum(), len(te), ok[contam].mean() if contam.any() else np.nan, ok[~contam].mean()))
    a = np.array(rows, float)
    print(f"thr {thr}: contaminated test imgs/seed {a[:,0].mean():.1f} of {a[:,1].mean():.0f} ({100*a[:,0].sum()/a[:,1].sum():.1f}%), acc contaminated {np.nanmean(a[:,2])*100:.1f}%, acc clean {a[:,3].mean()*100:.1f}%")
    out[str(thr)] = dict(per_seed=a.tolist())
json.dump(out, open(f"{OUT}/contamination.json", "w"))
