"""Final statistics: mean/SD/bootstrap CI, Nadeau-Bengio corrected paired t, sign-flip permutation, Holm, 2x2 factorial, contamination."""
import json, itertools, numpy as np, sys
from scipy import stats
O = "/home/claude/exp/out"
def load(tag): return json.load(open(f"{O}/results_{tag}.json"))
M = load("main"); G = json.load(open(f"{O}/results_grp.json")) if len(sys.argv) > 1 else None
def seeds(R, arm): return sorted(s for s in R if s.isdigit() and arm in R[s])
def vec(R, arm, k, S): return np.array([R[s][arm]["test"][k] for s in S]) * 100
def boot(a, B=20000, rng=np.random.default_rng(0)):
    m = rng.choice(a, (B, len(a)), replace=True).mean(1); return np.percentile(m, [2.5, 97.5])
def nb(d, ratio):
    k = len(d); v = d.var(ddof=1)
    if v == 0: return 0.0, 1.0, (0, 0)
    se = np.sqrt((1 / k + ratio) * v); t = d.mean() / se; p = 2 * stats.t.sf(abs(t), k - 1)
    h = stats.t.ppf(0.975, k - 1) * se; return t, p, (d.mean() - h, d.mean() + h)
def flip(d):  # exact sign-flip permutation test on the mean difference
    k = len(d); signs = np.array(list(itertools.product([1, -1], repeat=k)))
    return float((np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, i in enumerate(o): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1, run)
    return adj
def table(R, arms, comps, ratio, metrics=("acc", "macro_f1", "trash_f1")):
    S = sorted(set.intersection(*[set(seeds(R, a)) for a in arms]))
    out = dict(seeds=S, arms={}, comps={})
    for a in arms:
        out["arms"][a] = {k: dict(mean=float(vec(R, a, k, S).mean()), sd=float(vec(R, a, k, S).std(ddof=1)), ci=boot(vec(R, a, k, S)).tolist()) for k in ("acc", "macro_f1", "weighted_f1", "trash_f1", "auc_macro")}
    for k in metrics:
        rows = []
        for a, b in comps:
            d = vec(R, a, k, S) - vec(R, b, k, S); t, p, ci = nb(d, ratio)
            rows.append(dict(a=a, b=b, diff=float(d.mean()), wins=int((d > 0).sum()), n=len(d), p_nb=float(p), p_flip=flip(d), ci=list(map(float, ci)),
                             p_naive=float(stats.ttest_rel(vec(R, a, k, S), vec(R, b, k, S)).pvalue), dz=float(d.mean() / d.std(ddof=1)) if d.std() > 0 else 0))
        adj = holm([r["p_nb"] for r in rows])
        for r, h in zip(rows, adj): r["p_holm"] = float(h)
        out["comps"][k] = rows
    return out
res = {}
ARMS = ["p1_ce", "p1_cw", "p2_ce", "p2_cw", "p2_focal", "p2_os", "p2_hoc", "p2_hoa", "p2_ftc"]
arms = [a for a in ARMS if len(seeds(M, a)) >= 5]
comps = [("p2_ce", "p1_ce"), ("p2_cw", "p2_ce"), ("p2_focal", "p2_ce"), ("p2_os", "p2_ce"), ("p2_os", "p2_cw"), ("p2_cw", "p1_cw"),
         ("p2_ce", "p2_hoa"), ("p2_ce", "p2_ftc"), ("p2_hoa", "p2_hoc"), ("p2_ftc", "p2_hoc"), ("p2_hoa", "p1_ce"), ("p2_hoc", "p1_ce")]
comps = [c for c in comps if c[0] in arms and c[1] in arms]
res["main"] = table(M, arms, comps, 380 / 1768)
# 2x2 factorial (frozen/FT x noaug/aug), phase-2 arms
if all(a in arms for a in ("p2_hoc", "p2_hoa", "p2_ftc", "p2_ce")):
    S = res["main"]["seeds"]; fac = {}
    for k in ("acc", "macro_f1", "trash_f1"):
        hoc, hoa, ftc, fta = [vec(M, a, k, S) for a in ("p2_hoc", "p2_hoa", "p2_ftc", "p2_ce")]
        eff = {"fine-tuning (main effect)": (ftc + fta) / 2 - (hoc + hoa) / 2, "augmentation (main effect)": (hoa + fta) / 2 - (hoc + ftc) / 2,
               "interaction": (fta - ftc) - (hoa - hoc)}
        rows = []
        for n, d in eff.items():
            t, p, ci = nb(d, 380 / 1768); rows.append(dict(effect=n, diff=float(d.mean()), ci=list(map(float, ci)), p_nb=float(p), p_flip=flip(d), wins=int((d > 0).sum())))
        for r, h in zip(rows, holm([r["p_nb"] for r in rows])): r["p_holm"] = float(h)
        fac[k] = rows
    res["factorial"] = fac
# grouped
if G is not None:
    ga = [a for a in ARMS if len(seeds(G, a)) >= 5]
    gc = [c for c in comps if c[0] in ga and c[1] in ga]
    res["grouped"] = table(G, ga, gc, 420 / 1685)
# per-class for all arms (mean over seeds)
S = res["main"]["seeds"]
res["perclass"] = {a: {k: (np.array([M[s][a]["test"]["per_class"][k] for s in S]).mean(0) * 100).tolist() for k in ("precision", "recall", "f1")} for a in arms}
res["pooled_cm"] = {a: sum(np.array(M[s][a]["test"]["confusion"]) for s in S).tolist() for a in arms}
# architectures
import os
for arch in ("mobilenetv3_large_100", "efficientnet_b0"):
    f = f"{O}/results_arch_{arch}.json"
    if os.path.exists(f):
        A = json.load(open(f)); SA = seeds(A, "p2_ce"); res.setdefault("arch", {})[arch] = dict(seeds=SA, params=A.get("_params"),
            **{a: {k: dict(mean=float(vec(A, a, k, SA).mean()), sd=float(vec(A, a, k, SA).std(ddof=1))) for k in ("acc", "macro_f1", "trash_f1")} for a in ("p1_ce", "p2_ce")},
            diff_acc=float((vec(A, "p2_ce", "acc", SA) - vec(A, "p1_ce", "acc", SA)).mean()), wins=int(((vec(A, "p2_ce", "acc", SA) - vec(A, "p1_ce", "acc", SA)) > 0).sum()))
json.dump(res, open(f"{O}/final_stats.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in res if k in ("factorial",)}, indent=1)[:3000])
for k, v in res["main"]["arms"].items(): print(k, {m: round(v[m]["mean"], 1) for m in ("acc", "macro_f1", "trash_f1")}, "n=", len(S))
for r in res["main"]["comps"]["acc"]: print(r["a"], "-", r["b"], round(r["diff"], 2), r["wins"], "/", r["n"], "pNB", round(r["p_nb"], 3), "holm", round(r["p_holm"], 3), "flip", round(r["p_flip"], 3), "naive", round(r["p_naive"], 3))
