"""Near-duplicate analysis of TrashNet: perceptual hash (dHash) + ImageNet-embedding cosine similarity."""
import json, sys, numpy as np, torch
sys.path.insert(0, "/home/claude/exp/code")
from common import *
torch.set_num_threads(1)
x, y, files = load_images(224)
N = len(y)
# --- dHash 64 bit
import PIL.Image as I
def dhash(a):
    im = I.fromarray(a.transpose(1, 2, 0)).convert("L").resize((9, 8), I.LANCZOS)
    v = np.asarray(im, dtype=np.int16)
    return (v[:, 1:] > v[:, :-1]).flatten()
H = np.stack([dhash(a) for a in x])                       # N x 64 bool
Hd = (H[:, None, :] != H[None, :, :]).sum(-1)            # N x N Hamming
np.fill_diagonal(Hd, 99)
# --- embeddings (frozen ImageNet backbone, pooled 1280-d)
lower_clean = torch.load(f"{OUT}/lower_clean_224.pt")
top = Top(build_backbone()).eval()
with torch.no_grad():
    Z = torch.cat([top.pooled(lower_clean[i:i + 128].float()) for i in range(0, N, 128)]).numpy()
Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
Cs = Zn @ Zn.T
np.fill_diagonal(Cs, -1)
np.savez(f"{OUT}/dup_matrices.npz", Hd=Hd.astype(np.int8), Cs=Cs.astype(np.float16))
print("dHash NN distance quantiles", np.percentile(Hd.min(1), [5, 25, 50, 75, 95]))
print("cosine NN quantiles", np.percentile(Cs.max(1), [5, 25, 50, 75, 95]).round(3))
for t in (0, 2, 4, 6, 8, 10): print("hash<=", t, "images with >=1 neighbour:", int((Hd.min(1) <= t).sum()), "same-class share of pairs:",
      round(float((y[:, None] == y[None, :])[Hd <= t].mean()), 3) if (Hd <= t).any() else None)
for t in (0.80, 0.85, 0.90, 0.93, 0.95): print("cos>=", t, "images with >=1 neighbour:", int((Cs.max(1) >= t).sum()),
      "same-class share of pairs:", round(float((y[:, None] == y[None, :])[Cs >= t].mean()), 3))
