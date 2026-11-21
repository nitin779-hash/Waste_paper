import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],"font.size":9})
O="/home/claude/exp/out"; R=json.load(open(f"{O}/results_main.json")); S=sorted(R); G=json.load(open(f"{O}/results_grp.json")); SG=sorted(G)
C=["cardboard","glass","metal","paper","plastic","trash"]
# ---- Fig 1
fig,ax=plt.subplots(figsize=(6.6,3.3),dpi=300); ax.axis("off"); ax.set_xlim(0,100); ax.set_ylim(0,50)
def box(x,y,w,h,t,fc,fs=6.9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=1.2",fc=fc,ec="#333",lw=0.9)); ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs)
def arr(x1,x2,y): ax.annotate("",xy=(x2,y),xytext=(x1,y),arrowprops=dict(arrowstyle="->",lw=1.1,color="#333"))
box(0.5,36,21,12,"TrashNet\n2,527 images\n6 classes","#eef2f7"); arr(22,25,42)
box(25.5,36,24,12,"Split per seed\n(a) random 70/15/15\n(b) duplicate-grouped","#eef2f7"); arr(50,53,42)
box(53.5,36,21,12,"Phase 1 (10 ep.)\nhead on frozen\nbackbone","#dbe7d3"); arr(75,78,42)
box(78.5,36,21,12,"Phase 2 (15 ep.)\n2×2 control:\nfreeze/tune × aug","#f3e3c3")
box(0.5,19,32,12,"Imbalance remedy\nnone | weights\nfocal | oversampling","#f4d9d9")
box(35,19,30,12,"Extra backbones (3 seeds)\nMobileNetV3-L\nEfficientNet-B0","#e3dcef")
box(67.5,19,32,12,"Statistics, 10 seeds\nNadeau–Bengio t, sign-flip\nHolm, bootstrap CI","#e3dcef")
box(0.5,2,99,12,"Protocol: no early stopping or checkpoint selection; learning rate fixed on\nseed-0 validation data only; final epoch evaluated once on the held-out test split.","#fff6d6",6.9)
plt.savefig("fig1_pipeline_300dpi.png",dpi=300,bbox_inches="tight",facecolor="white"); plt.close()
# ---- Fig 2
V=[("p1_ce","Head-only\nbaseline"),("p2_hoc","Frozen\nclean"),("p2_hoa","Frozen\n+ aug"),("p2_ftc","Tuned\nclean"),("p2_ce","Tuned\n+ aug")]
fig,(a,b)=plt.subplots(1,2,figsize=(6.8,3.5),dpi=300,gridspec_kw={"width_ratios":[1.15,1]})
w=0.38
rm=[np.mean([R[s][v]["test"]["acc"] for s in S])*100 for v,_ in V]; rs=[np.std([R[s][v]["test"]["acc"] for s in S],ddof=1)*100 for v,_ in V]
a.bar(np.arange(5)-w/2,rm,w,yerr=rs,capsize=2,color="#3b6ea5",label="Random split (10 seeds)",error_kw=dict(lw=0.8))
gv=[v for v,_ in V if v!="p2_hoc"]; gi=[i for i,(v,_) in enumerate(V) if v!="p2_hoc"]
gm=[np.mean([G[s][v]["test"]["acc"] for s in SG])*100 for v in gv]; gs=[np.std([G[s][v]["test"]["acc"] for s in SG],ddof=1)*100 for v in gv]
a.bar(np.array(gi)+w/2,gm,w,yerr=gs,capsize=2,color="#c9833a",label="Grouped split (5 seeds)",error_kw=dict(lw=0.8))
a.set_xticks(range(5)); a.set_xticklabels([n for _,n in V],fontsize=7.5); a.set_ylim(78,95); a.set_ylabel("Test accuracy (%), mean ± SD",fontsize=8.5)
a.tick_params(axis="y",labelsize=8); a.legend(fontsize=7.5,loc="upper left",frameon=False); a.set_title("(a) Fine-tuning × augmentation control",fontsize=9); a.spines[["top","right"]].set_visible(False)
cm=sum(np.array(R[s]["p2_ce"]["test"]["confusion"]) for s in S); cmn=cm/cm.sum(1,keepdims=True)
b.imshow(cmn,cmap="Blues",vmin=0,vmax=1); b.set_xticks(range(6)); b.set_yticks(range(6)); b.set_xticklabels(C,rotation=45,ha="right",fontsize=8); b.set_yticklabels(C,fontsize=8)
for i in range(6):
    for j in range(6): b.text(j,i,f"{cm[i,j]}",ha="center",va="center",fontsize=8,color="white" if cmn[i,j]>0.5 else "black")
b.set_xlabel("Predicted",fontsize=8.5); b.set_ylabel("True",fontsize=8.5); b.set_title("(b) Tuned + aug, pooled counts",fontsize=9)
plt.tight_layout(); plt.savefig("fig2_results_300dpi.png",dpi=300,bbox_inches="tight",facecolor="white"); plt.close()
print(cm.sum(), [round(x,1) for x in rm], [round(x,1) for x in gm])
