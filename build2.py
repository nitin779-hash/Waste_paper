import json, numpy as np
from scipy import stats
from docx import Document
from docx.shared import Pt, Mm, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

O = "/home/claude/exp/out"
R = json.load(open(f"{O}/results_main.json")); S = sorted(R); NS = len(S)
D = json.load(open(f"{O}/final_stats.json")); GR = json.load(open(f"{O}/results_grp.json")); SG = sorted(GR)
E = json.load(open(f"{O}/edge_bench.json"))
C = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
def g(v, k, split="test"): return np.array([R[s][v][split][k] for s in S])
def ms(v, k, d=1):
    a = g(v, k) * 100; return f"{a.mean():.{d}f} ± {a.std(ddof=1):.{d}f}"
def mean(v, k): return g(v, k).mean() * 100
def paired(a, b, k):
    x, y = g(a, k), g(b, k); d = x - y
    return d.mean() * 100, int((d > 0).sum()), stats.ttest_rel(x, y).pvalue
NAMES = {"p1_ce": "Head-only, cross-entropy (baseline)", "p1_cw": "Head-only + class weights",
         "p2_ce": "Two-phase, cross-entropy", "p2_cw": "Two-phase + class weights",
         "p2_focal": "Two-phase + focal loss", "p2_os": "Two-phase + oversampling"}
VARS = list(NAMES)

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Mm(210), Mm(297)
sec.left_margin = sec.right_margin = Mm(32); sec.top_margin = Mm(28); sec.bottom_margin = Mm(28)
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(10)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.space_after = Pt(0); st.paragraph_format.line_spacing = 1.08

def P(text="", size=10, bold=False, italic=False, align="j", before=0, after=4, indent=True, keep=False):
    p = doc.add_paragraph()
    p.alignment = {"j": WD_ALIGN_PARAGRAPH.JUSTIFY, "c": WD_ALIGN_PARAGRAPH.CENTER, "l": WD_ALIGN_PARAGRAPH.LEFT}[align]
    p.paragraph_format.space_before = Pt(before); p.paragraph_format.space_after = Pt(after)
    if indent and align == "j": p.paragraph_format.first_line_indent = Mm(4.5)
    if keep: p.paragraph_format.keep_with_next = True
    add_runs(p, text, size, bold, italic); return p

def add_runs(p, text, size=10, bold=False, italic=False):
    # inline markup: *italic*  and  ^sup^
    import re
    for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^|~[^~]+~|`[^`]+`)", text):
        if not tok: continue
        r = None
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4: r = p.add_run(tok[2:-2]); r.bold = True
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2: r = p.add_run(tok[1:-1]); r.italic = True
        elif tok.startswith("^") and tok.endswith("^"): r = p.add_run(tok[1:-1]); r.font.superscript = True
        elif tok.startswith("~") and tok.endswith("~") and len(tok) > 2: r = p.add_run(tok[1:-1]); r.font.subscript = True
        elif tok.startswith("`") and tok.endswith("`") and len(tok) > 2: r = p.add_run(tok[1:-1]); r.font.name = "Courier New"; r.font.size = Pt(size-1); continue
        else: r = p.add_run(tok); r.italic = italic
        r.font.size = Pt(size); r.bold = (bold or r.bold) or None
        r.font.name = "Times New Roman"

def H1(t): return P(t, 12, bold=True, align="l", before=10, after=5, indent=False, keep=True)
def H2(t): return P(t, 10, bold=True, align="l", before=6, after=3, indent=False, keep=True)
def EQ(t, n):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(2); p.paragraph_format.space_after = Pt(5)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_runs(p, t + "      (" + str(n) + ")", 10)
def BUL(items):
    for t in items:
        p = doc.add_paragraph(); p.paragraph_format.left_indent = Mm(8); p.paragraph_format.first_line_indent = Mm(-4)
        p.paragraph_format.space_after = Pt(2); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        add_runs(p, "•  " + t, 10)
    doc.paragraphs[-1].paragraph_format.space_after = Pt(5)

def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), color); tcPr.append(sh)
def borders(tbl):
    tblPr = tbl._tbl.tblPr; b = OxmlElement("w:tblBorders")
    for e in ("top", "bottom", "insideH"):
        x = OxmlElement(f"w:{e}"); x.set(qn("w:val"), "single"); x.set(qn("w:sz"), "4"); x.set(qn("w:color"), "808080"); b.append(x)
    lay = tblPr.find(qn('w:tblLayout'))
    (lay.addprevious(b) if lay is not None else tblPr.append(b))
def TABLE(caption, header, rows, widths, align=None, bold_rows=(), fs=8.5):
    P(caption, 9, align="l", before=6, after=3, indent=False, keep=True)
    t = doc.add_table(rows=1 + len(rows), cols=len(header)); t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
    borders(t)
    for i, row in enumerate([header] + rows):
        for j, val in enumerate(row):
            c = t.cell(i, j); c.width = Mm(widths[j]); c.text = ""
            p = c.paragraphs[0]; p.paragraph_format.space_after = Pt(1); p.paragraph_format.space_before = Pt(1)
            a = (align[j] if align else ("l" if j == 0 else "c"))
            p.alignment = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": WD_ALIGN_PARAGRAPH.CENTER}[a]
            add_runs(p, str(val), fs, bold=(i == 0 or (i - 1) in bold_rows))
            if i == 0: shade(c, "E8E8E8")
    # repeat header
    trPr = t.rows[0]._tr.get_or_add_trPr(); h = OxmlElement("w:tblHeader"); h.set(qn("w:val"), "true"); trPr.append(h)
    for r in t.rows:
        trPr = r._tr.get_or_add_trPr(); cs = OxmlElement("w:cantSplit"); cs.set(qn("w:val"), "true"); trPr.append(cs)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
def FIG(path, caption, width_mm):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(6); p.add_run().add_picture(path, width=Mm(width_mm))
    P(caption, 9, align="j", before=2, after=8, indent=False)

# ---------------------------------------------------------------- helpers for numbers
def g(v, k, split="test"): return np.array([R[s][v][split][k] for s in S])
def gg(v, k): return np.array([GR[s][v]["test"][k] for s in SG])
def ms(v, k, d=1): a = g(v, k) * 100; return f"{a.mean():.{d}f} ± {a.std(ddof=1):.{d}f}"
def msg(v, k, d=1): a = gg(v, k) * 100; return f"{a.mean():.{d}f} ± {a.std(ddof=1):.{d}f}"
def mean(v, k): return g(v, k).mean() * 100
def gmean(v, k): return gg(v, k).mean() * 100
def comp(a, b, met="acc", grp=False):
    src = D["grouped"]["comps"] if grp else D["main"]["comps"]
    for c in src[met]:
        if c["a"] == a and c["b"] == b: return c
    raise KeyError((a, b, met))
def fp(p): return "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"
def stat(c, holm=True):
    s = f"{c['diff']:+.1f} pts, 95% CI [{c['ci'][0]:+.1f}, {c['ci'][1]:+.1f}], better in {c['wins']}/{c['n']} seeds, *p*~NB~ = {fp(c['p_nb'])}"
    if holm: s += f", Holm-adjusted *p* = {fp(c['p_holm'])}"
    return s
fac = {e["effect"]: e for e in D["factorial"]["acc"]}
FT, AUG, INT = fac["fine-tuning (main effect)"], fac["augmentation (main effect)"], fac["interaction"]
bench1, bench2 = E["latency"]["1_thread"], E["latency"]["2_thread"]
n_img = 2527
A = {k: mean(k, "acc") for k in ["p1_ce", "p1_cw", "p2_ce", "p2_cw", "p2_focal", "p2_os", "p2_hoc", "p2_hoa", "p2_ftc"]}
GA = {k: gmean(k, "acc") for k in ["p1_ce", "p1_cw", "p2_ce", "p2_cw", "p2_os", "p2_hoa", "p2_ftc"]}
PC = D["perclass"]
CON = json.load(open(f"{O}/contamination.json"))
cr = np.array(CON["0.8"]["per_seed"]); cont_n = cr[:, 0].mean(); cont_frac = (cr[:, 0] / cr[:, 1]).mean() * 100
cont_acc_with = cr[:, 2].mean() * 100; cont_acc_wo = cr[:, 3].mean() * 100
cr9 = np.array(CON["0.9"]["per_seed"]); cr85 = np.array(CON["0.85"]["per_seed"])
GRP = json.load(open(f"{O}/groups_cos080.json")); from collections import Counter
gc = Counter(GRP); n_groups = len(gc); multi = [v for v in gc.values() if v > 1]
n_multi = len(multi); imgs_multi = sum(multi); max_group = max(multi)
gsplit = [GR[s]["split_counts"] for s in SG]; gte = np.mean([sum(x["test"]) for x in gsplit]); gtr = np.mean([sum(x["train"]) for x in gsplit]); gva = np.mean([sum(x["val"]) for x in gsplit])
ARCH = D["arch"]
def arch_row(name, key):
    a = ARCH[key]; return [name, f"{a['params']/1e6:.2f}", f"{a['p1_ce']['acc']['mean']:.1f} ± {a['p1_ce']['acc']['sd']:.1f}", f"{a['p2_ce']['acc']['mean']:.1f} ± {a['p2_ce']['acc']['sd']:.1f}", f"{a['p2_ce']['macro_f1']['mean']:.1f}", f"{a['diff_acc']:+.1f}", f"{a['wins']}/3"]
s3 = ["0", "1", "2"]
m2_p1 = np.mean([R[s]["p1_ce"]["test"]["acc"] for s in s3]) * 100; m2_p2 = np.mean([R[s]["p2_ce"]["test"]["acc"] for s in s3]) * 100
m2_f1 = np.mean([R[s]["p2_ce"]["test"]["macro_f1"] for s in s3]) * 100
m2_w = sum(R[s]["p2_ce"]["test"]["acc"] > R[s]["p1_ce"]["test"]["acc"] for s in s3)
C = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NAMES = {"p1_ce": "Head-only, CE (baseline)", "p1_cw": "Head-only + class weights", "p2_hoc": "Frozen, clean, +15 ep. (HO-C)",
         "p2_hoa": "Frozen, augmented (HO-A)", "p2_ftc": "Tuned, clean (FT-C)", "p2_ce": "Tuned, augmented (FT-A)",
         "p2_cw": "FT-A + class weights", "p2_focal": "FT-A + focal loss", "p2_os": "FT-A + oversampling"}
ORDER = ["p1_ce", "p1_cw", "p2_hoc", "p2_hoa", "p2_ftc", "p2_ce", "p2_cw", "p2_focal", "p2_os"]
cmall = sum(np.array(R[s]["p2_ce"]["test"]["confusion"]) for s in S)
tr_p, tr_r = {v: PC[v]["precision"][5] for v in PC}, {v: PC[v]["recall"][5] for v in PC}
tot_gain = comp("p2_ce", "p1_ce"); hoc_gain = comp("p2_hoc", "p1_ce"); aug_hoc = comp("p2_hoa", "hoc" if False else "p2_hoc")
ft_hoc = comp("p2_ftc", "p2_hoc"); ce_hoa = comp("p2_ce", "p2_hoa"); ce_ftc = comp("p2_ce", "p2_ftc"); hoa_p1 = comp("p2_hoa", "p1_ce")
cw_ce, fo_ce, os_ce = comp("p2_cw", "p2_ce"), comp("p2_focal", "p2_ce"), comp("p2_os", "p2_ce")
gtot = comp("p2_ce", "p1_ce", grp=True)

# ================================================================ TITLE BLOCK
P("Disentangling Fine-Tuning, Augmentation and Training Length in Two-Phase MobileNetV2 Transfer Learning: A Multi-Seed, Duplicate-Aware Study on TrashNet",
  14, bold=True, align="c", before=0, after=10, indent=False)
P("Shalza Aggarwal", 11, align="c", after=2, indent=False)
P("Jagan Institute of Management Studies (JIMS), Rohini, New Delhi, India  ·  [department and e-mail address to be inserted by the author]", 9, italic=True, align="c", after=10, indent=False)
P("Abstract.", 9, bold=True, align="j", after=0, indent=False, keep=True)
abstract = (
 f"Reported results on the public TrashNet benchmark often come from a single random split and a training recipe whose parts are never separated, so it is unclear which "
 f"design choice produces the gain. We study a common recipe, an ImageNet-pretrained MobileNetV2 trained in two phases (a new head on a frozen backbone, then adaptation of the last blocks), "
 f"and ask what each ingredient contributes. In a 2×2 control (frozen or tuned backbone, with or without data augmentation) over {NS} seeds, test accuracy rises from {A['p1_ce']:.1f}% for the head-only baseline to {A['p2_ce']:.1f}% "
 f"for the full recipe ({tot_gain['diff']:+.1f} points, 95% CI [{tot_gain['ci'][0]:+.1f}, {tot_gain['ci'][1]:+.1f}]). The gain is not due to fine-tuning alone: extra training epochs, augmentation and backbone tuning each contribute, "
 f"and only the fine-tuning main effect ({FT['diff']:+.1f} points) is individually reliable after Holm correction. "
 f"We then quantify a second confound. About {cont_frac:.0f}% of the test images of a random split (seeds 0–4) have a near-duplicate of the same class in the training set (cosine similarity ≥ 0.80 of ImageNet embeddings), "
 f"and these images are classified far better ({cont_acc_with:.1f}% against {cont_acc_wo:.1f}%). A duplicate-grouped split lowers head-only accuracy to {GA['p1_ce']:.1f}% and the full recipe to {GA['p2_ce']:.1f}%, "
 f"so the benefit of tuning survives but absolute numbers are inflated on random splits. Class weighting, focal loss and oversampling give no reliable gain in overall accuracy, and a three-seed check with MobileNetV3-Large and EfficientNet-B0 "
 f"shows the same direction of effect. We report model size ({E['params_m']:.2f} M parameters, {E['macs_m']:.0f} M multiply-accumulates) and a server-CPU latency, and we do not claim edge-device performance. Code, splits and logs are released.")
P(abstract, 9, after=4, indent=False)
P("Keywords:", 9, bold=True, align="l", after=0, indent=False, keep=True)
P("waste classification; transfer learning; MobileNetV2; ablation; data augmentation; near-duplicate images; statistical testing; TrashNet", 9, after=8, indent=False)

# ================================================================ 1 INTRODUCTION
H1("1  Introduction")
P("Sorting municipal waste by material is labour-intensive, and mistakes at the source reduce the value of what is recovered. Camera-based classifiers are a promising way to automate "
  "the step, whether in a smart bin, on a conveyor or in a phone application. Because such devices are cost- and power-constrained, the networks used in them are usually compact models "
  "pretrained on ImageNet, with MobileNetV2 [1] a common choice.")
P("A recurring pattern in the literature on this task is that a pretrained lightweight network is adapted to a small public dataset, most often TrashNet [9], and a single headline accuracy is reported [2–4]. "
  "Three questions are hard to answer from such papers. First, how much of the final accuracy comes from fine-tuning the backbone, as opposed to training a new head longer, or to the data augmentation that "
  "usually accompanies fine-tuning? Second, do the standard measures against class imbalance actually help? Third, is the number an honest estimate for unseen objects, given that TrashNet contains several "
  "photographs of the same physical item, so that a random split can put near-identical pictures in both the training and the test set?")
P("We address these questions for one popular recipe. A new classification head is first learned on a frozen backbone, and the last inverted-residual blocks are then unfrozen and trained at a small learning rate. "
  "Two-phase schedules like this are standard and MobileNetV2 transfer learning for recyclable materials has been reported before [8], so we do not present the recipe as new. "
  "The contribution is a controlled measurement of what each part of it is worth, and how much of the measured value is an artefact of the evaluation split. Concretely:")
BUL([
 "We separate fine-tuning from augmentation and from training length with a 2×2 control (frozen or tuned backbone, with or without augmentation) plus a matched-length frozen baseline, run over ten seeds.",
 "We quantify near-duplicate contamination of random splits with an embedding-based similarity analysis and re-run the main arms on duplicate-grouped splits, which gives a more conservative estimate of accuracy on unseen objects.",
 "We replace uncorrected paired *t*-tests on five overlapping splits by the variance-corrected resampled *t*-test of Nadeau and Bengio [20], exact sign-flip tests, bootstrap confidence intervals and Holm correction [21].",
 "We compare three imbalance remedies for the rare *trash* class and report precision, recall and F1 for every class and every arm.",
 "We repeat the headline comparison on two other lightweight backbones (three seeds each) and release code, split files, results and logs.",
])
P("The paper is organised as follows. Section 2 reviews related work, Section 3 describes data, splits and the duplicate analysis, Section 4 the model and training variants, Section 5 the experimental setup and statistics, "
  "Section 6 the results, Section 7 the discussion, Section 8 the limitations and Section 9 the conclusions. Abbreviations are defined at first use and summarised in Table 2.")

# ================================================================ 2 RELATED WORK
H1("2  Related Work")
H2("2.1  Compact networks and transfer learning")
P("MobileNetV2 [1] builds on depthwise-separable convolutions [10] and adds inverted residual blocks with linear bottlenecks. MobileNetV3 [17] adds a searched architecture and a lighter classification neck, "
  "and EfficientNet [18] scales depth, width and resolution jointly; the smallest variants of both (MobileNetV3-Large and EfficientNet-B0) are used in Section 6.5. "
  "Pretraining on ImageNet [12] and re-using the features for a small target set is the dominant strategy when labelled images are scarce. Yosinski et al. [13] showed that transferability of features decreases "
  "towards the top of the network, which motivates tuning only the upper layers. This is the logic behind the two-phase schedule studied here: the head is trained alone first, so that random head weights do not send large gradients "
  "into the pretrained backbone, and the upper blocks are adapted afterwards.")
H2("2.2  Waste classification with deep networks")
P("TrashNet [9] is the most widely used public benchmark for household waste images and has six classes photographed on a plain background. Xu et al. [2] combine transfer learning with a lightweight network, "
  "Yong et al. [3] apply MobileNetV2 and Poudel and Poudyal [4] a transfer-learned CNN. Srivatsan et al. [5] study transfer learning with convolutional networks for the same task. Application-oriented work "
  "includes Kure et al. [6] on energy-efficient robotic waste sorting and Frost et al. [7] on classifying meal waste for composting. Andriani et al. [8] report MobileNetV2 transfer learning for plastic, paper and glass items "
  "on a three-class collection with a single 80:20 train/validation split and without an ablation of the training components. Recent work has moved towards larger collections and deployment. Verber et al. [14] combine transfer learning with edge deployment "
  "on 13,933 images in ten classes, and Theera-Ampornpunt et al. [15] propose a knowledge-distillation framework for real-time garbage classification and introduce a 15,681-image, 13-class dataset. "
  "These studies show where lightweight classifiers are heading, but they do not separate the effect of tuning from that of augmentation, and they do not report how near-duplicate images affect their splits.")
H2("2.3  Evaluation pitfalls: duplicates, variance and significance")
P("Barz and Denzler [19] showed that CIFAR contains near-duplicates across its training and test partitions and that removing them lowers reported accuracy; the same concern applies to a dataset such as TrashNet, in which each object is photographed several times. "
  "On the statistical side, Dietterich [22] and Nadeau and Bengio [20] pointed out that repeated random splits give overlapping training and test sets, so that the variance of the mean difference is underestimated by a naive paired *t*-test and a corrected variance is needed. "
  "Demšar [23] recommends non-parametric tests and care with multiple comparisons, and Bouthillier et al. [24] argue that benchmark conclusions should account for several sources of variation, including data splits and random initialisation. "
  "We adopt these recommendations; Section 5.2 gives the details.")
H2("2.4  Class imbalance")
P("TrashNet is moderately imbalanced: the *trash* class has 137 images against 594 for *paper*. Cost-sensitive learning scales each sample's loss by a class-dependent weight, usually the inverse class frequency, and "
  "class-balanced variants based on the effective number of samples were proposed by Cui et al. [25]. Loss re-shaping, of which the focal loss of Lin et al. [26] is the best known, down-weights examples that are already classified confidently. "
  "Data-level methods re-sample the training set. We place the three families side by side under identical conditions.")
H2("2.5  Positioning of this work")
P("Compared with the studies above, we do not propose an architecture and do not claim a record accuracy. The scientific question is narrower: for a standard recipe on a dataset with known near-duplicates, which part of the recipe "
  "produces the gain, and how much of the measured accuracy is attributable to the evaluation split rather than to the model? The answer, that the gain is shared between several ingredients and that random-split accuracy is inflated by "
  "duplicates, is useful to anyone who compares numbers across TrashNet papers.")

# ================================================================ 3 DATA
H1("3  Dataset, Splits and Duplicate Analysis")
H2("3.1  Dataset")
P(f"We use the “dataset-resized” release of TrashNet [9] with {n_img:,} photographs in six classes: cardboard (403 images), glass (501), metal (410), paper (594), plastic (482) and trash (137). "
  "Each object is photographed against a plain background, and the images are provided at reduced resolution. The ratio between the largest and smallest class is 4.3. The *trash* class is both the rarest and the most heterogeneous, "
  "so it is the natural stress test for imbalance handling. No image was removed and no external data was added.")
H2("3.2  Random stratified splits")
P(f"For every seed s ∈ {{0, …, {NS-1}}} we draw a stratified 70/15/15 split into training, validation and test images with `train_test_split` from scikit-learn (`random_state` = s, applied twice). "
  "The split differs for every seed, and the test images are never used for training or model selection. Class counts are identical for all seeds (1,768 / 379 / 380 images, Table 1) because only the assignment of individual images changes.")
c0 = R["0"]["split_counts"]
TABLE("**Table 1.** Number of images per class and partition for the random splits (seed 0; identical class counts for all ten seeds).",
      ["Partition"] + C + ["Total"],
      [[k.capitalize()] + [str(x) for x in c0[k]] + [str(sum(c0[k]))] for k in ("train", "val", "test")] +
      [["All"] + [str(c0["train"][i] + c0["val"][i] + c0["test"][i]) for i in range(6)] + [str(n_img)]],
      [22, 17, 14, 14, 14, 16, 14, 14], bold_rows=(3,))
H2("3.3  Test-set isolation")
P("Three rules apply to every experiment. (i) *Hyper-parameters are chosen on validation data only.* The only tuned value was the Phase-2 learning rate, compared on the seed-0 validation split (Section 5.3). "
  "(ii) *There is no early stopping and no checkpoint selection*; each model is trained for a fixed number of epochs and its final weights are used. (iii) *The test split is evaluated once per trained model.* "
  "These rules give **test-set isolation**: the test images do not influence training, model selection or tuning. They do not give a duplicate-free evaluation, which is a separate property treated next. "
  "For this reason we avoid the term “leakage-free” and use “test-set isolated” throughout.")
H2("3.4  Near-duplicate analysis")
P("TrashNet contains several photographs of the same physical object taken from slightly different angles, and no object identifier is distributed with the data. We therefore estimate similarity from image content. "
  "A perceptual hash (dHash) turned out to be unreliable here: with plain, light backgrounds many different objects share nearly the same coarse structure, and hash distances do not separate same-object pairs from unrelated ones. "
  "We instead use the cosine similarity of the 1280-dimensional pooled embeddings of the frozen ImageNet backbone. For pairs with similarity ≥ 0.85, 98.7% of pairs belong to the same class, which indicates that the measure picks up object identity and not just background. "
  "We use a threshold of 0.80 and require equal class to link two images; the connected components of this graph are called *groups*. "
  f"At this threshold the {n_img:,} images form {n_groups:,} groups, of which {n_multi} contain more than one image ({imgs_multi} images in total; the largest group has {max_group} images). "
  "The embedding-based groups are a proxy, not ground-truth object identities: they can merge visually similar but distinct items and miss duplicates seen under a very different pose, so the analysis bounds the problem rather than removing it exactly.")
P(f"*Contamination of random splits.* For each of seeds 0–4 we count the test images that have a same-class training image with similarity ≥ 0.80. On average {cont_n:.0f} of 380 test images ({cont_frac:.0f}%) are affected. "
  f"These images are classified with {cont_acc_with:.1f}% accuracy, against {cont_acc_wo:.1f}% for the remaining test images (FT-C model). The proportions are smaller at stricter thresholds ({cr85[:,0].mean():.0f} images at 0.85 and {cr9[:,0].mean():.0f} images at 0.90), "
  "but the gap between the two subsets remains. This shows that a random split rewards recognition of previously seen objects, and that a part of the accuracy reported on TrashNet measures memorisation of objects and not generalisation to new ones.")
P(f"*Grouped splits.* To obtain a duplicate-robust estimate we re-draw the splits with `StratifiedGroupKFold` (six folds, shuffled with seed s; fold 0 is the test set, fold 1 the validation set and folds 2–5 the training set), so that no group appears in more than one partition. "
  f"The partitions contain on average {gtr:.0f} / {gva:.0f} / {gte:.0f} images. We run this protocol for seeds 0–4 and for the arms listed in Section 6.3.")
H2("3.5  Metrics")
P("Accuracy is the share of correctly classified test images. Because the classes are unequal, we also report macro-averaged F1, weighted F1, per-class precision, recall and F1, and the macro-averaged one-vs-rest area under the ROC curve (AUC) from the softmax probabilities. "
  "The pooled confusion matrix of Fig. 2(b) sums the counts of ten test sets that are drawn from the same 2,527 images and therefore overlap; it describes error patterns, not a single 3,800-image evaluation, and its counts must not be treated as independent observations.")
# ================================================================ 4 METHOD
H1("4  Model and Training Variants")
H2("4.1  Architecture and layer indexing")
P("The backbone is MobileNetV2 with width multiplier 1.0 and the ImageNet-1k weights of the *d-li14/mobilenetv2.pytorch* repository. These weights come from a different training run than the Keras weights used in many earlier studies, "
  "so absolute numbers need not match Keras-based reports. The 1000-way classifier is removed. On the 1280-channel output of the final 1×1 convolution we place global average pooling (GAP), a dense layer with 128 units and ReLU, dropout with rate 0.3 and a six-way output layer.")
P("*Indexing.* In the implementation `features[0]` is the stem convolution and `features[1]` to `features[17]` are the 17 inverted-residual blocks, which we number 0 to 16. The split point is `features[:15]`: the **lower part** is the stem and blocks 0–13 and is never trained, "
  "and the **upper part** is blocks 14–16 (`features[15:18]`) plus the final 1×1 convolution. The complete network has 2,388,614 parameters, distributed as follows: 697,792 in the lower part (29.2%), 1,113,920 in blocks 14–16, 412,160 in the 1×1 convolution and 164,742 in the head. "
  "Phase 1 therefore trains 164,742 parameters (6.9%) and Phase 2 trains 1,690,822 (70.8%).")
P("*Why this is called “partial” fine-tuning.* The 70.8% figure is a share of parameters, not of depth. The later blocks have the widest channel dimensions, so they hold most of the weights: the three tuned blocks are 3 of the 17 blocks (and the stem) but contain 1,113,920 of the 2,223,872 backbone parameters. "
  "The 14 lower blocks and the stem, which carry generic edge, texture and colour detectors, stay frozen. We therefore use “partial” in the sense of *top-block* fine-tuning, and we do not claim that only a small fraction of the weights changes.")
H2("4.2  Two-phase training and the 2×2 control")
P("*Phase 1* trains only the head, with the whole backbone frozen, for 10 epochs with Adam [11] at learning rate 10^−3^ and batch size 32 on un-augmented images. "
  "*Phase 2* starts from the Phase-1 head and runs 15 epochs with Adam at 10^−4^ and batch size 32. In the full recipe it trains the upper part and the head, with on-the-fly augmentation (rotation up to ±20°, shift up to ±10%, zoom up to ±15%, horizontal flip, reflection padding). "
  "Batch-normalisation layers stay in inference mode, so their statistics remain those learned on ImageNet. No augmentation is applied to validation or test images.")
P("The original design compared the Phase-1 head with a Phase-2 model that differs in four respects at once: 15 additional epochs, augmentation, trainable backbone blocks, and a lower learning rate. "
  "To separate them we add three control arms that use exactly the Phase-2 schedule (15 epochs, learning rate 10^−4^, same Phase-1 head, same random seed), and vary only the factors of interest (Table 3):")
BUL([
 "**HO-C** (frozen, clean): the head keeps training for 15 epochs on un-augmented features of a frozen backbone. It isolates the effect of training longer.",
 "**HO-A** (frozen, augmented): the head is trained on augmented images through the frozen backbone. Compared with HO-C it isolates augmentation without tuning.",
 "**FT-C** (tuned, clean): the upper part and head are tuned without augmentation. Compared with HO-C it isolates fine-tuning without augmentation.",
 "**FT-A** (tuned, augmented): the full recipe, which was called “two-phase” in the earlier version of this study.",
])
FIG("/home/claude/paper/fig1_pipeline_300dpi.png","**Fig. 1.** Overview of the study. Splits are drawn per seed either at random or grouped by near-duplicates. Phase 2 is run as a 2×2 control of backbone state (frozen or tuned) and augmentation. The imbalance remedy, the extra backbones and the statistical analysis are layered on top; the protocol is the same for all arms.",150)
H2("4.3  Handling the rare class")
P("Let N be the number of training images, C = 6 the number of classes and N~c~ the number of training images of class c. We compare four ways of training Phase 2 on the tuned and augmented model (and class weights additionally in Phase 1):")
P("*Cross-entropy (CE).* The standard loss, ℓ = −log p~y~, where p~y~ is the softmax probability of the true class.", indent=False)
P("*Class weights (CW).* The loss of an image of class y is multiplied by the inverse-frequency weight", indent=False, after=0)
EQ("w~c~ = N / (C · N~c~),    ℓ = −w~y~ log p~y~,", 1)
P("so that the weights average to one over the training set. For seed 0 they range from 0.71 (paper) to 3.07 (trash).", indent=False)
P("*Focal loss (FL).* The loss of Lin et al. [26] with focusing parameter γ = 2 and no class-dependent factor,", indent=False, after=0)
EQ("ℓ = −(1 − p~y~)^γ^ log p~y~.", 2)
P("*Oversampling (OS).* At every epoch N training images are drawn with replacement with probability proportional to 1/N~y~, so that every class is seen about equally often; the loss is plain cross-entropy.", indent=False)
P("In Phase 2 all variants start from the same unweighted Phase-1 head, so differences between them come from Phase 2 only. A head-only arm trained with class weights (Phase 1 only) separates the effect of weighting from that of tuning.")
H2("4.4  Efficient implementation")
P("Because the lower part of the backbone is frozen in both phases we compute its output once. For Phase 1 we cache the un-augmented lower features and train the head on pooled vectors. For Phase 2 we draw a fresh augmentation of every training image in every epoch, "
  "pass it once through the frozen lower part and store the result in half precision; all Phase-2 variants of a seed read from the same cache, so they see exactly the same augmented images. This makes the arms directly comparable and makes the experiments feasible on a two-core CPU. "
  "As a consequence, Phase-1 features come from un-augmented images and the Phase-2 variants share augmentation draws within a seed.")
TABLE("**Table 2.** Abbreviations used in the paper.", ["Abbreviation", "Meaning"],
      [["CE / CW / FL / OS", "cross-entropy / class weights / focal loss / oversampling"],
       ["HO-C, HO-A", "frozen backbone (head only), clean / augmented images (Phase-2 schedule)"],
       ["FT-C, FT-A", "tuned top blocks, clean / augmented images (FT-A is the full recipe)"],
       ["GAP", "global average pooling"],
       ["MACs", "multiply-accumulate operations for one 224×224 image"],
       ["BN", "batch normalisation"],
       ["CI", "confidence interval (bootstrap over seeds)"],
       ["NB test", "Nadeau–Bengio variance-corrected resampled paired *t*-test [20]"],
       ["AUC", "area under the ROC curve (macro one-vs-rest)"]],
      [38, 108], align=["l", "l"])
TABLE("**Table 3.** Arms of the study. “Backbone” states which weights are updated in Phase 2.", ["Arm", "Phase 1", "Phase 2 backbone", "Augm.", "Loss"],
      [["Baseline (head-only, CE)", "CE, 10 ep.", "—", "—", "—"],
       ["Head-only + CW", "CW, 10 ep.", "—", "—", "—"],
       ["HO-C", "CE", "frozen", "no", "CE"],
       ["HO-A", "CE", "frozen", "yes", "CE"],
       ["FT-C", "CE", "blocks 14–16, conv", "no", "CE"],
       ["FT-A", "CE", "blocks 14–16, conv", "yes", "CE"],
       ["FT-A + CW / FL / OS", "CE", "blocks 14–16, conv", "yes", "CW / FL / OS"]],
      [46, 24, 40, 14, 22], align=["l", "c", "c", "c", "c"])

# ================================================================ 5 SETUP
H1("5  Experimental Setup")
H2("5.1  Implementation, weights and hardware")
P(f"All experiments use PyTorch {E['torch'].split('+')[0]} [16], Python {E['python']} and, for the additional backbones, timm 1.0.31 [27], on a CPU-only virtual machine with two vCPUs ({E['cpu']}). Images are resized to 224×224 pixels with bilinear interpolation and normalised with the ImageNet mean and standard deviation. "
  "The seed controls the data split, the head initialisation, the mini-batch order and the augmentation draws. The MobileNetV2 weights are those of d-li14/mobilenetv2.pytorch; the MobileNetV3-Large and EfficientNet-B0 weights are the timm release files "
  "`mobilenetv3_large_100_ra-f55367f5.pth` and `efficientnet_b0_ra-3dd342df.pth`. Table 4 lists all settings.")
TABLE("**Table 4.** Complete list of settings.", ["Item", "Setting"],
      [["Backbone / weights", "MobileNetV2 (width 1.0), ImageNet-1k weights from d-li14/mobilenetv2.pytorch"],
       ["Input", "224 × 224 RGB, ImageNet mean/std normalisation"],
       ["Head", "GAP → Dense(128, ReLU) → Dropout(0.3) → Dense(6)"],
       ["Random splits", "stratified 70/15/15, seeds 0–9, one split per seed"],
       ["Grouped splits", "StratifiedGroupKFold (6 folds), groups from cosine ≥ 0.80 within class, seeds 0–4"],
       ["Phase 1", "head only; Adam, lr 1e-3; 10 epochs; batch 32; no augmentation"],
       ["Phase 2", "15 epochs; Adam, lr 1e-4; batch 32; trainable set per arm (Table 3)"],
       ["BatchNorm", "inference mode (statistics frozen) in both phases"],
       ["Augmentation", "rotation ±20°, shift ±10%, zoom ±15%, horizontal flip; reflection padding"],
       ["Class weights / focal", "w~c~ = N / (C · N~c~) / γ = 2, no α term"],
       ["Early stopping", "none; final epoch evaluated once on the test split"],
       ["Extra backbones", "MobileNetV3-Large, EfficientNet-B0 (timm), seeds 0–2, last stages + neck tuned, BN frozen"],
       ["Hardware", "2 vCPU, CPU only (no GPU)"]],
      [38, 108], align=["l", "l"])
H2("5.2  Statistical analysis")
P(f"For every arm we report the mean and sample standard deviation over seeds. A comparison of two arms uses the per-seed difference of the metric, and we report four quantities. "
  "(i) A **bootstrap 95% confidence interval** of the mean difference, obtained with 20,000 resamples of the seeds. "
  "(ii) The **Nadeau–Bengio (NB) corrected resampled t-test** [20]: repeated random splits share training and test images, so the variance of the mean difference is multiplied by (1/k + n~test~/n~train~) instead of 1/k, where k is the number of seeds; this removes the over-optimism of the naive paired test. "
  "(iii) An **exact sign-flip permutation test** over the seeds, which makes no normality assumption; with ten seeds its smallest attainable *p*-value is 0.002 and with five seeds 0.063. "
  "(iv) **Holm step-down correction** [21] of the NB *p*-values over the twelve planned accuracy comparisons of Table 7. The number of seeds in which one arm beats the other is given as well. "
  "We also report the uncorrected paired *t*-test *p*-value in Table 7 for comparison with earlier practice, and we use it only to show how optimistic it is. "
  "Even the corrected tests describe variation over splits of this one dataset; they do not support claims about other datasets.")
H2("5.3  Choice of the Phase-2 learning rate")
P("The Phase-2 learning rate was the only tuned hyper-parameter. A pilot run on seed 0 compared 10^−5^ and 10^−4^ on the validation split only. The larger rate gave a validation accuracy of 91.0% and the smaller one 90.0%, so 10^−4^ was used for all arms and seeds. "
  "The pilot used 10 Phase-2 epochs; the main experiments use 15, a value fixed after measuring the training speed and not tuned. The same learning rate was used for all control arms and for the additional backbones, which is a limitation for the latter (Section 8).")

# ================================================================ 6 RESULTS
H1("6  Results")
H2(f"6.1  Main comparison on random splits ({NS} seeds)")
rows = [[NAMES[v], ms(v, "acc"), ms(v, "macro_f1"), ms(v, "weighted_f1"), ms(v, "trash_f1"), ms(v, "auc_macro", 2)] for v in ORDER]
TABLE(f"**Table 5.** Test-set results on random splits (%, mean ± SD over {NS} seeds; each seed has its own split). AUC is the macro one-vs-rest area under the ROC curve.",
      ["Arm", "Accuracy", "Macro-F1", "Weighted-F1", "Trash F1", "AUC"], rows, [50, 21, 21, 22, 21, 19], bold_rows=(5,), fs=8)
P(f"Table 5 reports all nine arms. The full recipe (FT-A) reaches {A['p2_ce']:.1f}% accuracy and {mean('p2_ce','macro_f1'):.1f}% macro-F1, against {A['p1_ce']:.1f}% and {mean('p1_ce','macro_f1'):.1f}% for the head-only baseline. "
  f"The difference is {stat(tot_gain)}, so it is consistent in direction (ten of ten seeds) and survives Holm correction. The three control arms lie in between, in the order HO-C ({A['p2_hoc']:.1f}%), HO-A ({A['p2_hoa']:.1f}%), FT-C ({A['p2_ftc']:.1f}%), which already indicates that the +{tot_gain['diff']:.1f} points are shared among several factors.")
H2("6.2  What produces the gain? The 2×2 control")
P(f"Table 6 and Fig. 2(a) decompose the gain. Training the head for 15 further epochs on clean frozen features (HO-C) adds {hoc_gain['diff']:+.1f} points over the baseline (CI [{hoc_gain['ci'][0]:+.1f}, {hoc_gain['ci'][1]:+.1f}], *p*~NB~ = {fp(hoc_gain['p_nb'])}). "
  f"Adding augmentation to the frozen model (HO-A − HO-C) gives {aug_hoc['diff']:+.1f} points, and tuning the backbone without augmentation (FT-C − HO-C) gives {ft_hoc['diff']:+.1f} points (CI [{ft_hoc['ci'][0]:+.1f}, {ft_hoc['ci'][1]:+.1f}]). "
  f"Combining tuning and augmentation (FT-A) adds a further {ce_ftc['diff']:+.1f} points over FT-C and {ce_hoa['diff']:+.1f} points over HO-A (CI [{ce_hoa['ci'][0]:+.1f}, {ce_hoa['ci'][1]:+.1f}], *p*~NB~ = {fp(ce_hoa['p_nb'])}, Holm-adjusted *p* = {fp(ce_hoa['p_holm'])}).")
P(f"The factorial contrasts, computed from the four 2×2 cells (HO-C, HO-A, FT-C, FT-A), are as follows. The **fine-tuning main effect** is {FT['diff']:+.1f} points (95% CI [{FT['ci'][0]:+.1f}, {FT['ci'][1]:+.1f}], better in {FT['wins']}/10 seeds, *p*~NB~ = {fp(FT['p_nb'])}, Holm-adjusted *p* = {fp(FT['p_holm'])}). "
  f"The **augmentation main effect** is {AUG['diff']:+.1f} points (CI [{AUG['ci'][0]:+.1f}, {AUG['ci'][1]:+.1f}], *p*~NB~ = {fp(AUG['p_nb'])}), and the **interaction** is {INT['diff']:+.1f} points (CI [{INT['ci'][0]:+.1f}, {INT['ci'][1]:+.1f}], *p*~NB~ = {fp(INT['p_nb'])}). "
  "Hence fine-tuning is the only factor whose individual effect is reliable. The augmentation effect is positive but its interval includes zero, and the data neither confirm nor exclude an interaction.")
rows6 = []
for lab, a_, b_ in [("Total gain: FT-A − baseline", "p2_ce", "p1_ce"), ("Longer training: HO-C − baseline", "p2_hoc", "p1_ce"), ("Augmentation (frozen): HO-A − HO-C", "p2_hoa", "p2_hoc"),
                    ("Tuning (clean): FT-C − HO-C", "p2_ftc", "p2_hoc"), ("Tuning + aug. vs frozen + aug.: FT-A − HO-A", "p2_ce", "p2_hoa"), ("Augmentation (tuned): FT-A − FT-C", "p2_ce", "p2_ftc"),
                    ("HO-A − baseline", "p2_hoa", "p1_ce"), ("Class weights: FT-A+CW − FT-A", "p2_cw", "p2_ce"), ("Focal loss: FT-A+FL − FT-A", "p2_focal", "p2_ce"),
                    ("Oversampling: FT-A+OS − FT-A", "p2_os", "p2_ce"), ("Oversampling − class weights", "p2_os", "p2_cw"), ("CW in Phase 2 vs head-only CW", "p2_cw", "p1_cw")]:
    c = comp(a_, b_); rows6.append([lab, f"{c['diff']:+.2f}", f"[{c['ci'][0]:+.1f}, {c['ci'][1]:+.1f}]", f"{c['wins']}/10", fp(c['p_naive']), fp(c['p_nb']), fp(c['p_flip']), fp(c['p_holm'])])
TABLE("**Table 6.** Paired accuracy differences (percentage points) over ten seeds with bootstrap CI, number of seeds won, uncorrected paired *t* *p*-value (naive), Nadeau–Bengio *p* (NB), sign-flip *p* and Holm-adjusted NB *p* over the twelve comparisons.",
      ["Comparison", "Δ acc.", "95% CI", "Wins", "naive p", "NB p", "flip p", "Holm p"], rows6, [56, 12, 21, 10, 13, 11, 11, 11], bold_rows=(0,), fs=7.5)
P("The naive and corrected columns of Table 6 illustrate the point made in Section 5.2: the uncorrected *p*-values are smaller than the NB values, by up to two orders of magnitude for the largest effects, and several comparisons that look significant with the naive test (for example HO-A against the baseline) are not once the split overlap is accounted for. "
  "Only the total gain and the class-weighted tuning gain survive Holm correction among the planned comparisons; the FT-A over HO-A contrast has a Holm-adjusted *p* above 0.05.")
FIG("/home/claude/paper/fig2_results_300dpi.png",
    "**Fig. 2.** (a) Test accuracy (mean ± SD) of the baseline and of the four cells of the 2×2 control on random splits (10 seeds) and on duplicate-grouped splits (5 seeds; the frozen-clean arm was not run on grouped splits). The y-axis starts at 78%. "
    "(b) Confusion matrix of the full recipe summed over the ten random test sets (counts, 3,800 predictions; shading is the row-normalised rate). The ten test sets overlap, so the counts are pooled predictions and not independent observations.", 150)
H2("6.3  Duplicate-grouped evaluation")
rowsg = [[NAMES[v], msg(v, "acc"), msg(v, "macro_f1"), msg(v, "trash_f1"), ms(v, "acc") if True else ""] for v in ["p1_ce", "p1_cw", "p2_hoa", "p2_ftc", "p2_ce", "p2_cw", "p2_os"]]
# random-split accuracy restricted to same five seeds for a fair like-for-like column
def ms5(v): a = np.array([R[s][v]["test"]["acc"] for s in SG]) * 100; return f"{a.mean():.1f}"
rowsg = [[NAMES[v], msg(v, "acc"), msg(v, "macro_f1"), msg(v, "trash_f1"), ms5(v)] for v in ["p1_ce", "p1_cw", "p2_hoa", "p2_ftc", "p2_ce", "p2_cw", "p2_os"]]
TABLE("**Table 7.** Test results (%) on duplicate-grouped splits (mean ± SD, seeds 0–4) and mean accuracy of the same arms on the random splits of the same five seeds. Grouped test sets contain about 420 images and are not the same images as the random ones.",
      ["Arm", "Accuracy", "Macro-F1", "Trash F1", "Acc. random (same seeds)"], rowsg, [58, 24, 24, 24, 28], bold_rows=(4,), fs=8)
P(f"On grouped splits the head-only baseline falls to {GA['p1_ce']:.1f}% (random splits, same seeds: {ms5('p1_ce')}%) and the full recipe to {GA['p2_ce']:.1f}% (random: {ms5('p2_ce')}%). Random-split accuracy is therefore higher than accuracy on unseen object groups by {float(ms5('p1_ce'))-GA['p1_ce']:.1f} points for the baseline and {float(ms5('p2_ce'))-GA['p2_ce']:.1f} points for the full recipe. "
  f"The benefit of the recipe is at least as large on grouped splits: {stat(gtot, holm=False)}. With five seeds this cannot be called significant after correction (Holm-adjusted *p* = {fp(gtot['p_holm'])}, smallest attainable sign-flip *p* = 0.063), but the direction is the same in all five seeds. "
  f"The order of the control arms is preserved (HO-A {GA['p2_hoa']:.1f}%, FT-C {GA['p2_ftc']:.1f}%, FT-A {GA['p2_ce']:.1f}%). The imbalance remedies again do not separate: class weighting is {GA['p2_cw']-GA['p2_ce']:+.1f} and oversampling {GA['p2_os']-GA['p2_ce']:+.1f} points relative to FT-A, "
  "with intervals that include zero. The seed-to-seed variation is larger on grouped splits because groups of very different sizes fall into the test fold.")
H2("6.4  Imbalance remedies and per-class results")
P(f"Among the tuned and augmented variants, class weighting is {cw_ce['diff']:+.1f}, focal loss {fo_ce['diff']:+.1f} and oversampling {os_ce['diff']:+.1f} points of accuracy relative to plain CE (all intervals include zero; all Holm-adjusted *p* = 1.00). "
  f"The seed-to-seed standard deviation of about {g('p2_ce','acc').std(ddof=1)*100:.1f} points is larger than any of these differences. We therefore conclude that none of the three remedies improves overall accuracy on TrashNet, and the ranking among them cannot be established.")
P("Tables 8–10 give precision, recall and F1 for every class and every arm, averaged over the ten random splits. The previous version gave these only for two arms, which left statements about class weighting unsupported by a table.")
def pc_table(key, title):
    rows_ = [[NAMES[v]] + [f"{x:.1f}" for x in PC[v][key]] for v in ORDER]
    TABLE(title, ["Arm"] + C, rows_, [52, 16, 14, 14, 14, 16, 14], bold_rows=(5,), fs=8)
pc_table("precision", "**Table 8.** Per-class precision (%), mean over ten random splits.")
pc_table("recall", "**Table 9.** Per-class recall (%), mean over ten random splits.")
pc_table("f1", "**Table 10.** Per-class F1 score (%), mean over ten random splits.")
best_r = max(ORDER, key=lambda v: tr_r[v]); best_p = max(ORDER, key=lambda v: tr_p[v])
P(f"The *trash* class is the weakest in every arm. For the full recipe its precision is {tr_p['p2_ce']:.1f}% and its recall {tr_r['p2_ce']:.1f}%. Class weighting gives a trash recall of {tr_r['p2_cw']:.1f}% and a precision of {tr_p['p2_cw']:.1f}%; "
  f"the highest trash recall is obtained by {NAMES[best_r]} ({tr_r[best_r]:.1f}%) and the highest trash precision by {NAMES[best_p]} ({tr_p[best_p]:.1f}%). Oversampling gives {tr_p['p2_os']:.1f}% precision and {tr_r['p2_os']:.1f}% recall, with the highest trash F1 ({PC['p2_os']['f1'][5]:.1f}%) of the tuned arms. "
  "Weighting thus moves the operating point of the rare class, trading precision for recall, without improving the overall score. With only about 20 trash images in each test set, a change of three predictions moves the class F1 by several points, so these differences stay descriptive.")
P("Errors concentrate on a few pairs (Fig. 2(b)): glass is confused with metal and plastic, cardboard with paper, and trash with paper (and the reverse). These are classes whose members share colour and texture, and the confusions are also present in the head-only model; tuning mostly reduces them without removing them.")
H2("6.5  Other lightweight backbones")
mv, ef = "mobilenetv3_large_100", "efficientnet_b0"
TABLE("**Table 11.** Head-only and tuned results (FT-A recipe) for three backbones on seeds 0–2 of the random splits (accuracy %, mean ± SD over three seeds). “Wins” counts seeds in which tuning beats head-only. Descriptive only.",
      ["Backbone", "Params (M)", "Head-only", "Tuned", "Tuned macro-F1", "Δ acc.", "Wins"],
      [["MobileNetV2", "2.39", f"{m2_p1:.1f}", f"{m2_p2:.1f}", f"{m2_f1:.1f}", f"{m2_p2-m2_p1:+.1f}", f"{m2_w}/3"],
       arch_row("MobileNetV3-Large", mv), arch_row("EfficientNet-B0", ef)],
      [36, 18, 24, 24, 22, 12, 12], fs=8)
P("To test whether the observations are specific to MobileNetV2 we ran the head-only and tuned recipes on MobileNetV3-Large [17] and EfficientNet-B0 [18] for three seeds. For both, we tune the last stages (from `blocks[5]` on, with the convolutional neck) with batch normalisation frozen and keep all other settings, including the learning rate, from the MobileNetV2 experiments. "
  "Tuning improves accuracy for both backbones on average (Table 11), and for MobileNetV3-Large in all three seeds, but the three-seed results carry no significance claim; the EfficientNet-B0 gain is smaller and not consistent across seeds. "
  "We did not run a ResNet-18 baseline because no pretrained weights could be obtained in the environment. The additional backbones also use the augmentation-plus-tuning recipe only, so the 2×2 decomposition was not repeated for them.")
H2("6.6  Model size and CPU latency")
TABLE("**Table 12.** Cost of the MobileNetV2 classifier. Latency is measured on a server CPU and is a proxy only.", ["Quantity", "Value"],
      [["Parameters", f"{E['params_m']:.2f} M"], ["Multiply-accumulates (224×224)", f"{E['macs_m']:.0f} M"], ["float32 weights", f"{E['fp32_size_mb']:.1f} MB"],
       ["Median latency, 1 thread", f"{bench1['median_ms']:.1f} ms (95th pct. {bench1['p95_ms']:.1f} ms)"], ["Median latency, 2 threads", f"{bench2['median_ms']:.1f} ms (95th pct. {bench2['p95_ms']:.1f} ms)"],
       ["Processor", f"{E['cpu']} (server class)"]], [60, 80], align=["l", "l"])
P("Model size and operation count are properties of the network and transfer to other hardware. The latency was measured on a server processor and says little about phones or microcontrollers, whose memory systems, instruction sets and thermal limits differ. "
  "We therefore claim only that the model is small and inexpensive in operation count; we do not claim a demonstrated edge deployment.")
# ================================================================ 7 DISCUSSION
H1("7  Discussion")
H2("7.1  What the study adds")
P(f"The earlier reading of this experiment, that the +{tot_gain['diff']:.1f}-point improvement is the effect of fine-tuning, is not supported as stated. The 2×2 control shows that the improvement is shared: roughly {hoc_gain['diff']:.1f} points come from training for longer, "
  f"{aug_hoc['diff']:.1f} to {AUG['diff']:.1f} points from augmentation, and {ft_hoc['diff']:.1f} to {FT['diff']:.1f} points from tuning the backbone, depending on the contrast used; only the pooled fine-tuning main effect has an interval that excludes zero. "
  "The practical reading is that a user who wants the best accuracy should use all three, but a reader who attributes the gain of a published recipe to fine-tuning alone may be overstating what the data show. The study also provides a template for a minimal control, three extra arms that reuse the Phase-2 schedule, that can be added to similar transfer-learning papers at low cost.")
P("The second finding concerns evaluation. A quarter of the random-split test images have a near-duplicate of the same class in training, and these images are classified much better than the rest. Accuracies measured on random splits of TrashNet therefore mix two abilities, "
  "recognising a new object and recognising another photograph of a known object, and numbers from different papers that use different random splits are not comparable at the level of one or two points. The duplicate-grouped numbers are lower and are the more conservative estimate of what a deployed bin would see on new items, although the embedding-based groups are only a proxy for object identity.")
H2("7.2  Why imbalance remedies do not help here")
P("With 137 trash images against 594 for paper, the imbalance is moderate, and the pretrained features already separate the classes well (AUC above 98.5% in all arms). Class weighting shifts the decision threshold of the rare class, which raises its recall at the price of its precision, and oversampling raises recall a little without lowering precision (Tables 8–9), "
  "but the number of trash test images is small and the net effect on accuracy is within noise. Focal loss reduces the weight of confident examples, which are the majority here, but also does not help. Studies that report gains from such remedies on this dataset should therefore show the seed-to-seed spread and a corrected test.")
H2("7.3  Comparison with related studies")
P("A numerical comparison with other TrashNet papers is not made in a table because splits, resolutions, pretrained weights and protocols differ, and some of those papers evaluate on validation data or on different class sets; the duplicate analysis above shows that even the choice of random split can move accuracy by more than the gaps usually reported. "
  "Relative to Andriani et al. [8], whose three-class study reports a single run on an 80:20 train/validation split and no separate test set, our contribution is the decomposition of the recipe, the ten-seed statistics and the grouped evaluation. "
  "Relative to larger-scale and deployment-oriented work [14, 15], our results do not address new datasets or distillation; they concern the reliability of the evaluation that such work builds on.")
H2("7.4  Practical recommendations")
BUL(["Report results on duplicate-grouped splits, or at least report the share of test images with a same-class near-duplicate in the training set.",
     "When a recipe changes several things at once (schedule length, augmentation, trainable layers), add control arms that change one thing at a time.",
     "Use at least ten seeds, a variance-corrected test, bootstrap intervals and a multiplicity correction; do not rely on p-values from five overlapping splits.",
     "Describe deployment in terms of parameters and operation counts unless measurements on the target device are available."])

# ================================================================ 8 LIMITATIONS
H1("8  Limitations")
BUL([
 "**One dataset.** TrashNet is small, photographed on plain backgrounds and not representative of conveyor-belt or street conditions. The conclusions concern this benchmark.",
 "**Duplicate groups are estimated.** Groups come from embedding similarity, not from object identifiers. Grouped splits are stricter than random ones, but they are not guaranteed to be duplicate-free, and the grouped experiments use five seeds and omit the HO-C and FL arms.",
 "**Statistics.** Ten seeds give moderate power. The NB test corrects the variance for overlapping splits under a resampling model that is itself an approximation. Differences of one point or less between arms cannot be resolved.",
 "**Single learning rate.** The Phase-2 learning rate was chosen once on seed-0 validation data and reused for all arms and backbones. It may not be optimal for the other backbones or for the clean arms, and the 2×2 comparison inherits this choice.",
 "**Pretrained weights.** The MobileNetV2 weights come from a PyTorch re-implementation; Keras weights could give different absolute numbers.",
 "**Other backbones.** MobileNetV3-Large and EfficientNet-B0 were run for three seeds and the full recipe only; ResNet-18 was not run.",
 "**Deployment.** Latency was measured on a server CPU only. No mobile or embedded measurement was made.",
 "**Pooled confusion matrix.** Fig. 2(b) sums ten overlapping test sets and is descriptive.",
])

# ================================================================ 9 CONCLUSION
H1("9  Conclusion")
P(f"A two-phase MobileNetV2 recipe improves TrashNet test accuracy from {A['p1_ce']:.1f}% to {A['p2_ce']:.1f}% over ten random splits, but a 2×2 control shows that the improvement is shared between longer training, augmentation and backbone tuning, "
  f"and only the tuning effect ({FT['diff']:+.1f} points) is individually reliable. Roughly a quarter of the test images of a random split have a same-class near-duplicate in training, and on duplicate-grouped splits accuracy falls to {GA['p1_ce']:.1f}% (baseline) and {GA['p2_ce']:.1f}% (full recipe), "
  "while the benefit of the recipe remains. Class weighting, focal loss and oversampling do not improve overall accuracy. MobileNetV3-Large and EfficientNet-B0 show the same direction of effect on three seeds. "
  "We recommend that future TrashNet studies report grouped splits, one-factor-at-a-time controls and corrected statistics, and we release the code and splits needed to do so. Future work includes ground-truth object identities, additional datasets, tuned learning rates per arm, and on-device measurements.")

# ================================================================ AVAILABILITY
H1("Data and Code Availability")
P("TrashNet is publicly available [9]; we used the “dataset-resized” release. All code, per-seed random and grouped split files, duplicate groups, result files, training logs, the environment file (`requirements.txt`), and the scripts that generate every table and both figures are provided in a reproducibility package. "
  "Repository: **[public repository URL to be inserted by the author before submission]**. The package contains: `code/` (training, duplicate analysis, statistics, latency benchmark), `splits/` (file-name lists for every seed and for the grouped splits, with the group assignments), `results/` (per-seed JSON of every arm and the final statistics file), "
  "`logs/` (complete training logs), `figures/` (the figure scripts and the 300-dpi figures), and a `README` with step-by-step instructions that reproduce Tables 5–12. The random seeds are 0–9 (random splits) and 0–4 (grouped splits); pretrained weight sources are given in Section 5.1.", indent=False)
P("*Declarations.* [Funding and competing-interest statements to be completed by the author.]", indent=False)

# ================================================================ REFERENCES
H1("References")
REFS = [
 "Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.-C.: MobileNetV2: Inverted Residuals and Linear Bottlenecks. In: Proc. IEEE/CVF CVPR, pp. 4510–4520 (2018). https://doi.org/10.1109/CVPR.2018.00474",
 "Xu, X., Qi, X., Diao, X.: Reach on Waste Classification and Identification by Transfer Learning and Lightweight Neural Network. arXiv:2212.11050 (2022)",
 "Yong, L., Ma, L., Sun, D., Du, L.: Application of MobileNetV2 to Waste Classification. PLOS ONE 18(3), e0282336 (2023). https://doi.org/10.1371/journal.pone.0282336",
 "Poudel, S., Poudyal, P.: Classification of Waste Materials using CNN Based on Transfer Learning. In: Proc. 14th Annual Meeting of the Forum for Information Retrieval Evaluation (FIRE), ACM, pp. 29–33 (2022)",
 "Srivatsan, K., Dhiman, S., Jain, A.: Waste Classification using Transfer Learning with Convolutional Neural Networks. IOP Conf. Ser.: Earth Environ. Sci. 775(1), 012010 (2021)",
 "Kure, H.I., Retnakumari, J., Nwajana, A.O., Ismail, U.M., Romo, B.A., Egho-Promise, E.: Integrating Trustworthy Artificial Intelligence with Energy-Efficient Robotic Arms for Waste Sorting. arXiv:2510.17408 (2025)",
 "Frost, S., Tor, B., Agrawal, R., Forbes, A.G.: CompostNet: An Image Classifier for Meal Waste. In: 2019 IEEE Global Humanitarian Technology Conference (GHTC), pp. 1–4 (2019)",
 "Andriani, F., Yuliati, A., Yaturahmah, A., Mutmainah, S.: MobileNetV2 Transfer Learning Implementation for Waste Classification of Recyclable Materials (Plastic, Paper, Glass). Indonesian Applied Research Computing and Informatics 1(2), 37–46 (2026). Year as given by the journal's publication record (publication-history dates in February 2026)",
 "Thung, G., Yang, M.: Classification of Trash for Recyclability Status. Stanford CS229 Project Report (2016)",
 "Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., Adam, H.: MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv:1704.04861 (2017)",
 "Kingma, D.P., Ba, J.: Adam: A Method for Stochastic Optimization. In: Proc. ICLR (2015)",
 "Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., Fei-Fei, L.: ImageNet: A Large-Scale Hierarchical Image Database. In: Proc. IEEE CVPR, pp. 248–255 (2009). https://doi.org/10.1109/CVPR.2009.5206848",
 "Yosinski, J., Clune, J., Bengio, Y., Lipson, H.: How Transferable are Features in Deep Neural Networks? In: Advances in Neural Information Processing Systems 27, pp. 3320–3328 (2014)",
 "Verber, D., Grneva, T., Dugonik, J.: Image-Based Waste Classification Using a Hybrid Deep Learning Architecture with Transfer Learning and Edge AI Deployment. Mathematics 14(7), 1176 (2026). https://doi.org/10.3390/math14071176",
 "Theera-Ampornpunt, N., Treepong, P., Jannu, P., Sritongkul, A.: Sustainable Waste Management Through Deep Learning: A Knowledge Distillation Framework for Real-Time Garbage Classification. Sustainability 18(13), 6392 (2026). https://doi.org/10.3390/su18136392",
 "Paszke, A., et al.: PyTorch: An Imperative Style, High-Performance Deep Learning Library. In: Advances in Neural Information Processing Systems 32 (2019)",
 "Howard, A., Sandler, M., Chu, G., et al.: Searching for MobileNetV3. In: Proc. IEEE/CVF ICCV, pp. 1314–1324 (2019). https://doi.org/10.1109/ICCV.2019.00140",
 "Tan, M., Le, Q.V.: EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. In: Proc. ICML, PMLR 97, pp. 6105–6114 (2019)",
 "Barz, B., Denzler, J.: Do We Train on Test Data? Purging CIFAR of Near-Duplicates. J. Imaging 6(6), 41 (2020). https://doi.org/10.3390/jimaging6060041",
 "Nadeau, C., Bengio, Y.: Inference for the Generalization Error. Machine Learning 52(3), 239–281 (2003)",
 "Holm, S.: A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian Journal of Statistics 6(2), 65–70 (1979)",
 "Dietterich, T.G.: Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. Neural Computation 10(7), 1895–1923 (1998)",
 "Demšar, J.: Statistical Comparisons of Classifiers over Multiple Data Sets. Journal of Machine Learning Research 7, 1–30 (2006)",
 "Bouthillier, X., Delaunay, P., Bronzi, M., et al.: Accounting for Variance in Machine Learning Benchmarks. In: Proc. Machine Learning and Systems (MLSys) 3 (2021)",
 "Cui, Y., Jia, M., Lin, T.-Y., Song, Y., Belongie, S.: Class-Balanced Loss Based on Effective Number of Samples. In: Proc. IEEE/CVF CVPR, pp. 9268–9277 (2019)",
 "Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P.: Focal Loss for Dense Object Detection. In: Proc. IEEE ICCV, pp. 2980–2988 (2017)",
 "Wightman, R.: PyTorch Image Models (timm). GitHub repository (2019). https://github.com/huggingface/pytorch-image-models",
]
for i, r in enumerate(REFS, 1):
    p = doc.add_paragraph(); p.paragraph_format.left_indent = Mm(6); p.paragraph_format.first_line_indent = Mm(-6)
    p.paragraph_format.space_after = Pt(2); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    add_runs(p, f"{i}.  {r}", 8.5)

fp_ = sec.footer.paragraphs[0]; fp_.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = fp_.add_run(); r.font.size = Pt(9)
for t, txt in (("begin", None), (None, "PAGE"), ("end", None)):
    if t: e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t)
    else: e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt
    r._r.append(e)
hp = sec.header.paragraphs[0]; hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
hr = hp.add_run("Fine-Tuning, Augmentation and Duplicates in Two-Phase MobileNetV2 Transfer Learning on TrashNet"); hr.font.size = Pt(8.5); hr.italic = True
doc.core_properties.title = "Disentangling Fine-Tuning, Augmentation and Training Length in Two-Phase MobileNetV2 Transfer Learning"
doc.core_properties.author = "Shalza Aggarwal"
for z in doc.settings.element.iter(qn("w:zoom")): z.set(qn("w:percent"), "100")
doc.save("/home/claude/paper/Waste_Classification_Revised.docx")
print("saved")
