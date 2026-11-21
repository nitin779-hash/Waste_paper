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

