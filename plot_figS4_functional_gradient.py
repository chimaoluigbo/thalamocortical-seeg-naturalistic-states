"""
plot_figS4_functional_gradient.py
-----------------------------------
Generates Supplementary Figure S4:
  Functional (not anatomical) gradient of thalamocortical PC1 scores
  across thalamic nuclei.

Key result:
  Functional gradient (group rank vs PC1): Spearman r = +0.869, p = 0.005
  Anatomical gradient (y_mni vs PC1):      Spearman r = -0.476, p = 0.233

Run from seeg_study directory:
    source /home/chima/seeg_env/bin/activate
    cd /mnt/c/Users/chima/seeg_study
    python plot_figS4_functional_gradient.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(".")
RES_DIR  = DATA_DIR / "outputs/results"
FIG_DIR  = DATA_DIR / "outputs/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

GT_CSV   = RES_DIR / "thalamic_nuclei_ground_truth.csv"
CONN_CSV = RES_DIR / "step5_connectivity.csv"

# ── Functional groups ──────────────────────────────────────────────────────────
MANUSCRIPT_NUCLEI = ["VLP", "Pul", "MD", "CM", "VPL", "AV", "VA", "MGN"]

FUNC_GROUPS = {
    "Motor relay\n(VA, VLP, VPL)": {
        "nuclei": ["VA", "VLP", "VPL"],
        "color":  "#4E9AF1",
        "rank":   1,
        "desc":   "Basal ganglia/cerebellar\n→ premotor cortex",
    },
    "Integrative\n(MD, CM, MGN)": {
        "nuclei": ["MD", "CM", "MGN"],
        "color":  "#66BB6A",
        "rank":   2,
        "desc":   "Multimodal / intralaminar\n→ prefrontal, cingulate",
    },
    "Limbic-association\n(AV, Pul)": {
        "nuclei": ["AV", "Pul"],
        "color":  "#EF5350",
        "rank":   3,
        "desc":   "Papez circuit / salience\n→ cingulate, hippocampus",
    },
}

NUC_GROUP = {nuc: (grp, info["rank"], info["color"])
             for grp, info in FUNC_GROUPS.items()
             for nuc in info["nuclei"]}

# ── Load & compute PC1 projections ────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

gt   = pd.read_csv(GT_CSV)
conn = pd.read_csv(CONN_CSV)

feat_cols = [c for c in ["icoh_delta","icoh_theta","icoh_alpha",
                          "icoh_beta","icoh_lgamma","icoh_broadband"]
             if c in conn.columns]

all_sub = conn[conn["nucleus"]=="all_thalamus"].dropna(subset=feat_cols)
scaler  = StandardScaler().fit(all_sub[feat_cols].fillna(0).values)
pca_grp = PCA(n_components=2).fit(scaler.transform(
              all_sub[feat_cols].fillna(0).values))
var = pca_grp.explained_variance_ratio_

# Nucleus centroids
thal = gt[(gt["is_thalamic"]==True) &
          (gt["thalamic_nucleus"].isin(MANUSCRIPT_NUCLEI))].copy()
for c in ["x_mni","y_mni","z_mni"]:
    thal[c] = pd.to_numeric(thal[c], errors="coerce")
thal = thal.dropna(subset=["x_mni","y_mni","z_mni"])
centroids = (thal.groupby("thalamic_nucleus")
             .agg(x=("x_mni","mean"), y=("y_mni","mean"),
                  z=("z_mni","mean"), n=("x_mni","count"))
             .reset_index())

# Project each nucleus
rows = []
for nuc in MANUSCRIPT_NUCLEI:
    sub = conn[conn["nucleus"]==nuc].dropna(subset=feat_cols)
    if len(sub) < 3:
        continue
    proj = pca_grp.transform(scaler.transform(
           sub[feat_cols].mean().values.reshape(1,-1)))[0]
    c_row = centroids[centroids["thalamic_nucleus"]==nuc]
    if c_row.empty:
        continue
    grp_label, grp_rank, grp_color = NUC_GROUP.get(nuc, ("Unknown",0,"#999"))
    rows.append({
        "nucleus":    nuc,
        "pc1":        float(proj[0]),
        "pc2":        float(proj[1]),
        "y_mni":      float(c_row["y"].values[0]),
        "n_contacts": int(c_row["n"].values[0]),
        "group":      grp_label,
        "group_rank": grp_rank,
        "color":      grp_color,
    })

df = pd.DataFrame(rows)

# ── Spearman correlations ──────────────────────────────────────────────────────
r_func, p_func = stats.spearmanr(df["group_rank"], df["pc1"])
r_anat, p_anat = stats.spearmanr(df["y_mni"],      df["pc1"])

print(f"Functional gradient: Spearman r = {r_func:.3f}, p = {p_func:.4f}")
print(f"Anatomical gradient: Spearman r = {r_anat:.3f}, p = {p_anat:.4f}")

# ── Figure ─────────────────────────────────────────────────────────────────────
MSTYLE = {"font.family":"sans-serif","font.size":11,
           "axes.spines.top":False,"axes.spines.right":False,
           "axes.linewidth":0.9,"pdf.fonttype":42}
plt.rcParams.update(MSTYLE)

fig = plt.figure(figsize=(18, 8))
gs  = GridSpec(1, 3, figure=fig, width_ratios=[1.5, 1.5, 1.2],
               wspace=0.42)

# ── Panel A: Functional gradient (group rank vs PC1) ──────────────────────────
axA = fig.add_subplot(gs[0])

# Jitter x within each group
np.random.seed(42)
for grp_label, info in FUNC_GROUPS.items():
    sub = df[df["group"]==grp_label]
    xs  = np.full(len(sub), info["rank"]) + \
          np.random.uniform(-0.12, 0.12, len(sub))
    axA.scatter(xs, sub["pc1"], c=info["color"], s=200,
                edgecolors="white", linewidths=1.5,
                zorder=4, alpha=0.92)
    # Label each nucleus
    for _, row in sub.iterrows():
        x_dot = info["rank"] + np.random.uniform(-0.12, 0.12)
        axA.annotate(row["nucleus"],
                     xy=(info["rank"], row["pc1"]),
                     xytext=(info["rank"] + 0.18, row["pc1"]),
                     fontsize=9, fontweight="bold",
                     color=info["color"],
                     va="center",
                     arrowprops=dict(arrowstyle="-", color="#CCCCCC",
                                     lw=0.5, shrinkA=8, shrinkB=0))

# Group mean bars
for grp_label, info in FUNC_GROUPS.items():
    sub = df[df["group"]==grp_label]
    mean_pc1 = sub["pc1"].mean()
    axA.hlines(mean_pc1, info["rank"]-0.28, info["rank"]+0.28,
               colors=info["color"], linewidth=3, zorder=3, alpha=0.85)
    axA.text(info["rank"], mean_pc1 + 0.08,
             f"mean={mean_pc1:+.2f}",
             ha="center", fontsize=8.5, color=info["color"], style="italic")

# Regression line
x_fit = np.linspace(0.7, 3.3, 100)
m, b  = np.polyfit(df["group_rank"], df["pc1"], 1)
axA.plot(x_fit, m*x_fit+b, color="#888888", lw=1.5,
         linestyle="--", alpha=0.6, zorder=1)

axA.axhline(0, color="black", lw=0.8, zorder=2)
axA.set_xticks([1, 2, 3])
axA.set_xticklabels(
    [g.replace("\n","\n") for g in FUNC_GROUPS.keys()],
    fontsize=10)
axA.set_ylabel("PC1 projection score\n(← passive / cognitive/sleep →)", fontsize=11)
axA.set_xlim(0.5, 4.0)
axA.set_facecolor("#F9F9F9")

# Significance annotation
sig_str = f"Spearman r = {r_func:.3f}\np = {p_func:.4f}"
axA.text(0.97, 0.97, sig_str,
         transform=axA.transAxes, fontsize=10,
         ha="right", va="top",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                   edgecolor="#AAAAAA", alpha=0.9))
axA.set_title("A   Functional gradient\n(thalamic circuit class → PC1)",
              fontsize=12, fontweight="bold", loc="left")

# Add passive/cognitive labels on y axis
axA.text(-0.22, 0.08, "Cognitive /\nSleep →",
         transform=axA.transAxes, fontsize=8.5,
         color="#B22222", rotation=90, ha="center", va="bottom")
axA.text(-0.22, 0.0, "← Passive",
         transform=axA.transAxes, fontsize=8.5,
         color="#3B5FA0", rotation=90, ha="center", va="top")

# ── Panel B: Anatomical gradient (y_mni vs PC1) — NOT significant ─────────────
axB = fig.add_subplot(gs[1])

for _, row in df.iterrows():
    axB.scatter(row["y_mni"], row["pc1"],
                c=row["color"], s=200,
                edgecolors="white", linewidths=1.5,
                zorder=4, alpha=0.92)
    axB.annotate(row["nucleus"],
                 xy=(row["y_mni"], row["pc1"]),
                 xytext=(row["y_mni"]+1.2, row["pc1"]+0.04),
                 fontsize=9, fontweight="bold", color=row["color"],
                 arrowprops=dict(arrowstyle="-", color="#CCCCCC",
                                 lw=0.5, shrinkA=8, shrinkB=0))

# Regression line
x_fit2 = np.linspace(df["y_mni"].min()-1, df["y_mni"].max()+1, 100)
m2, b2 = np.polyfit(df["y_mni"], df["pc1"], 1)
axB.plot(x_fit2, m2*x_fit2+b2, color="#888888", lw=1.5,
         linestyle="--", alpha=0.6)

axB.axhline(0, color="black", lw=0.8)
axB.set_xlabel("Anterior-posterior position\n(y_MNI; mm, more negative = more posterior)",
               fontsize=10)
axB.set_ylabel("PC1 projection score", fontsize=11)
axB.set_facecolor("#F9F9F9")

# Not significant annotation
sig_str2 = f"Spearman r = {r_anat:.3f}\np = {p_anat:.4f}  (n.s.)"
axB.text(0.97, 0.97, sig_str2,
         transform=axB.transAxes, fontsize=10,
         ha="right", va="top",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF8F8",
                   edgecolor="#DDAAAA", alpha=0.9))
axB.set_title("B   Anatomical gradient\n(anterior-posterior position → PC1)",
              fontsize=12, fontweight="bold", loc="left")

# Annotate VA and AV to show they are spatially close but functionally opposite
axB.annotate("VA & AV:\n4–5 mm apart\nbut opposite PC1 poles",
             xy=(df[df["nucleus"]=="VA"]["y_mni"].values[0],
                 df[df["nucleus"]=="VA"]["pc1"].values[0]),
             xytext=(-10, 0.3),
             fontsize=8.5, color="#555555",
             arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFEE",
                       edgecolor="#CCCC88", alpha=0.9))

# ── Panel C: Summary comparison bar chart ─────────────────────────────────────
axC = fig.add_subplot(gs[2])

labels   = ["Functional\ngradient\n(circuit class)", "Anatomical\ngradient\n(A-P position)"]
r_vals   = [r_func,  r_anat]
p_vals   = [p_func,  p_anat]
bar_cols = ["#E8573C", "#AAAAAA"]

bars = axC.bar([0, 1], [abs(r) for r in r_vals],
               color=bar_cols, width=0.5,
               edgecolor="white", linewidth=1.2, alpha=0.88)

# R² labels inside bars
for i, (r, p) in enumerate(zip(r_vals, p_vals)):
    sig = "***" if p < 0.01 else ("*" if p < 0.05 else "n.s.")
    axC.text(i, abs(r)/2, f"r = {r:+.3f}\nR² = {r**2:.3f}\n{sig}",
             ha="center", va="center", fontsize=10,
             color="white" if abs(r) > 0.5 else "#333333",
             fontweight="bold")

axC.set_xticks([0, 1])
axC.set_xticklabels(labels, fontsize=10)
axC.set_ylabel("|Spearman r|", fontsize=11)
axC.set_ylim(0, 1.0)
axC.axhline(0.5, color="#CCCCCC", lw=0.8, linestyle="--", alpha=0.7)
axC.text(1.02, 0.5, "Medium\neffect", transform=axC.get_yaxis_transform(),
         fontsize=7.5, color="#AAAAAA", va="center")
axC.set_facecolor("#F9F9F9")
axC.set_title("C   Functional vs anatomical\ngradient strength",
              fontsize=12, fontweight="bold", loc="left")

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_handles = [
    Line2D([0],[0], marker="o", color="w",
           markerfacecolor=info["color"], markersize=11,
           label=f"{grp.split(chr(10))[0]}  ({', '.join(info['nuclei'])})")
    for grp, info in FUNC_GROUPS.items()
]
fig.legend(handles=legend_handles, title="Functional group",
           loc="lower center", ncol=3, fontsize=9.5,
           title_fontsize=10, frameon=False,
           bbox_to_anchor=(0.5, -0.04))

fig.suptitle(
    "Supplementary Figure S4 — PC1 axis reflects thalamic functional hierarchy, not anatomy\n"
    f"PC1 explains {var[0]:.1%} of variance in thalamocortical iCoh profiles (n=27 subjects, 524 recordings)",
    fontsize=13, fontweight="bold", y=1.03
)

plt.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(str(FIG_DIR / f"FigS4_functional_gradient.{ext}"),
                dpi=300 if ext=="png" else None,
                bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved: FigS4_functional_gradient.pdf / .png")
print(f"\nKey stats for manuscript:")
print(f"  Functional gradient: Spearman r = {r_func:.3f}, p = {p_func:.4f}")
print(f"  Anatomical gradient: Spearman r = {r_anat:.3f}, p = {p_anat:.4f}")
print(f"  Functional R² = {r_func**2:.3f} vs anatomical R² = {r_anat**2:.3f}")
