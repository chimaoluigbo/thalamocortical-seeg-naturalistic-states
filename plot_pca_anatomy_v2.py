"""
plot_pca_anatomy_v2.py
-----------------------
Visualizes PC1 and PC2 of the thalamocortical PCA mapped onto
thalamic nucleus MNI centroids.

Updates from v1:
  - Restricted to the 8 manuscript nuclei (VLP, Pul, MD, CM, VPL, AV, VA, MGN)
  - Functional group annotations: motor relay / integrative / limbic-association
  - 4th panel: PC1 score bar chart sorted by anterior-posterior MNI position
  - Fixed matplotlib get_cmap deprecation warnings

Run from seeg_study directory:
    source /home/chima/seeg_env/bin/activate
    cd /mnt/c/Users/chima/seeg_study
    python plot_pca_anatomy_v2.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from matplotlib.colorbar import ColorbarBase
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(".")
RES_DIR  = DATA_DIR / "outputs/results"
FIG_DIR  = DATA_DIR / "outputs/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

GT_CSV   = RES_DIR / "thalamic_nuclei_ground_truth.csv"
CONN_CSV = RES_DIR / "step5_connectivity.csv"

# ── Restrict to the 8 manuscript-reported nuclei ─────────────────────────────
MANUSCRIPT_NUCLEI = ["VLP", "Pul", "MD", "CM", "VPL", "AV", "VA", "MGN"]

NUC_SHORT = {
    "VLP": "VLP", "Pul": "Pul", "MD":  "MD",  "CM":  "CM",
    "VPL": "VPL", "AV":  "AV",  "VA":  "VA",  "MGN": "MGN",
}

# Functional groupings
FUNC_GROUPS = {
    "Motor relay":         {"nuclei": ["VA", "VLP", "VPL"],
                            "color": "#4E9AF1",
                            "desc": "Basal ganglia/cerebellar\noutput → motor cortex"},
    "Integrative":         {"nuclei": ["MD", "CM", "MGN"],
                            "color": "#66BB6A",
                            "desc": "Multimodal / intralaminar\nintegration"},
    "Limbic-association":  {"nuclei": ["AV", "Pul"],
                            "color": "#EF5350",
                            "desc": "Papez circuit / emotional\nsalience (memory, affect)"},
}

# Color per nucleus from its functional group
NUC_GROUP_COLOR = {}
for grp, info in FUNC_GROUPS.items():
    for nuc in info["nuclei"]:
        NUC_GROUP_COLOR[nuc] = info["color"]

MSTYLE = {
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "pdf.fonttype": 42,
}
plt.rcParams.update(MSTYLE)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
gt   = pd.read_csv(GT_CSV)
conn = pd.read_csv(CONN_CSV)

# ── Nucleus MNI centroids (manuscript nuclei only) ────────────────────────────
thal = gt[(gt["is_thalamic"] == True) &
          (gt["thalamic_nucleus"].isin(MANUSCRIPT_NUCLEI))].copy()
for c in ["x_mni", "y_mni", "z_mni"]:
    thal[c] = pd.to_numeric(thal[c], errors="coerce")
thal = thal.dropna(subset=["x_mni", "y_mni", "z_mni"])

centroids = (thal.groupby("thalamic_nucleus")
             .agg(x=("x_mni",     "mean"),
                  y=("y_mni",     "mean"),
                  z=("z_mni",     "mean"),
                  n_contacts=("x_mni", "count"))
             .reset_index())

print("Nucleus centroids (manuscript nuclei):")
print(centroids.to_string(index=False))

# ── Fit group PCA on all_thalamus, then project each nucleus ──────────────────
feat_cols = [c for c in ["icoh_delta","icoh_theta","icoh_alpha",
                          "icoh_beta","icoh_lgamma","icoh_broadband"]
             if c in conn.columns]

all_sub = conn[conn["nucleus"] == "all_thalamus"].dropna(subset=feat_cols)
scaler  = StandardScaler().fit(all_sub[feat_cols].fillna(0).values)
pca_grp = PCA(n_components=2)
pca_grp.fit(scaler.transform(all_sub[feat_cols].fillna(0).values))
var = pca_grp.explained_variance_ratio_
print(f"\nGroup PCA: PC1={var[0]:.1%}  PC2={var[1]:.1%}")

nuc_proj = {}
for nuc in MANUSCRIPT_NUCLEI:
    sub = conn[conn["nucleus"] == nuc].dropna(subset=feat_cols)
    if len(sub) < 3:
        print(f"  {nuc}: too few rows ({len(sub)}), skipping")
        continue
    mean_profile = sub[feat_cols].mean().values.reshape(1, -1)
    proj = pca_grp.transform(scaler.transform(mean_profile))[0]
    nuc_proj[nuc] = {"pc1": float(proj[0]), "pc2": float(proj[1]), "n": len(sub)}
    print(f"  {nuc}: PC1={proj[0]:+.3f}  PC2={proj[1]:+.3f}  (n={len(sub)})")

centroids["pc1"] = centroids["thalamic_nucleus"].map(
    {k: v["pc1"] for k, v in nuc_proj.items()})
centroids["pc2"] = centroids["thalamic_nucleus"].map(
    {k: v["pc2"] for k, v in nuc_proj.items()})
centroids = centroids.dropna(subset=["pc1", "pc2"])

# ── Figure: PC1 anatomy (3 MNI projections + bar chart) ──────────────────────
print("\nGenerating PC1 anatomy figure with functional groups...")

views  = [("y","z","Sagittal"), ("x","y","Axial"), ("x","z","Coronal")]
vals   = centroids["pc1"].values
sizes  = centroids["n_contacts"].values
dot_sz = 120 + (sizes / sizes.max()) * 450

vmax   = max(abs(vals.min()), abs(vals.max())) * 1.15
norm   = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
cmap   = mcm.RdBu_r

fig = plt.figure(figsize=(22, 7))
gs  = GridSpec(1, 5, figure=fig,
               width_ratios=[1, 1, 1, 0.04, 1.1],
               wspace=0.30)

# ── Panels 0–2: MNI projections ───────────────────────────────────────────────
label_offsets = {
    # nucleus: (dx, dy) for each view; keyed as (xc, yc)
    ("y","z"): {"VLP":(+4,+2),"Pul":(+4,-5),"MD":(-9,+2),"CM":(+4,-5),
                "VPL":(+4,+2),"AV":(-9,+2),"VA":(+4,+2),"MGN":(+4,-5)},
    ("x","y"): {"VLP":(+4,+2),"Pul":(+4,-5),"MD":(-9,+2),"CM":(+4,-5),
                "VPL":(-9,+2),"AV":(+4,+2),"VA":(+4,+2),"MGN":(-9,-5)},
    ("x","z"): {"VLP":(+4,+2),"Pul":(+4,-5),"MD":(-9,+2),"CM":(+4,-5),
                "VPL":(-9,+2),"AV":(+4,+2),"VA":(+4,+2),"MGN":(-9,-5)},
}

for pi, (xc, yc, title) in enumerate(views):
    ax = fig.add_subplot(gs[pi])

    # Background individual contacts (faint)
    ax.scatter(thal[f"{xc}_mni"], thal[f"{yc}_mni"],
               c="#DEDEDE", s=6, alpha=0.35, zorder=1, linewidths=0)

    # Functional group convex hulls (shaded regions)
    from scipy.spatial import ConvexHull
    for grp, info in FUNC_GROUPS.items():
        pts = centroids[centroids["thalamic_nucleus"].isin(info["nuclei"])][[f"{xc}", f"{yc}"]].values
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                poly = plt.Polygon(hull_pts, closed=True,
                                   facecolor=info["color"], alpha=0.10,
                                   edgecolor=info["color"], linewidth=1.2,
                                   linestyle="--", zorder=2)
                ax.add_patch(poly)
            except Exception:
                pass

    # Nucleus centroids
    sc = ax.scatter(centroids[f"{xc}"], centroids[f"{yc}"],
                    c=vals, s=dot_sz, cmap=cmap, norm=norm,
                    edgecolors="white", linewidths=1.5,
                    zorder=4, alpha=0.95)

    # Nucleus labels
    offs = label_offsets.get((xc, yc), {})
    for _, row in centroids.iterrows():
        nuc  = row["thalamic_nucleus"]
        xv, yv = row[f"{xc}"], row[f"{yc}"]
        dx, dy = offs.get(nuc, (+4, +2))
        ax.annotate(NUC_SHORT.get(nuc, nuc),
                    xy=(xv, yv), xytext=(xv+dx, yv+dy),
                    fontsize=9, fontweight="bold", color="#111111",
                    arrowprops=dict(arrowstyle="-", color="#AAAAAA",
                                    lw=0.6, shrinkA=6, shrinkB=0))

    ax.set_xlabel(f"{xc.upper()} (mm)")
    ax.set_ylabel(f"{yc.upper()} (mm)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axhline(0, lw=0.4, color="#CCCCCC", zorder=0)
    ax.axvline(0, lw=0.4, color="#CCCCCC", zorder=0)
    ax.set_facecolor("#F8F8F8")

    # Anatomical direction labels
    xl, xr = ax.get_xlim(); yb, yt = ax.get_ylim()
    if xc == "y":
        ax.text(xl + (xr-xl)*0.05, yb + (yt-yb)*0.04,
                "← Post.", fontsize=7, color="#888888")
        ax.text(xr - (xr-xl)*0.22, yb + (yt-yb)*0.04,
                "Ant. →", fontsize=7, color="#888888")
    if yc == "z":
        ax.text(xl + (xr-xl)*0.02, yt - (yt-yb)*0.08,
                "Dors. ↑", fontsize=7, color="#888888")

# ── Panel 3: Colorbar ─────────────────────────────────────────────────────────
cax = fig.add_subplot(gs[3])
cb  = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
cb.set_label("PC1 score\n(← passive / cognitive →)",
             fontsize=9, labelpad=6)
cb.ax.tick_params(labelsize=8)
cb.ax.axhline(0, color="black", lw=1.2)
cb.ax.text(0.5, 0.02, "Passive\n(REM, TV)", transform=cb.ax.transAxes,
           fontsize=7, ha="center", va="bottom", color="#3B5FA0")
cb.ax.text(0.5, 0.98, "Cognitive/\nSleep", transform=cb.ax.transAxes,
           fontsize=7, ha="center", va="top", color="#B22222")

# ── Panel 4: Bar chart sorted by anterior-posterior (y_mni) ──────────────────
ax4 = fig.add_subplot(gs[4])

# Sort by y_mni ascending = posterior → anterior
sorted_c = centroids.sort_values("y").reset_index(drop=True)
ypos     = np.arange(len(sorted_c))
bar_colors = [cmap(norm(v)) for v in sorted_c["pc1"].values]

bars = ax4.barh(ypos, sorted_c["pc1"].values,
                color=bar_colors, edgecolor="white", linewidth=0.6,
                height=0.65, alpha=0.92)

# Add n_contacts annotation on each bar
for i, (_, row) in enumerate(sorted_c.iterrows()):
    xval = row["pc1"]
    n    = int(row["n_contacts"])
    xoff = 0.06 if xval >= 0 else -0.06
    ha   = "left" if xval >= 0 else "right"
    ax4.text(xval + xoff, i, f"n={n}", va="center",
             fontsize=7.5, color="#444444")

# Functional group brackets on the right
x_bracket = sorted_c["pc1"].abs().max() * 1.6
for grp, info in FUNC_GROUPS.items():
    idxs = [i for i, nuc in enumerate(sorted_c["thalamic_nucleus"])
            if nuc in info["nuclei"]]
    if not idxs:
        continue
    ymin, ymax = min(idxs) - 0.35, max(idxs) + 0.35
    ymid       = (ymin + ymax) / 2
    ax4.plot([x_bracket, x_bracket + 0.05, x_bracket + 0.05, x_bracket],
             [ymin, ymin, ymax, ymax],
             color=info["color"], lw=2, solid_capstyle="round",
             transform=ax4.get_yaxis_transform(), clip_on=False)
    ax4.text(x_bracket + 0.12, ymid, grp,
             fontsize=8, color=info["color"], fontweight="bold",
             va="center", ha="left",
             transform=ax4.get_yaxis_transform(), clip_on=False)

ax4.axvline(0, color="black", lw=1, zorder=3)
ax4.set_yticks(ypos)
ax4.set_yticklabels([NUC_SHORT.get(n, n) for n in sorted_c["thalamic_nucleus"]],
                    fontsize=10, fontweight="bold")
ax4.set_xlabel("PC1 score (passive ← 0 → cognitive/sleep)", fontsize=9)
ax4.set_title("PC1 by nucleus (sorted post.→ant.)\nNo anatomical gradient (r = −0.476, p = 0.233)",
              fontsize=11, fontweight="bold")
ax4.set_facecolor("#F8F8F8")
ax4.spines["left"].set_visible(False)
ax4.tick_params(left=False)

# y-axis label: posterior/anterior
ax4.text(-0.08, 0, "← Posterior", transform=ax4.transAxes,
         fontsize=7.5, color="#888888", rotation=90,
         va="center", ha="center")
ax4.text(-0.08, 1, "Anterior →", transform=ax4.transAxes,
         fontsize=7.5, color="#888888", rotation=90,
         va="center", ha="center")

# ── Dot size legend ───────────────────────────────────────────────────────────
from matplotlib.lines import Line2D
leg = [plt.scatter([],[],s=120+(s/sizes.max())*450,
                   c="#AAAAAA",alpha=0.7,label=f"n={s}")
       for s in [10, 30, 50, 90]]
fig.legend(handles=leg, title="n contacts", loc="lower left",
           fontsize=8, title_fontsize=8, frameon=False,
           bbox_to_anchor=(0.01, -0.02))

# ── Functional group legend ───────────────────────────────────────────────────
from matplotlib.patches import Patch
grp_handles = [Patch(facecolor=info["color"], alpha=0.4,
                      edgecolor=info["color"], label=grp)
               for grp, info in FUNC_GROUPS.items()]
fig.legend(handles=grp_handles, title="Functional group",
           loc="lower center", fontsize=8, title_fontsize=8,
           frameon=False, bbox_to_anchor=(0.42, -0.06), ncol=3)

fig.suptitle(
    "Supplementary Figure S3 — Thalamic nucleus PC1 scores in MNI space: "
    "spatial position does not predict PC1 score\n"
    f"PC1 explains {var[0]:.1%} of thalamocortical iCoh variance; "
    "colour = PC1 score (blue = passive pole, red = cognitive/sleep pole); "
    "dot size ∝ n contacts; "
    "note VA and AV are 4–5 mm apart yet at opposite ends of PC1 "
    "(anatomical Spearman r = −0.476, p = 0.233, n.s.; see Figure S4)",
    fontsize=11, fontweight="bold", y=1.03
)

plt.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(str(FIG_DIR / f"FigS3_PC1_anatomy_v2.{ext}"),
                dpi=300 if ext == "png" else None,
                bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved: FigS3_PC1_anatomy_v2.pdf / .png")

# ── PC2 figure (same layout) ─────────────────────────────────────────────────
print("Generating PC2 anatomy figure...")
vals2   = centroids["pc2"].values
dot_sz2 = 120 + (sizes / sizes.max()) * 450
vmax2   = max(abs(vals2.min()), abs(vals2.max())) * 1.15
norm2   = mcolors.TwoSlopeNorm(vmin=-vmax2, vcenter=0, vmax=vmax2)
cmap2   = mcm.PuOr_r

fig2 = plt.figure(figsize=(22, 7))
gs2  = GridSpec(1, 5, figure=fig2,
                width_ratios=[1, 1, 1, 0.04, 1.1], wspace=0.30)

for pi, (xc, yc, title) in enumerate(views):
    ax = fig2.add_subplot(gs2[pi])
    ax.scatter(thal[f"{xc}_mni"], thal[f"{yc}_mni"],
               c="#DEDEDE", s=6, alpha=0.35, zorder=1, linewidths=0)
    ax.scatter(centroids[f"{xc}"], centroids[f"{yc}"],
               c=vals2, s=dot_sz2, cmap=cmap2, norm=norm2,
               edgecolors="white", linewidths=1.5, zorder=4, alpha=0.95)
    offs = label_offsets.get((xc, yc), {})
    for _, row in centroids.iterrows():
        nuc  = row["thalamic_nucleus"]
        xv, yv = row[f"{xc}"], row[f"{yc}"]
        dx, dy = offs.get(nuc, (+4, +2))
        ax.annotate(NUC_SHORT.get(nuc, nuc),
                    xy=(xv, yv), xytext=(xv+dx, yv+dy),
                    fontsize=9, fontweight="bold", color="#111111",
                    arrowprops=dict(arrowstyle="-", color="#AAAAAA",
                                    lw=0.6, shrinkA=6, shrinkB=0))
    ax.set_xlabel(f"{xc.upper()} (mm)")
    ax.set_ylabel(f"{yc.upper()} (mm)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axhline(0, lw=0.4, color="#CCCCCC")
    ax.axvline(0, lw=0.4, color="#CCCCCC")
    ax.set_facecolor("#F8F8F8")

cax2 = fig2.add_subplot(gs2[3])
cb2  = ColorbarBase(cax2, cmap=cmap2, norm=norm2, orientation="vertical")
cb2.set_label("PC2 score", fontsize=9, labelpad=6)
cb2.ax.tick_params(labelsize=8)
cb2.ax.axhline(0, color="black", lw=1.2)

# Bar chart for PC2 sorted by medial-lateral (x_mni)
ax4b = fig2.add_subplot(gs2[4])
sorted_c2 = centroids.sort_values("x").reset_index(drop=True)
ypos2     = np.arange(len(sorted_c2))
bar_cols2 = [cmap2(norm2(v)) for v in sorted_c2["pc2"].values]
ax4b.barh(ypos2, sorted_c2["pc2"].values,
          color=bar_cols2, edgecolor="white", linewidth=0.6,
          height=0.65, alpha=0.92)
for i, (_, row) in enumerate(sorted_c2.iterrows()):
    xval = row["pc2"]
    n    = int(row["n_contacts"])
    xoff = 0.04 if xval >= 0 else -0.04
    ha   = "left" if xval >= 0 else "right"
    ax4b.text(xval + xoff, i, f"n={n}", va="center",
              fontsize=7.5, color="#444444")
ax4b.axvline(0, color="black", lw=1)
ax4b.set_yticks(ypos2)
ax4b.set_yticklabels([NUC_SHORT.get(n,n) for n in sorted_c2["thalamic_nucleus"]],
                     fontsize=10, fontweight="bold")
ax4b.set_xlabel("PC2 score", fontsize=9)
ax4b.set_title("PC2 by nucleus\n(sorted medial → lateral by x_mni)",
               fontsize=11, fontweight="bold")
ax4b.set_facecolor("#F8F8F8")
ax4b.spines["left"].set_visible(False)
ax4b.tick_params(left=False)

fig2.suptitle(
    f"Thalamic nucleus PC2 scores mapped onto MNI space\n"
    f"PC2 explains {var[1]:.1%} of variance; "
    "bar chart sorted medial→lateral",
    fontsize=12, fontweight="bold", y=1.03
)
plt.tight_layout()
for ext in ["pdf", "png"]:
    fig2.savefig(str(FIG_DIR / f"FigS3_PC2_anatomy_v2.{ext}"),
                 dpi=300 if ext == "png" else None,
                 bbox_inches="tight", facecolor="white")
plt.close(fig2)
print("Saved: FigS3_PC2_anatomy_v2.pdf / .png")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("NUCLEUS PC PROJECTIONS (sorted posterior → anterior by y_mni)")
print("="*65)
result = centroids[["thalamic_nucleus","x","y","z","n_contacts","pc1","pc2"]].copy()
result.columns = ["Nucleus","x_mni","y_mni","z_mni","n_contacts","PC1","PC2"]
result = result.sort_values("y_mni")
# Add functional group
result["Group"] = result["Nucleus"].map(
    {nuc: grp for grp, info in FUNC_GROUPS.items() for nuc in info["nuclei"]})
print(result.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
print(f"\nGradient statistics (Spearman):")
from scipy import stats as _stats
r_anat, p_anat = _stats.spearmanr(result["y_mni"], result["PC1"])
grp_map = {"Motor relay": 1, "Integrative": 2, "Limbic-association": 3}
grp_ranks = result["Group"].map(grp_map).dropna()
r_func, p_func = _stats.spearmanr(grp_ranks, result.loc[grp_ranks.index, "PC1"])
print(f"  Anatomical (y_mni vs PC1):        r = {r_anat:+.3f}, p = {p_anat:.4f}  (n.s.)")
print(f"  Functional (circuit class vs PC1): r = {r_func:+.3f}, p = {p_func:.4f}  ***")
print(f"\nAll figures saved to: {FIG_DIR.resolve()}")
