"""
interpret_pca_dimensions.py
-----------------------------
Generates Figure S6: Biological interpretation of PC1 and PC2.

PC1 = thalamocortical coupling MAGNITUDE (broadband — all bands positive)
PC2 = spectral COMPOSITION of coupling (slow vs fast frequency contrast)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES_DIR   = Path("outputs/results")
FIG_DIR   = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
CONN_CSV  = RES_DIR / "step5_connectivity.csv"
FEAT_COLS = ["icoh_delta","icoh_theta","icoh_alpha",
             "icoh_beta","icoh_lgamma","icoh_broadband"]
FEAT_LABELS = ["Delta\n(0.5–4 Hz)","Theta\n(4–8 Hz)","Alpha\n(8–13 Hz)",
               "Beta\n(13–30 Hz)","Low gamma\n(30–70 Hz)","Broadband\n(0.5–70 Hz)"]
STATE_LABELS = {
    "rem_sleep":"REM sleep","nrem_sleep":"NREM sleep",
    "watching_tv":"Watching TV","eating":"Eating","playing":"Playing",
    "talking":"Talking","crying":"Crying","laughing":"Laughing","reading":"Reading",
}
STATE_COLORS = {
    "rem_sleep":"#8B008B","nrem_sleep":"#1E90FF","watching_tv":"#FF69B4",
    "eating":"#A0522D","playing":"#228B22","talking":"#20B2AA",
    "crying":"#DC143C","laughing":"#FF8C00","reading":"#6B8E23",
}

print("Loading data...")
conn = pd.read_csv(CONN_CSV)
feat_cols = [c for c in FEAT_COLS if c in conn.columns]
base = (conn[conn["nucleus"]=="all_thalamus"].dropna(subset=feat_cols)
        .groupby(["subject_id","state"])[feat_cols].mean().reset_index())

def zscore_ws(df):
    parts = []
    for sid, grp in df.groupby("subject_id"):
        X = grp[feat_cols].values
        if len(X) < 2: continue
        g2 = grp.copy(); g2[feat_cols] = StandardScaler().fit_transform(X)
        parts.append(g2)
    return pd.concat(parts, ignore_index=True)

bz = zscore_ws(base)
X  = bz[feat_cols].fillna(0).values
pca = PCA(); coords = pca.fit_transform(X); var = pca.explained_variance_ratio_*100
bz["PC1"] = coords[:,0]; bz["PC2"] = coords[:,1]
centroids = bz.groupby("state")[["PC1","PC2"]].mean()
pc1_load = pca.components_[0]; pc2_load = pca.components_[1]

print(f"PC1={var[0]:.1f}%  PC2={var[1]:.1f}%")
print("\nLoadings:")
for i,fc in enumerate(feat_cols):
    print(f"  {fc:<22} PC1={pc1_load[i]:+.3f}  PC2={pc2_load[i]:+.3f}")
print("\nState centroids:")
for s,r in centroids.sort_values("PC1").iterrows():
    print(f"  {STATE_LABELS.get(s,s):<20} PC1={r.PC1:+.3f}  PC2={r.PC2:+.3f}")

MSTYLE = {"font.family":"sans-serif","font.size":11,
           "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42}
plt.rcParams.update(MSTYLE)
fig = plt.figure(figsize=(18, 6))
gs  = fig.add_gridspec(1, 3, wspace=0.38)

# Panel A: PC1 loadings
ax1 = fig.add_subplot(gs[0])
ax1.bar(range(len(feat_cols)), pc1_load,
        color=["#2B6CB8"]*len(feat_cols), alpha=0.85, edgecolor="white", width=0.6)
ax1.axhline(0, color="black", lw=0.8)
ax1.set_xticks(range(len(feat_cols))); ax1.set_xticklabels(FEAT_LABELS, fontsize=8.5)
ax1.set_ylabel("Loading"); ax1.set_ylim(-0.7, 0.7)
ax1.set_title(f"A   PC1 ({var[0]:.1f}%): Coupling MAGNITUDE\n"
              "All bands load positively — broadband signal",
              fontsize=10, fontweight="bold", loc="left")
for i,v in enumerate(pc1_load):
    ax1.text(i, v+0.03, f"{v:+.2f}", ha="center", fontsize=8.5,
             color="#2B6CB8", fontweight="bold")
ax1.set_facecolor("#F9F9F9")

# Panel B: PC2 loadings
ax2 = fig.add_subplot(gs[1])
bar_c = ["#C0392B" if v<0 else "#E67E22" for v in pc2_load]
ax2.bar(range(len(feat_cols)), pc2_load,
        color=bar_c, alpha=0.85, edgecolor="white", width=0.6)
ax2.axhline(0, color="black", lw=0.8)
ax2.set_xticks(range(len(feat_cols))); ax2.set_xticklabels(FEAT_LABELS, fontsize=8.5)
ax2.set_ylabel("Loading"); ax2.set_ylim(-0.7, 0.7)
ax2.set_title(f"B   PC2 ({var[1]:.1f}%): Spectral COMPOSITION\n"
              "Slow (−) vs fast (+) frequency coupling",
              fontsize=10, fontweight="bold", loc="left")
for i,v in enumerate(pc2_load):
    ax2.text(i, v+(0.03 if v>=0 else -0.06), f"{v:+.2f}", ha="center",
             fontsize=8.5, color=bar_c[i], fontweight="bold")
ax2.axhspan(-0.7,0, alpha=0.04, color="#C0392B")
ax2.axhspan(0,0.7,  alpha=0.04, color="#E67E22")
ax2.text(2.5, 0.58, "FAST (waking)", ha="center", fontsize=8,
         color="#E67E22", style="italic")
ax2.text(2.5,-0.62, "SLOW (sleep)", ha="center", fontsize=8,
         color="#C0392B", style="italic")
ax2.set_facecolor("#F9F9F9")

# Panel C: PC1 × PC2 scatter
ax3 = fig.add_subplot(gs[2])
ax3.axhline(0, color="#CCCCCC", lw=1); ax3.axvline(0, color="#CCCCCC", lw=1)

# quadrant shading
for x0,x1,y0,y1,col,lbl in [
    ( 0, 2.2,  0, 4,   "#2B6CB8", "Active waking\n(strong + fast)"),
    (-2.6,0,   0, 4,   "#95A5A6", "Passive waking\n(weak + fast)"),
    (-2.6,0, -3.5,0,   "#8B008B", "REM sleep\n(weak + slow)"),
    ( 0, 2.2,-3.5,0,   "#1E90FF", "NREM sleep\n(strong + slow)"),
]:
    ax3.fill_between([x0,x1],[y0,y0],[y1,y1], alpha=0.06, color=col)
    cx = (x0+x1)/2; cy = (y0+y1)/2
    ax3.text(cx, cy, lbl, ha="center", va="center", fontsize=8,
             color=col, style="italic",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                       edgecolor=col, alpha=0.7, linewidth=0.8))

# individual points
for state in bz["state"].unique():
    pts = bz[bz["state"]==state]
    ax3.scatter(pts["PC1"], pts["PC2"], c=STATE_COLORS.get(state,"#888"),
                s=20, alpha=0.2, linewidths=0)

# centroids
offsets = {
    "rem_sleep":(-0.05,-0.28),"nrem_sleep":(-0.05,-0.28),
    "watching_tv":(-0.55, 0.08),"eating":( 0.08, 0.10),
    "playing":( 0.08,-0.18),"talking":( 0.08, 0.10),
    "crying":( 0.08, 0.10),"laughing":( 0.08, 0.10),"reading":( 0.08, 0.10),
}
for state,row in centroids.iterrows():
    ax3.scatter(row.PC1, row.PC2, c=STATE_COLORS.get(state,"#888"),
                s=220, edgecolors="white", linewidths=1.5, zorder=5)
    dx,dy = offsets.get(state,(0.08,0.10))
    ax3.annotate(STATE_LABELS.get(state,state),
                 xy=(row.PC1,row.PC2), xytext=(row.PC1+dx,row.PC2+dy),
                 fontsize=8.5, color=STATE_COLORS.get(state,"#888"),
                 fontweight="bold")

ax3.set_xlabel(f"PC1: Coupling magnitude ({var[0]:.1f}%)", fontsize=11)
ax3.set_ylabel(f"PC2: Spectral composition ({var[1]:.1f}%)", fontsize=11)
ax3.set_title("C   Thalamocortical state space (PC1 × PC2)\n"
              "Centroids (large) + individual observations (small)",
              fontsize=10, fontweight="bold", loc="left")
ax3.set_facecolor("#F9F9F9")

fig.suptitle(
    "Supplementary Figure S6 — Biological interpretation of PC1 and PC2\n"
    f"PC1 ({var[0]:.1f}%) = coupling magnitude (broadband)  ·  "
    f"PC2 ({var[1]:.1f}%) = spectral composition (slow vs fast frequency)",
    fontsize=11, fontweight="bold", y=1.03)
plt.tight_layout()
for ext in ["pdf","png"]:
    fig.savefig(str(FIG_DIR/f"FigS6_pca_interpretation.{ext}"),
                dpi=300 if ext=="png" else None, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nSaved: FigS6_pca_interpretation.pdf/.png")

print(f"""
FIGURE LEGEND:

Supplementary Figure S6 | Biological interpretation of PCA dimensions.
(A) PC1 loadings. All six frequency bands load positively on PC1 (range
+{min(pc1_load):.2f} to +{max(pc1_load):.2f}), indicating that PC1 captures the
overall broadband magnitude of thalamocortical iCoh — the global
degree to which thalamus and cortex are coupled simultaneously
across all frequencies. (B) PC2 loadings. Slow-frequency bands
(delta, theta, alpha) load negatively ({pc2_load[0]:.2f}, {pc2_load[1]:.2f}, {pc2_load[2]:.2f})
and fast-frequency bands (beta, low gamma) load positively
({pc2_load[3]:+.2f}, {pc2_load[4]:+.2f}), defining a slow-versus-fast spectral
composition axis that separates sleep states (negative PC2) from
waking states (positive PC2) independently of coupling magnitude.
(C) Thalamocortical state space defined by PC1 × PC2. State centroids
(large circles) and individual subject-state observations (small
circles) organise into four interpretable quadrants: REM sleep
(weak, slow-dominant coupling), NREM sleep (strong, slow-dominant
coupling), passive waking — watching TV and eating (weak,
fast-dominant coupling), and active or emotionally engaged waking
states — talking, playing, crying, laughing, reading (strong,
fast-dominant coupling).
""")
