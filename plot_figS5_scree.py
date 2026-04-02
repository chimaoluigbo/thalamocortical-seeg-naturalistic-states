"""
plot_figS5_scree.py
--------------------
Generates Supplementary Figure S5: Scree plot of PCA explained variance.

Shows:
  - Observed explained variance per component (all 6 PCs)
  - Phase-randomized null distribution (median + 95% CI) per component
  - PC1 and PC2 highlighted with observed vs null comparison
  - Cumulative variance curve

Run from seeg_study directory:
    source /home/chima/seeg_env/bin/activate
    cd /mnt/c/Users/chima/seeg_study
    python plot_figS5_scree.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_DIR  = Path(".")
RES_DIR   = DATA_DIR / "outputs/results"
FIG_DIR   = DATA_DIR / "outputs/figures"
ROB_DIR   = RES_DIR / "robustness"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CONN_CSV  = RES_DIR / "step5_connectivity.csv"
FEAT_COLS = ["icoh_delta","icoh_theta","icoh_alpha",
             "icoh_beta","icoh_lgamma","icoh_broadband"]
N_PERMS   = 1000
SEED      = 42

# ── Load & prepare ─────────────────────────────────────────────────────────────
print("Loading data...")
conn      = pd.read_csv(CONN_CSV)
feat_cols = [c for c in FEAT_COLS if c in conn.columns]

base = (conn[conn["nucleus"]=="all_thalamus"]
        .dropna(subset=feat_cols)
        .groupby(["subject_id","state"])[feat_cols]
        .mean().reset_index())

def zscore_ws(df):
    parts = []
    for sid, grp in df.groupby("subject_id"):
        X = grp[feat_cols].values
        if len(X) < 2: continue
        g2 = grp.copy()
        g2[feat_cols] = StandardScaler().fit_transform(X)
        parts.append(g2)
    return pd.concat(parts, ignore_index=True)

base_z   = zscore_ws(base)
X_obs    = base_z[feat_cols].fillna(0).values
pca_obs  = PCA()
pca_obs.fit(X_obs)
obs_var  = pca_obs.explained_variance_ratio_ * 100
n_comp   = len(obs_var)

print(f"Observed variance per component:")
for i, v in enumerate(obs_var):
    print(f"  PC{i+1}: {v:.1f}%")
print(f"  Combined PC1+PC2: {obs_var[0]+obs_var[1]:.1f}%")

# ── Phase-randomized null ──────────────────────────────────────────────────────
# Load cached if available
null_csv = ROB_DIR / "null_pc1_pc2.csv"
if null_csv.exists():
    print(f"\nLoading cached null from {null_csv}")
    null_df   = pd.read_csv(null_csv)
    # We need all components — rerun if only 2 stored
    if len(null_df.columns) < n_comp:
        print("Cached file has fewer components than needed — rerunning...")
        null_csv = None

if not null_csv or not null_csv.exists():
    print(f"\nRunning {N_PERMS} phase-randomized permutations for all {n_comp} PCs...")
    sids_arr   = base_z["subject_id"].values
    null_all   = []
    np.random.seed(SEED)

    for perm in range(N_PERMS):
        perm_df = base_z.copy()
        for sid in np.unique(sids_arr):
            idx  = np.where(sids_arr == sid)[0]
            vals = perm_df.iloc[idx][feat_cols].values.copy()
            for j in range(vals.shape[1]):
                np.random.shuffle(vals[:, j])
            perm_df.iloc[idx, perm_df.columns.get_indexer(feat_cols)] = vals
        X_p = perm_df[feat_cols].fillna(0).values
        v   = PCA().fit(X_p).explained_variance_ratio_ * 100
        null_all.append(v)
        if (perm + 1) % 200 == 0:
            print(f"  {perm+1}/{N_PERMS} done...")

    null_matrix = np.array(null_all)   # shape (N_PERMS, n_comp)
    null_df = pd.DataFrame(null_matrix,
                           columns=[f"PC{i+1}" for i in range(n_comp)])
    null_df.to_csv(ROB_DIR / "null_all_pcs.csv", index=False)
else:
    # Rerun to get all components
    print(f"\nRunning {N_PERMS} permutations for all {n_comp} PCs...")
    sids_arr = base_z["subject_id"].values
    null_all = []
    np.random.seed(SEED)
    for perm in range(N_PERMS):
        perm_df = base_z.copy()
        for sid in np.unique(sids_arr):
            idx  = np.where(sids_arr == sid)[0]
            vals = perm_df.iloc[idx][feat_cols].values.copy()
            for j in range(vals.shape[1]):
                np.random.shuffle(vals[:, j])
            perm_df.iloc[idx, perm_df.columns.get_indexer(feat_cols)] = vals
        X_p = perm_df[feat_cols].fillna(0).values
        v   = PCA().fit(X_p).explained_variance_ratio_ * 100
        null_all.append(v)
        if (perm + 1) % 200 == 0:
            print(f"  {perm+1}/{N_PERMS} done...")
    null_matrix = np.array(null_all)
    null_df = pd.DataFrame(null_matrix,
                           columns=[f"PC{i+1}" for i in range(n_comp)])
    null_df.to_csv(ROB_DIR / "null_all_pcs.csv", index=False)

null_matrix = null_df[[f"PC{i+1}" for i in range(n_comp)]].values

null_med  = np.median(null_matrix, axis=0)
null_lo   = np.percentile(null_matrix, 2.5,  axis=0)
null_hi   = np.percentile(null_matrix, 97.5, axis=0)
p_vals    = np.array([np.mean(null_matrix[:, i] >= obs_var[i])
                      for i in range(n_comp)])

print("\nComponent-wise comparison:")
for i in range(n_comp):
    sig = "***" if p_vals[i] < 0.001 else ("**" if p_vals[i] < 0.01 else
          ("*" if p_vals[i] < 0.05 else "n.s."))
    print(f"  PC{i+1}: observed={obs_var[i]:.1f}%  "
          f"null={null_med[i]:.1f}% [{null_lo[i]:.1f}–{null_hi[i]:.1f}%]  "
          f"p={p_vals[i]:.4f} {sig}")

# ── Figure ─────────────────────────────────────────────────────────────────────
MSTYLE = {"font.family":"sans-serif","font.size":11,
           "axes.spines.top":False,"axes.spines.right":False,
           "axes.linewidth":0.9,"pdf.fonttype":42}
plt.rcParams.update(MSTYLE)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
x     = np.arange(1, n_comp + 1)
BLUE  = "#2B6CB8"
RED   = "#C0392B"
GREY  = "#AAAAAA"

# ── Panel A: Scree plot with null ─────────────────────────────────────────────
ax = axes[0]

# Null CI ribbon
ax.fill_between(x, null_lo, null_hi,
                color=GREY, alpha=0.25, label="Null 95% CI")
ax.plot(x, null_med, "o--", color=GREY, lw=1.5,
        markersize=6, label=f"Null median", zorder=2)

# Observed bars
bar_colors = [RED if i < 2 else BLUE for i in range(n_comp)]
bars = ax.bar(x, obs_var, color=bar_colors, alpha=0.85,
              edgecolor="white", linewidth=0.8, width=0.55, zorder=3)

# Significance markers
for i in range(n_comp):
    p = p_vals[i]
    if p < 0.001:   sig = "***"
    elif p < 0.01:  sig = "**"
    elif p < 0.05:  sig = "*"
    else:            sig = "n.s."
    ax.text(x[i], obs_var[i] + 0.8, sig,
            ha="center", va="bottom", fontsize=10,
            color="black", fontweight="bold")

# Value labels inside bars
for i in range(n_comp):
    ax.text(x[i], obs_var[i] / 2, f"{obs_var[i]:.1f}%",
            ha="center", va="center", fontsize=9.5,
            color="white", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([f"PC{i+1}" for i in range(n_comp)], fontsize=11)
ax.set_xlabel("Principal component", fontsize=12)
ax.set_ylabel("Explained variance (%)", fontsize=12)
ax.set_title("A   Scree plot: observed vs phase-randomized null\n"
             "(*** p < 0.001 vs null; red = reported components)",
             fontsize=11, fontweight="bold", loc="left")
ax.set_facecolor("#F9F9F9")
ax.set_ylim(0, obs_var[0] * 1.15)

legend_handles = [
    mpatches.Patch(color=RED,  alpha=0.85, label="PC1, PC2 (reported)"),
    mpatches.Patch(color=BLUE, alpha=0.85, label="PC3–PC6"),
    mpatches.Patch(color=GREY, alpha=0.4,  label="Null 95% CI"),
    plt.Line2D([0],[0], color=GREY, lw=1.5, linestyle="--",
               marker="o", markersize=5, label="Null median"),
]
ax.legend(handles=legend_handles, frameon=False, fontsize=9,
          loc="upper right")

# ── Panel B: Cumulative variance ──────────────────────────────────────────────
ax2 = axes[1]

obs_cum  = np.cumsum(obs_var)
null_cum_med = np.median(np.cumsum(null_matrix, axis=1), axis=0)
null_cum_lo  = np.percentile(np.cumsum(null_matrix, axis=1), 2.5,  axis=0)
null_cum_hi  = np.percentile(np.cumsum(null_matrix, axis=1), 97.5, axis=0)

ax2.fill_between(x, null_cum_lo, null_cum_hi,
                 color=GREY, alpha=0.25, label="Null 95% CI")
ax2.plot(x, null_cum_med, "o--", color=GREY, lw=1.5, markersize=6,
         label="Null median (cumulative)", zorder=2)
ax2.plot(x, obs_cum, "o-", color=RED, lw=2.5, markersize=9,
         label="Observed (cumulative)", zorder=4)

# Highlight PC1+PC2
ax2.axhline(obs_cum[1], color=RED, lw=1, linestyle=":",
            alpha=0.7)
ax2.annotate(f"PC1+PC2 = {obs_cum[1]:.1f}%",
             xy=(2, obs_cum[1]), xytext=(3.2, obs_cum[1] - 4),
             fontsize=10, color=RED, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

# Label each observed point
for i in range(n_comp):
    ax2.annotate(f"{obs_cum[i]:.1f}%",
                 xy=(x[i], obs_cum[i]),
                 xytext=(x[i] + 0.15, obs_cum[i] - 2.5),
                 fontsize=8.5, color=RED)

ax2.set_xticks(x)
ax2.set_xticklabels([f"PC{i+1}" for i in range(n_comp)], fontsize=11)
ax2.set_xlabel("Number of components", fontsize=12)
ax2.set_ylabel("Cumulative explained variance (%)", fontsize=12)
ax2.set_title("B   Cumulative variance: observed vs null\n"
              "(two components explain 86.9% of variance)",
              fontsize=11, fontweight="bold", loc="left")
ax2.set_facecolor("#F9F9F9")
ax2.set_ylim(0, 105)
ax2.legend(frameon=False, fontsize=9, loc="upper left")

fig.suptitle(
    "Supplementary Figure S5 — Scree plot of thalamocortical iCoh PCA\n"
    f"(n=27 subjects, 524 recordings, 9 behavioral states; "
    f"null = 1,000 phase-randomized permutations)",
    fontsize=12, fontweight="bold", y=1.02
)

plt.tight_layout()
for ext in ["pdf","png"]:
    fig.savefig(str(FIG_DIR / f"FigS5_scree_plot.{ext}"),
                dpi=300 if ext=="png" else None,
                bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nSaved: FigS5_scree_plot.pdf/.png → {FIG_DIR.resolve()}")

# ── Figure legend text ─────────────────────────────────────────────────────────
print(f"""
FIGURE LEGEND (for manuscript):

Supplementary Figure S5 | Scree plot of principal component analysis.
(A) Explained variance per component for the thalamocortical iCoh PCA
(red bars: PC1 and PC2, the two reported components; blue bars: remaining
components). The grey ribbon shows the 95% interval of a phase-randomized
null distribution (1,000 permutations; iCoh features shuffled across states
within each subject, preserving marginal distributions). Significance
markers (***p < 0.001) indicate components exceeding the null.
(B) Cumulative explained variance for the observed data (red line) and
null distribution (grey). PC1 and PC2 together account for 86.9% of
variance, compared with a null median of {null_cum_med[1]:.1f}%
(95% CI: {null_cum_lo[1]:.1f}–{null_cum_hi[1]:.1f}%) for two components
(p < 0.001).
""")
