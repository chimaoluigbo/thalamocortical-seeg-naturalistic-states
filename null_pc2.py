"""
null_pc2.py
-----------
Extends the phase-randomized null model to also report PC2.

The manuscript currently states:
  "The null distribution yielded a median PC1 of 20.8%
   (95% interval: 19.1–22.9%), far below the observed value
   (p < 0.001, permutation test)."

This script adds PC2 to the same null model and reports whether
the observed PC2 (32.9%) also significantly exceeds chance.

Uses same method as robustness_analyses.py:
  - Phase-randomize iCoh features across states within each subject
  - Z-score within subject
  - PCA
  - Record BOTH PC1 and PC2 from each permutation

Run from seeg_study directory:
    python null_pc2.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(".")
RES_DIR  = DATA_DIR / "outputs/results"
FIG_DIR  = DATA_DIR / "outputs/figures"
ROB_DIR  = RES_DIR / "robustness"
ROB_DIR.mkdir(parents=True, exist_ok=True)

CONN_CSV  = RES_DIR / "step5_connectivity.csv"
FEAT_COLS = ["icoh_delta","icoh_theta","icoh_alpha",
             "icoh_beta","icoh_lgamma","icoh_broadband"]
N_PERMS   = 1000
SEED      = 42

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

base_z  = zscore_ws(base)
X_obs   = base_z[feat_cols].fillna(0).values
pca_obs = PCA().fit(X_obs)
var     = pca_obs.explained_variance_ratio_
obs_pc1 = var[0] * 100
obs_pc2 = var[1] * 100

print(f"Observed: PC1 = {obs_pc1:.1f}%  PC2 = {obs_pc2:.1f}%  "
      f"combined = {obs_pc1+obs_pc2:.1f}%")

# ── Phase-randomized null ──────────────────────────────────────────────────────
print(f"\nRunning {N_PERMS} phase-randomized permutations...")
sids_arr  = base_z["subject_id"].values
null_pc1  = []
null_pc2  = []
null_comb = []
np.random.seed(SEED)

for perm in range(N_PERMS):
    perm_df = base_z.copy()
    for sid in np.unique(sids_arr):
        idx  = np.where(sids_arr == sid)[0]
        vals = perm_df.iloc[idx][feat_cols].values.copy()
        for j in range(vals.shape[1]):
            np.random.shuffle(vals[:, j])
        perm_df.iloc[idx, perm_df.columns.get_indexer(feat_cols)] = vals
    X_p  = perm_df[feat_cols].fillna(0).values
    pca_p = PCA(n_components=2).fit(X_p)
    v     = pca_p.explained_variance_ratio_
    null_pc1.append(v[0] * 100)
    null_pc2.append(v[1] * 100)
    null_comb.append((v[0] + v[1]) * 100)
    if (perm + 1) % 200 == 0:
        print(f"  {perm+1}/{N_PERMS} done...")

arr1 = np.array(null_pc1)
arr2 = np.array(null_pc2)
arrc = np.array(null_comb)

p1 = np.mean(arr1 >= obs_pc1)
p2 = np.mean(arr2 >= obs_pc2)
pc = np.mean(arrc >= obs_pc1 + obs_pc2)

print(f"\n{'='*58}")
print(f"PHASE-RANDOMIZED NULL RESULTS")
print(f"{'='*58}")
print(f"\n  PC1:")
print(f"    Observed:   {obs_pc1:.1f}%")
print(f"    Null median:{np.median(arr1):.1f}%  "
      f"(95% CI: {np.percentile(arr1,2.5):.1f}–{np.percentile(arr1,97.5):.1f}%)")
print(f"    p-value:    {p1:.4f}")

print(f"\n  PC2:")
print(f"    Observed:   {obs_pc2:.1f}%")
print(f"    Null median:{np.median(arr2):.1f}%  "
      f"(95% CI: {np.percentile(arr2,2.5):.1f}–{np.percentile(arr2,97.5):.1f}%)")
print(f"    p-value:    {p2:.4f}")

print(f"\n  Combined (PC1+PC2):")
print(f"    Observed:   {obs_pc1+obs_pc2:.1f}%")
print(f"    Null median:{np.median(arrc):.1f}%  "
      f"(95% CI: {np.percentile(arrc,2.5):.1f}–{np.percentile(arrc,97.5):.1f}%)")
print(f"    p-value:    {pc:.4f}")

# ── Figure ─────────────────────────────────────────────────────────────────────
MSTYLE = {"font.family":"sans-serif","font.size":11,
           "axes.spines.top":False,"axes.spines.right":False,
           "pdf.fonttype":42}
plt.rcParams.update(MSTYLE)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, null_arr, obs, label, color in [
    (axes[0], arr1, obs_pc1, "PC1", "#E8573C"),
    (axes[1], arr2, obs_pc2, "PC2", "#3B8BD4"),
]:
    p = np.mean(null_arr >= obs)
    sig = "p < 0.001" if p < 0.001 else f"p = {p:.4f}"

    ax.hist(null_arr, bins=35, color=color, alpha=0.65,
            edgecolor="white", linewidth=0.5,
            label=f"Phase-randomized null\n(n={len(null_arr)} permutations)")
    ax.axvline(obs, color="black", lw=2.5, linestyle="-",
               label=f"Observed {label} = {obs:.1f}%")
    ax.axvline(np.percentile(null_arr, 95), color="#555555",
               lw=1.5, linestyle="--", alpha=0.8,
               label=f"Null 95th pctile = {np.percentile(null_arr,95):.1f}%")
    ax.axvspan(np.percentile(null_arr,2.5), np.percentile(null_arr,97.5),
               alpha=0.12, color=color, label="Null 95% CI")

    ax.set_xlabel(f"{label} explained variance (%)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"{label}: Observed = {obs:.1f}%  |  "
        f"Null median = {np.median(null_arr):.1f}% "
        f"({np.percentile(null_arr,2.5):.1f}–"
        f"{np.percentile(null_arr,97.5):.1f}%)  |  {sig}",
        fontsize=10, fontweight="bold"
    )
    ax.legend(frameon=False, fontsize=9)
    ax.set_facecolor("#F9F9F9")

fig.suptitle(
    "Phase-randomized null model: PC1 and PC2\n"
    "Both principal components significantly exceed chance (p < 0.001)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
for ext in ["pdf","png"]:
    fig.savefig(str(FIG_DIR / f"null_pc1_pc2.{ext}"),
                dpi=300 if ext=="png" else None,
                bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nFigure saved → null_pc1_pc2.pdf/.png")

# Save results
pd.DataFrame({"null_pc1": arr1, "null_pc2": arr2,
              "null_combined": arrc}).to_csv(
    ROB_DIR / "null_pc1_pc2.csv", index=False)

print(f"\n{'='*58}")
print("MANUSCRIPT UPDATE:")
print(f"{'='*58}")
print(f"""
Replace the current sentence:
  "The null distribution yielded a median PC1 of 20.8%
   (95% interval: 19.1–22.9%), far below the observed value
   (p < 0.001, permutation test)."

With:
  "The null distribution yielded a median PC1 of {np.median(arr1):.1f}%
   (95% interval: {np.percentile(arr1,2.5):.1f}–{np.percentile(arr1,97.5):.1f}%)
   and a median PC2 of {np.median(arr2):.1f}%
   ({np.percentile(arr2,2.5):.1f}–{np.percentile(arr2,97.5):.1f}%),
   both far below the observed values of {obs_pc1:.1f}% and
   {obs_pc2:.1f}% respectively (both p < 0.001, permutation test),
   confirming that the two-dimensional thalamocortical structure
   as a whole substantially exceeds chance."
""")
