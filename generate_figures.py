"""
generate_figures.py
--------------------
Standalone publication figure generator for the thalamocortical SEEG manuscript.
Reads from pipeline outputs only — does not re-run any analysis.

Corrections vs. original pipeline script:
  - Nucleus names fixed: "Pulvinar"→"Pul", "AN"→"AV"
  - PCA axis labels updated: 54.0% / 32.9%
  - SEM replaced with 95% CI (1.96 × SEM) on all bar charts
  - Individual subject data points overlaid on bar charts (Fig 4 & 5)
  - Figure numbering aligned to manuscript:
      Fig2 = PCA manifold
      Fig3 = iCoh heatmap (state × frequency)
      Fig4 = Granger directionality  ← was Fig5 in pipeline script
      Fig5 = Nucleus dissociation    ← was Fig4 in pipeline script
      Fig6 = Spectral radar profiles
      FigS1 = Coverage heatmap
      FigS2 = Significant contrasts dot plot

Run from seeg_study directory:
    source /home/chima/seeg_env/bin/activate
    cd /mnt/c/Users/chima/seeg_study
    python generate_figures.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(".")
RES_DIR    = DATA_DIR / "outputs/results"
FIG_DIR    = DATA_DIR / "outputs/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CONN_CSV   = RES_DIR / "step5_connectivity.csv"
STAT_CSV   = RES_DIR / "step6_statistics.csv"
MANIF_CSV  = RES_DIR / "step6_manifold.csv"
GT_CSV     = RES_DIR / "thalamic_nuclei_ground_truth.csv"

# ── Style ──────────────────────────────────────────────────────────────────────
MSTYLE = {
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "pdf.fonttype": 42,
}
plt.rcParams.update(MSTYLE)

STATES_ORDERED = [
    "rem_sleep", "watching_tv", "eating", "playing",
    "talking", "crying", "laughing", "reading", "nrem_sleep"
]

STATE_LABELS = {
    "rem_sleep": "REM sleep", "nrem_sleep": "NREM sleep",
    "watching_tv": "Watching TV", "eating": "Eating",
    "playing": "Playing", "talking": "Talking",
    "crying": "Crying", "laughing": "Laughing", "reading": "Reading",
}

STATE_COLORS = {
    "rem_sleep":    "#4E9AF1",
    "nrem_sleep":   "#2B5EA7",
    "watching_tv":  "#9B8EC4",
    "eating":       "#F4A460",
    "playing":      "#66BB6A",
    "talking":      "#FFA726",
    "crying":       "#EF5350",
    "laughing":     "#AB47BC",
    "reading":      "#26A69A",
}

NUC_COLORS = {
    "CM":  "#E8573C",
    "Pul": "#3B8BD4",
    "AV":  "#2EAA6E",
    "VLP": "#9B59B6",
    "MD":  "#F39C12",
    "VPL": "#1ABC9C",
    "VA":  "#E74C3C",
    "MGN": "#95A5A6",
}

BAND_LABELS = {
    "icoh_delta":    "Delta\n(0.5–4 Hz)",
    "icoh_theta":    "Theta\n(4–8 Hz)",
    "icoh_alpha":    "Alpha\n(8–13 Hz)",
    "icoh_beta":     "Beta\n(13–30 Hz)",
    "icoh_lgamma":   "Low gamma\n(30–70 Hz)",
    "icoh_broadband":"Broadband\n(0.5–70 Hz)",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(str(FIG_DIR / f"{name}.{ext}"),
                    dpi=300 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {name}.pdf / .png")

def ci95(series):
    """Return 95% CI half-width = 1.96 * SEM."""
    n = series.count()
    if n < 2:
        return np.nan
    return 1.96 * series.std(ddof=1) / np.sqrt(n)

def states_in(df, col="state"):
    return [s for s in STATES_ORDERED if s in df[col].unique()]

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading pipeline outputs...")
conn_df   = pd.read_csv(CONN_CSV)
stat_df   = pd.read_csv(STAT_CSV)
manif_df  = pd.read_csv(MANIF_CSV)
band_cols = [c for c in BAND_LABELS if c in conn_df.columns]
print(f"  conn_df:  {len(conn_df)} rows")
print(f"  stat_df:  {len(stat_df)} rows")
print(f"  manif_df: {len(manif_df)} rows")

# ── Figure 1 — Thalamic contact locations ─────────────────────────────────────
print("\nFig 1 — Thalamic contacts (MNI space)")
gt   = pd.read_csv(GT_CSV)
thal = gt[gt["is_thalamic"] == True].copy()
for c in ["x_mni", "y_mni", "z_mni"]:
    thal[c] = pd.to_numeric(thal[c], errors="coerce")
thal = thal.dropna(subset=["x_mni", "y_mni", "z_mni"])

try:
    import nilearn.plotting as nplot
    coords = thal[["x_mni", "y_mni", "z_mni"]].values
    nc     = [NUC_COLORS.get(n, "#999") for n in thal["thalamic_nucleus"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Thalamic SEEG contact locations (n=27, MNI space)",
                 fontsize=12, fontweight="bold")
    for ax, dm, title in zip(axes, ["x", "z", "y"],
                             ["Sagittal", "Axial", "Coronal"]):
        d = nplot.plot_glass_brain(None, display_mode=dm, axes=ax,
                                   colorbar=False, alpha=0.3, title=title)
        d.add_markers(marker_coords=coords,
                      marker_color=nc, marker_size=[30]*len(thal))
except ImportError:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    views = [("x_mni","z_mni","Sagittal (x–z)"),
             ("x_mni","y_mni","Axial (x–y)"),
             ("y_mni","z_mni","Coronal (y–z)")]
    for ax, (xc, yc, title) in zip(axes, views):
        for nuc, grp in thal.groupby("thalamic_nucleus"):
            ax.scatter(grp[xc], grp[yc],
                       c=NUC_COLORS.get(nuc, "#999"),
                       s=35, alpha=0.75, label=nuc)
        ax.set_xlabel(xc.split("_")[0].upper() + " (mm)")
        ax.set_ylabel(yc.split("_")[0].upper() + " (mm)")
        ax.set_title(title)
        ax.axhline(0, lw=0.5, color="gray")
        ax.axvline(0, lw=0.5, color="gray")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Thalamic contact locations (MNI space)",
                 fontsize=12, fontweight="bold")
plt.tight_layout()
save(fig, "Fig1")

# ── Figure 2 — PCA manifold ────────────────────────────────────────────────────
print("Fig 2 — PCA manifold")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
for state in states_in(manif_df):
    pts = manif_df[manif_df["state"] == state]
    ax.scatter(pts["PC1"], pts["PC2"],
               c=STATE_COLORS.get(state, "#999"),
               alpha=0.35, s=18, zorder=2)
centroids = manif_df.groupby("state")[["PC1","PC2"]].mean()
for state, row in centroids.iterrows():
    c = STATE_COLORS.get(state, "#999")
    ax.scatter(row["PC1"], row["PC2"], c=c, s=130,
               edgecolors="white", linewidths=1.5, zorder=3)
    ax.text(row["PC1"]+0.05, row["PC2"]+0.05,
            STATE_LABELS.get(state, state),
            fontsize=8, color=c, fontweight="bold")
ax.set_xlabel("PC1 (54.0% variance)")          # CORRECTED
ax.set_ylabel("PC2 (32.9% variance)")          # CORRECTED
ax.set_title("Thalamocortical connectivity manifold",
             fontsize=11, fontweight="bold")
ax.axhline(0, lw=0.5, color="gray")
ax.axvline(0, lw=0.5, color="gray")

ax2 = axes[1]
pc1 = manif_df.groupby("state")["PC1"].mean().reindex(STATES_ORDERED).dropna()
ax2.barh(range(len(pc1)), pc1.values,
         color=[STATE_COLORS.get(s,"#999") for s in pc1.index],
         alpha=0.85, edgecolor="white")
ax2.set_yticks(range(len(pc1)))
ax2.set_yticklabels([STATE_LABELS.get(s,s) for s in pc1.index], fontsize=10)
ax2.set_xlabel("PC1 score")
ax2.set_title("State ordering along PC1\n"
              "(negative = passive/social; positive = cognitive/sleep)",
              fontsize=11, fontweight="bold")
ax2.axvline(0, lw=0.8, color="black")
plt.tight_layout()
save(fig, "Fig2")

# ── Figure 3 — iCoh heatmap (state × frequency) ───────────────────────────────
print("Fig 3 — iCoh heatmap")
sub   = conn_df[conn_df["nucleus"] == "all_thalamus"]
pivot = (sub.groupby("state")[band_cols].mean()
         .reindex([s for s in STATES_ORDERED if s in sub["state"].unique()]))
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlBu_r")
ax.set_xticks(range(len(band_cols)))
ax.set_xticklabels([BAND_LABELS[b] for b in band_cols], fontsize=9)
ax.set_yticks(range(len(pivot)))
ax.set_yticklabels([STATE_LABELS.get(s,s) for s in pivot.index], fontsize=10)
ax.tick_params(left=False, bottom=False)
mn, sd = pivot.values.mean(), pivot.values.std()
for i in range(len(pivot)):
    for j in range(len(band_cols)):
        v  = pivot.values[i,j]
        tc = "white" if abs(v-mn) > 0.5*sd else "black"
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                fontsize=8, color=tc)
plt.colorbar(im, ax=ax, shrink=0.8, label="iCoh")
ax.set_title("Thalamocortical iCoh: state × frequency band\n"
             "(all_thalamus, n=27, 524 independent recordings)",
             fontsize=11, fontweight="bold", pad=12)
plt.tight_layout()
save(fig, "Fig3")

# ── Figure 4 — Granger directionality ─────────────────────────────────────────
# (was Fig5 in original script — corrected to match manuscript)
print("Fig 4 — Granger directionality (manuscript Fig 4)")
sub_a  = conn_df[conn_df["nucleus"] == "all_thalamus"]
s_ord  = states_in(sub_a)
x      = np.arange(len(s_ord))
w      = 0.35

# Subject-level means for TC and CT
subj_tc = sub_a.groupby(["state","subject_id"])["granger_tc"].mean().reset_index()
subj_ct = sub_a.groupby(["state","subject_id"])["granger_ct"].mean().reset_index()
subj_net= sub_a.groupby(["state","subject_id"])["granger_net"].mean().reset_index()

tc_mean = subj_tc.groupby("state")["granger_tc"].mean().reindex(s_ord)
ct_mean = subj_ct.groupby("state")["granger_ct"].mean().reindex(s_ord)
net_mean= subj_net.groupby("state")["granger_net"].mean().reindex(s_ord)
tc_ci   = subj_tc.groupby("state")["granger_tc"].apply(ci95).reindex(s_ord)
ct_ci   = subj_ct.groupby("state")["granger_ct"].apply(ci95).reindex(s_ord)
net_ci  = subj_net.groupby("state")["granger_net"].apply(ci95).reindex(s_ord)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel — TC vs CT grouped bars + 95% CI
ax = axes[0]
ax.bar(x-w/2, tc_mean.values, w, label="Thalamus→Cortex",
       color="#E8573C", alpha=0.8, edgecolor="white",
       yerr=tc_ci.values, capsize=4, error_kw={"linewidth":1.2})
ax.bar(x+w/2, ct_mean.values, w, label="Cortex→Thalamus",
       color="#3B8BD4", alpha=0.8, edgecolor="white",
       yerr=ct_ci.values, capsize=4, error_kw={"linewidth":1.2})
# Overlay individual subject points
for i, state in enumerate(s_ord):
    tc_s = subj_tc[subj_tc["state"]==state]["granger_tc"].values
    ct_s = subj_ct[subj_ct["state"]==state]["granger_ct"].values
    jit  = (np.random.RandomState(42).rand(len(tc_s))-0.5)*0.18
    ax.scatter(i-w/2+jit[:len(tc_s)], tc_s, c="#E8573C",
               s=12, alpha=0.45, zorder=3)
    ax.scatter(i+w/2+jit[:len(ct_s)], ct_s, c="#3B8BD4",
               s=12, alpha=0.45, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels([STATE_LABELS.get(s,s) for s in s_ord],
                   rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Spectral Granger causality")
ax.set_title("Thalamocortical directionality by state",
             fontsize=11, fontweight="bold")
ax.legend(frameon=False, fontsize=9)

# Right panel — Net Granger + 95% CI + individual points
ax2 = axes[1]
bar_colors = ["#E8573C" if v>0 else "#3B8BD4" for v in net_mean.values]
ax2.bar(range(len(s_ord)), net_mean.values,
        color=bar_colors, alpha=0.85, edgecolor="white",
        yerr=net_ci.values, capsize=4, error_kw={"linewidth":1.2})
for i, state in enumerate(s_ord):
    net_s = subj_net[subj_net["state"]==state]["granger_net"].values
    jit   = (np.random.RandomState(42+i).rand(len(net_s))-0.5)*0.3
    ax2.scatter(i+jit, net_s,
                c=["#E8573C" if v>0 else "#3B8BD4" for v in net_s],
                s=12, alpha=0.45, zorder=3)
ax2.axhline(0, lw=1, color="black")
ax2.set_xticks(range(len(s_ord)))
ax2.set_xticklabels([STATE_LABELS.get(s,s) for s in s_ord],
                    rotation=35, ha="right", fontsize=8)
ax2.set_ylabel("Net Granger (TC − CT)")
ax2.set_title("Net directionality\n"
              "(red = thalamus-driven; blue = cortex-driven)",
              fontsize=11, fontweight="bold")
if "crying" in s_ord:
    idx = s_ord.index("crying")
    val = net_mean["crying"]
    ax2.annotate("Only TC-dominant state\n(gc_net = \u22120.0007)",
                 xy=(idx, val), xytext=(idx+0.9, val+0.006),
                 fontsize=7.5, color="#E8573C",
                 arrowprops=dict(arrowstyle="->", color="#E8573C", lw=1))
plt.tight_layout()
save(fig, "Fig4")

# ── Figure 5 — Nucleus dissociation ───────────────────────────────────────────
# (was Fig4 in original script — corrected nucleus names and 95% CI)
print("Fig 5 — Nucleus dissociation (manuscript Fig 5)")
NUCLEI = [
    ("CM",  NUC_COLORS["CM"],  "Centromedian (CM)"),
    ("Pul", NUC_COLORS["Pul"], "Pulvinar (Pul)"),       # FIXED: was "Pulvinar"
    ("AV",  NUC_COLORS["AV"],  "Anterior ventral (AV)"),# FIXED: was "AN"
]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax_i, (nucleus, color, label) in enumerate(NUCLEI):
    ax   = axes[ax_i]
    sub2 = conn_df[conn_df["nucleus"] == nucleus]
    if not len(sub2):
        ax.set_visible(False)
        print(f"  WARNING: no data for nucleus '{nucleus}'")
        continue
    s_ord2 = states_in(sub2)
    # Subject-level means
    subj_m = (sub2.groupby(["state","subject_id"])["icoh_broadband"]
              .mean().reset_index())
    sm  = subj_m.groupby("state")["icoh_broadband"].mean().reindex(s_ord2)
    ci  = subj_m.groupby("state")["icoh_broadband"].apply(ci95).reindex(s_ord2)
    ax.bar(range(len(sm)), sm.values,
           yerr=ci.values,
           color=[STATE_COLORS.get(s,"#999") for s in sm.index],
           alpha=0.85, edgecolor="white", capsize=4,
           error_kw={"linewidth":1.2})
    # Individual subject points
    for i, state in enumerate(s_ord2):
        pts = subj_m[subj_m["state"]==state]["icoh_broadband"].values
        jit = (np.random.RandomState(i*7).rand(len(pts))-0.5)*0.3
        ax.scatter(i+jit, pts, c="black", s=10, alpha=0.35, zorder=3)
    ax.set_xticks(range(len(sm)))
    ax.set_xticklabels([STATE_LABELS.get(s,s) for s in sm.index],
                       rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Broadband iCoh")
    ax.set_title(label, fontsize=11, fontweight="bold", color=color)
    ax.set_ylim(0, sm.max()*1.35)
    # Annotate peak state
    peak = sm.idxmax()
    ax.text(list(sm.index).index(peak), sm.max()*1.15,
            "peak", ha="center", fontsize=7.5, color=color, style="italic")
fig.suptitle("Nucleus-specific thalamocortical coupling:\n"
             "CM peaks during reading, Pulvinar peaks during crying",
             fontsize=11, fontweight="bold")
plt.tight_layout()
save(fig, "Fig5")

# ── Figure 6 — Spectral radar profiles ────────────────────────────────────────
print("Fig 6 — Spectral radar profiles")
N      = len(band_cols)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]
band_short = ["Delta", "Theta", "Alpha", "Beta", "L.gamma", "Broadband"]
fig, axes = plt.subplots(3, 3, figsize=(14, 14),
                         subplot_kw=dict(polar=True))
axes = axes.flatten()
sub_a = conn_df[conn_df["nucleus"] == "all_thalamus"]
for ax_i, state in enumerate(STATES_ORDERED):
    if ax_i >= len(axes): break
    ax   = axes[ax_i]
    sub2 = sub_a[sub_a["state"] == state]
    if not len(sub2):
        ax.set_visible(False)
        continue
    vals  = [sub2[b].mean() for b in band_cols]
    vals += vals[:1]
    ax.plot(angles, vals,
            color=STATE_COLORS.get(state,"#999"), linewidth=2)
    ax.fill(angles, vals,
            color=STATE_COLORS.get(state,"#999"), alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(band_short, fontsize=8)
    ax.set_title(STATE_LABELS.get(state, state),
                 fontsize=10, fontweight="bold",
                 color=STATE_COLORS.get(state,"#333"), pad=12)
    ax.tick_params(labelleft=False)
    ax.spines["polar"].set_visible(False)
    ax.grid(color="gray", alpha=0.3)
for i in range(len(STATES_ORDERED), len(axes)):
    axes[i].set_visible(False)
fig.suptitle("Thalamocortical iCoh spectral profiles by state\n(all_thalamus, n=27)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
save(fig, "Fig6")

# ── Figure S1 — Coverage heatmap ──────────────────────────────────────────────
print("Fig S1 — Coverage heatmap")
# Build from connectivity CSV (n_clips_total per subject×state)
dedup = (conn_df.groupby(["state","subject_id"]).first().reset_index())
pn = (dedup.groupby(["subject_id","state"])["n_clips_total"]
      .sum().unstack(fill_value=0))
pe = (dedup.groupby(["subject_id","state"])["n_epochs_total"]
      .sum().unstack(fill_value=0))
for df_ in [pn, pe]:
    df_.index = df_.index.astype(str)
    df_.sort_index(inplace=True)
pn = pn.reindex(columns=[s for s in STATES_ORDERED if s in pn.columns])
pe = pe.reindex(columns=[s for s in STATES_ORDERED if s in pe.columns])
fig, axes = plt.subplots(1, 2, figsize=(16, 10))
for ax, data, title in zip(
        axes,
        [pn, pe],
        ["Recordings per subject × state",
         "Total epochs per subject × state"]):
    im = ax.imshow(data.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([STATE_LABELS.get(s,s) for s in data.columns],
                       rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=8)
    ax.set_xlabel("Behavioural state")
    ax.set_ylabel("Subject")
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            v  = data.values[i,j]
            if v > 0:
                tc = "white" if v > data.values.max()*0.6 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color=tc)
    plt.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle("Supplementary Figure S1 — Data coverage\n"
             "(n=27 subjects, 524 independent recordings)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save(fig, "FigS1")

# ── Figure S2 — Significant contrasts dot plot ────────────────────────────────
print("Fig S2 — Significant contrasts")
sig = (stat_df[stat_df["significant"]==True]
       .sort_values("q_fdr").head(30).copy())
if len(sig):
    sig["label"] = (
        sig["state1"].map(STATE_LABELS).fillna(sig["state1"]) + " vs " +
        sig["state2"].map(STATE_LABELS).fillna(sig["state2"]) + "\n(" +
        sig["metric"].str.replace("icoh_","").str.replace("_"," ") + ")")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(sig["cohen_d"], range(len(sig)),
               c=["#E8573C" if d>0 else "#3B8BD4" for d in sig["cohen_d"]],
               s=[-np.log10(q)*40 for q in sig["q_fdr"]],
               alpha=0.8, zorder=3)
    ax.axvline(0, lw=0.8, color="black")
    ax.axvline(0.5,  lw=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(-0.5, lw=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(range(len(sig)))
    ax.set_yticklabels(sig["label"], fontsize=8)
    ax.set_xlabel("Cohen's d (effect size)")
    ax.set_title(f"Supplementary Figure S2 — Significant pairwise contrasts\n"
                 f"(top {len(sig)} of {int(stat_df['significant'].sum())}, "
                 f"FDR q < 0.05; point size \u221d \u2212log\u2081\u2080(q))",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=[
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor=c, markersize=8, label=lbl)
        for lbl, c in [("State 1 > State 2","#E8573C"),
                       ("State 1 < State 2","#3B8BD4")]
    ], frameon=False, fontsize=9, loc="lower right")
    plt.tight_layout()
    save(fig, "FigS2")
else:
    print("  No significant contrasts found — FigS2 skipped")

print(f"\nAll figures saved to: {FIG_DIR.resolve()}")
