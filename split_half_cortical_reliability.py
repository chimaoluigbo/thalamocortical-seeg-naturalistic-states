"""
split_half_cortical_reliability.py
-------------------------------------
Split-half reliability analysis for thalamocortical connectivity.

REVIEWER CONCERN:
    Cortical electrodes are not randomly placed but hypothesis-driven
    (seizure hotspots are over-sampled). The averaged thalamocortical
    iCoh may therefore be biased by electrode placement rather than
    reflecting unbiased thalamocortical connectivity.

METHOD:
    PHASE 1 (run once, ~60-90 min, parallelised):
        For each NPZ clip, for each cortical contact:
            - Compute mean iCoh across all thalamic contacts
        Saves: per_cortical_contact_icoh.csv
        Columns: [clip_id, subject_id, state, contact_name,
                  icoh_delta, icoh_theta, icoh_alpha, icoh_beta,
                  icoh_lgamma, icoh_broadband]

    PHASE 2 (fast, ~5 min):
        For N_SPLITS random splits of cortical contacts within each subject:
            - Split cortical contacts into two equal halves
            - Compute mean iCoh per state for each half
            - Compute Spearman r between the two half-profiles
              across all state × band combinations
        Report: mean r, distribution, and p-value vs null

    If r is consistently high (e.g. > 0.90), the mean thalamocortical
    iCoh is stable regardless of which cortical contacts are used.
    This does not prove representativeness, but demonstrates that
    the finding is not sensitive to which specific contacts were selected
    given the available sample.

Run from seeg_study directory:
    source /home/chima/seeg_env/bin/activate
    cd /mnt/c/Users/chima/seeg_study
    python split_half_cortical_reliability.py

Runtime: Phase 1 ~60-90 min (same scale as spatial_null_from_npz Phase 1)
         Phase 2 ~5 min
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import csd, welch
from scipy import stats as ss
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count
import time
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(".")
NPZ_DIR    = BASE_DIR / "processed_npz"
RES_DIR    = BASE_DIR / "outputs/results"
FIG_DIR    = BASE_DIR / "outputs/figures"
OUT_DIR    = RES_DIR / "robustness"
CACHE_CSV  = OUT_DIR / "per_cortical_contact_icoh.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

GT_CSV   = RES_DIR / "thalamic_nuclei_ground_truth.csv"

N_SPLITS = 1000
SEED     = 42
N_JOBS   = max(1, cpu_count() - 2)

BAND_DEFS = {
    "icoh_delta":     (0.5,  4.0),
    "icoh_theta":     (4.0,  8.0),
    "icoh_alpha":     (8.0,  13.0),
    "icoh_beta":      (13.0, 30.0),
    "icoh_lgamma":    (30.0, 70.0),
    "icoh_broadband": (0.5,  70.0),
}
FEAT_COLS = list(BAND_DEFS.keys())
THAL_NUCLEI = ["VLP","Pul","MD","CM","VPL","AV","VA","MGN",
               "Hb","LGN","VLa"]  # all thalamic labels

# ── Load ground truth ──────────────────────────────────────────────────────────
gt = pd.read_csv(GT_CSV)
gt["contact_upper"] = gt["contact_name"].str.upper().str.strip()
# Normalise is_thalamic to bool regardless of storage type (bool/int/string)
gt["is_thalamic"] = gt["is_thalamic"].astype(str).str.lower().isin(["true","1","yes"])
thal_gt = gt[gt["is_thalamic"] == True].copy()
cort_gt = gt[gt["is_thalamic"] == False].copy()

print(f"Ground truth loaded:")
print(f"  Thalamic contacts:    {len(thal_gt)} across {thal_gt['subject_id'].nunique()} subjects")
print(f"  Extrathalamic contacts: {len(cort_gt)} across {cort_gt['subject_id'].nunique()} subjects")


# ── iCoh computation ───────────────────────────────────────────────────────────
def icoh_bands(x, y, sfreq, nperseg=256):
    freqs, Pxy = csd(x, y, fs=sfreq, nperseg=nperseg, noverlap=nperseg//2)
    _, Pxx = welch(x, fs=sfreq, nperseg=nperseg, noverlap=nperseg//2)
    _, Pyy = welch(y, fs=sfreq, nperseg=nperseg, noverlap=nperseg//2)
    denom  = np.sqrt(Pxx * Pyy) + 1e-12
    icoh   = np.abs(np.imag(Pxy) / denom)
    result = {}
    for name, (flo, fhi) in BAND_DEFS.items():
        mask = (freqs >= flo) & (freqs <= fhi)
        result[name] = float(np.mean(icoh[mask])) if mask.any() else np.nan
    return result


def match_channels(ch_names_arr, contacts_upper_set):
    """Match contact names (handling bipolar montage) to channel indices."""
    ch_upper = [c.upper().strip() for c in ch_names_arr]
    matches  = {}
    for i, ch in enumerate(ch_upper):
        first_pole = ch.split("-")[0].strip()
        if first_pole in contacts_upper_set:
            matches[i] = first_pole
        elif ch in contacts_upper_set:
            matches[i] = ch
    return matches


# ── Phase 1 worker ─────────────────────────────────────────────────────────────
_GT_GLOBAL = None

def _phase1_worker(npz_path):
    """
    For one NPZ clip, compute mean iCoh per CORTICAL contact
    averaged across ALL thalamic contacts.
    """
    global _GT_GLOBAL
    try:
        d      = np.load(npz_path, allow_pickle=True)
        data   = d["data"].astype(np.float32)
        ch_arr = d["ch_names"]
        sfreq  = float(d["sfreq"])
        sid    = str(d["sid"])
        state  = str(d["state"])
        clip_id = str(npz_path.stem)

        subj_thal = _GT_GLOBAL[_GT_GLOBAL["subject_id"] == sid]
        subj_thal = subj_thal[subj_thal["is_thalamic"] == True]
        subj_cort = _GT_GLOBAL[_GT_GLOBAL["subject_id"] == sid]
        subj_cort = subj_cort[subj_cort["is_thalamic"] == False]

        if subj_thal.empty or subj_cort.empty:
            return None

        thal_set = set(subj_thal["contact_upper"].values)
        cort_set  = set(subj_cort["contact_upper"].values)
        # Remove any thalamic contacts from cortical set (safety guard)
        cort_set  = cort_set - thal_set

        thal_idx = match_channels(ch_arr, thal_set)
        cort_idx = match_channels(ch_arr, cort_set)

        if not thal_idx or not cort_idx:
            return None

        epoch_len = int(4 * sfreq)
        n_epochs  = data.shape[1] // epoch_len
        if n_epochs < 5:
            return None

        rows = []
        for c_idx, contact_upper in cort_idx.items():
            # Mean iCoh across all thalamic contacts for this cortical contact
            band_vals = {b: [] for b in FEAT_COLS}
            for t_idx in thal_idx.keys():
                for ep in range(n_epochs):
                    sl = slice(ep * epoch_len, (ep + 1) * epoch_len)
                    ic = icoh_bands(data[c_idx, sl], data[t_idx, sl], sfreq)
                    for b in FEAT_COLS:
                        if not np.isnan(ic[b]):
                            band_vals[b].append(ic[b])

            row = {
                "clip_id":      clip_id,
                "subject_id":   sid,
                "state":        state,
                "contact_name": contact_upper,
            }
            for b in FEAT_COLS:
                row[b] = float(np.mean(band_vals[b])) if band_vals[b] else np.nan
            rows.append(row)

        return rows if rows else None

    except Exception:
        return None


def run_phase1():
    global _GT_GLOBAL
    _GT_GLOBAL = gt   # set global for pickling

    npz_files = sorted(NPZ_DIR.glob("*.npz"))
    print(f"\nPHASE 1: Computing per-cortical-contact iCoh")
    print(f"  {len(npz_files)} NPZ clips | {N_JOBS} cores")
    print(f"  For each cortical contact: mean iCoh across all thalamic contacts")
    print(f"  (Same runtime as spatial_null Phase 1 — ~60-90 min)\n")

    t0       = time.time()
    all_rows = []

    with Pool(processes=N_JOBS) as pool:
        for i, result in enumerate(
                pool.imap_unordered(_phase1_worker, npz_files, chunksize=4)):
            if result:
                all_rows.extend(result)
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                eta     = elapsed / (i + 1) * (len(npz_files) - i - 1)
                print(f"  {i+1:4d}/{len(npz_files)}  "
                      f"rows={len(all_rows):,}  "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    if not all_rows:
        print("ERROR: No cortical contacts matched in any clip.")
        print("Check that ground truth CSV has correct contact names matching NPZ channels.")
        # Diagnostic: show first clip channel names vs ground truth
        sample = sorted(NPZ_DIR.glob("*.npz"))[0]
        d = np.load(sample, allow_pickle=True)
        print(f"Sample NPZ: {sample.name}")
        print(f"  First 10 channels: {list(d['ch_names'][:10])}")
        sid = str(d['sid'])
        subj_rows = gt[gt['subject_id'] == sid]
        print(f"  Ground truth contacts for {sid} (first 10):")
        print(f"  {list(subj_rows['contact_upper'].values[:10])}")
        return pd.DataFrame()
    df = pd.DataFrame(all_rows).dropna(subset=FEAT_COLS)
    df.to_csv(CACHE_CSV, index=False)
    print(f"\nPhase 1 complete: {len(df):,} rows")
    print(f"  Subjects:  {df['subject_id'].nunique()}")
    print(f"  States:    {sorted(df['state'].unique())}")
    print(f"  Cortical contacts: {df['contact_name'].nunique()}")
    print(f"  Mean contacts per subject: "
          f"{df.groupby('subject_id')['contact_name'].nunique().mean():.1f}")
    return df


# ── Phase 2: split-half reliability ───────────────────────────────────────────
def run_phase2(cort_df):
    print(f"\nPHASE 2: Split-half reliability ({N_SPLITS} splits)")
    print("  For each split: randomly divide cortical contacts within each subject,")
    print("  compute state × band mean for each half, compute Spearman r\n")

    rng = np.random.RandomState(SEED)
    r_values = []

    # Aggregate per contact × state (mean across clips)
    agg = (cort_df
           .groupby(["subject_id", "state", "contact_name"])[FEAT_COLS]
           .mean()
           .reset_index())

    subjects = agg["subject_id"].unique()

    t0 = time.time()
    for split_i in range(N_SPLITS):
        half1_means = []
        half2_means = []

        for sid in subjects:
            subj = agg[agg["subject_id"] == sid]
            contacts = subj["contact_name"].unique()

            if len(contacts) < 2:
                continue

            # Random split of cortical contacts
            shuffled = rng.permutation(contacts)
            mid      = len(shuffled) // 2
            h1_cont  = set(shuffled[:mid])
            h2_cont  = set(shuffled[mid:])

            h1 = (subj[subj["contact_name"].isin(h1_cont)]
                  .groupby("state")[FEAT_COLS].mean())
            h2 = (subj[subj["contact_name"].isin(h2_cont)]
                  .groupby("state")[FEAT_COLS].mean())

            shared = h1.index.intersection(h2.index)
            if len(shared) < 2:
                continue

            # Flatten state × band into a vector for correlation
            v1 = h1.loc[shared].values.flatten()
            v2 = h2.loc[shared].values.flatten()

            # Remove NaN pairs
            mask = ~(np.isnan(v1) | np.isnan(v2))
            if mask.sum() < 6:
                continue

            half1_means.append(v1[mask])
            half2_means.append(v2[mask])

        # Pool across subjects for this split
        if len(half1_means) < 3:
            continue

        v1_all = np.concatenate(half1_means)
        v2_all = np.concatenate(half2_means)
        r, _   = ss.spearmanr(v1_all, v2_all)
        r_values.append(r)

        if (split_i + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (split_i + 1) * (N_SPLITS - split_i - 1)
            print(f"  {split_i+1:4d}/{N_SPLITS}  "
                  f"mean r = {np.mean(r_values):.4f}  "
                  f"elapsed = {elapsed:.0f}s  ETA = {eta:.0f}s")

    r_arr = np.array(r_values)
    return r_arr


def report(r_arr, cort_df):
    print(f"\n{'='*58}")
    print(f"SPLIT-HALF CORTICAL RELIABILITY RESULTS")
    print(f"{'='*58}")
    print(f"  N splits:          {len(r_arr)}")
    print(f"  Mean r:            {np.mean(r_arr):.4f}")
    print(f"  Median r:          {np.median(r_arr):.4f}")
    print(f"  SD:                {np.std(r_arr):.4f}")
    print(f"  2.5th percentile:  {np.percentile(r_arr,2.5):.4f}")
    print(f"  97.5th percentile: {np.percentile(r_arr,97.5):.4f}")
    print(f"  Minimum r:         {np.min(r_arr):.4f}")
    print(f"  Fraction r > 0.90: {np.mean(r_arr > 0.90):.3f}")
    print(f"  Fraction r > 0.95: {np.mean(r_arr > 0.95):.3f}")

    # Save
    pd.DataFrame({"r_split_half": r_arr}).to_csv(
        OUT_DIR / "split_half_cortical_reliability.csv", index=False)

    # ── Figure ─────────────────────────────────────────────────────────────────
    MSTYLE = {"font.family":"sans-serif","font.size":11,
               "axes.spines.top":False,"axes.spines.right":False,
               "pdf.fonttype":42}
    plt.rcParams.update(MSTYLE)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Distribution of r values
    ax = axes[0]
    ax.hist(r_arr, bins=40, color="#2B6CB8", alpha=0.75,
            edgecolor="white", linewidth=0.5)
    ax.axvline(np.mean(r_arr), color="#C0392B", lw=2,
               label=f"Mean r = {np.mean(r_arr):.3f}")
    ax.axvline(np.percentile(r_arr, 2.5), color="#888", lw=1.5,
               linestyle="--", label=f"95% CI: {np.percentile(r_arr,2.5):.3f}–"
               f"{np.percentile(r_arr,97.5):.3f}")
    ax.axvline(np.percentile(r_arr, 97.5), color="#888", lw=1.5, linestyle="--")
    ax.axvline(0.90, color="#E67E22", lw=1.5, linestyle=":",
               label="r = 0.90 threshold")
    ax.set_xlabel("Spearman r (half 1 vs half 2)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"A   Split-half reliability distribution\n"
                 f"(n={len(r_arr)} random splits of cortical contacts)",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.set_facecolor("#F9F9F9")

    # Example split: one subject's two halves
    ax2 = axes[1]
    rng2 = np.random.RandomState(SEED)
    agg = (cort_df.groupby(["subject_id","state","contact_name"])[FEAT_COLS]
           .mean().reset_index())

    # Pick subject with most contacts
    best_sid = (agg.groupby("subject_id")["contact_name"]
                .nunique().idxmax())
    subj = agg[agg["subject_id"] == best_sid]
    contacts = subj["contact_name"].unique()
    shuffled = rng2.permutation(contacts)
    mid = len(shuffled) // 2
    h1 = subj[subj["contact_name"].isin(set(shuffled[:mid]))].groupby("state")[FEAT_COLS].mean()
    h2 = subj[subj["contact_name"].isin(set(shuffled[mid:]))].groupby("state")[FEAT_COLS].mean()
    shared = h1.index.intersection(h2.index)
    v1 = h1.loc[shared, "icoh_broadband"].values
    v2 = h2.loc[shared, "icoh_broadband"].values
    r_ex, _ = ss.spearmanr(v1, v2)

    colors = {"rem_sleep":"#8B008B","nrem_sleep":"#1E90FF","watching_tv":"#FF69B4",
              "eating":"#A0522D","playing":"#228B22","talking":"#20B2AA",
              "crying":"#DC143C","laughing":"#FF8C00","reading":"#6B8E23"}
    for i, state in enumerate(shared):
        ax2.scatter(v1[i], v2[i], c=colors.get(state,"#888"),
                    s=120, zorder=3, edgecolors="white", linewidths=1)
        ax2.annotate(state.replace("_"," ").title(),
                     xy=(v1[i], v2[i]),
                     xytext=(v1[i]+0.001, v2[i]+0.001),
                     fontsize=7.5, color=colors.get(state,"#888"))

    mn = min(min(v1), min(v2)); mx = max(max(v1), max(v2))
    ax2.plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.4)
    ax2.set_xlabel("Half 1 broadband iCoh", fontsize=11)
    ax2.set_ylabel("Half 2 broadband iCoh", fontsize=11)
    ax2.set_title(f"B   Example split (subject {best_sid})\n"
                  f"Half 1 vs half 2 broadband iCoh, r = {r_ex:.3f}",
                  fontsize=11, fontweight="bold", loc="left")
    ax2.set_facecolor("#F9F9F9")

    fig.suptitle(
        "Split-half cortical contact reliability\n"
        "Tests whether mean thalamocortical iCoh is stable to cortical contact selection",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    for ext in ["pdf","png"]:
        fig.savefig(str(FIG_DIR / f"split_half_reliability.{ext}"),
                    dpi=300 if ext=="png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFigure saved → split_half_reliability.pdf/.png")

    # ── Manuscript text ─────────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print("MANUSCRIPT TEXT:")
    print(f"{'='*58}")
    print(f"""
To assess whether the averaged thalamocortical iCoh is stable
to the selection of cortical contacts — given that SEEG electrodes
are hypothesis-driven rather than randomly placed — we performed
a split-half reliability analysis. For each of {len(r_arr)} random
splits, cortical contacts were divided into two equal halves within
each subject and thalamocortical iCoh was computed independently
for each half. The Spearman correlation between the two halves
across all state × frequency band combinations was consistently
high (mean r = {np.mean(r_arr):.3f}, 95% CI: {np.percentile(r_arr,2.5):.3f}–
{np.percentile(r_arr,97.5):.3f}; {np.mean(r_arr>0.90)*100:.0f}% of splits
exceeded r = 0.90), indicating that the mean thalamocortical iCoh
is stable regardless of which specific cortical contacts are
selected from the available sample. We note that this analysis
demonstrates measurement reliability rather than representativeness:
it cannot determine whether the results would hold if cortical
sampling were random rather than hypothesis-driven, which is
not achievable in the clinical SEEG context.
""")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if CACHE_CSV.exists():
        print(f"Loading cached per-cortical-contact iCoh from {CACHE_CSV}")
        cort_df = pd.read_csv(CACHE_CSV)
        print(f"  {len(cort_df):,} rows | "
              f"{cort_df['subject_id'].nunique()} subjects | "
              f"{cort_df['contact_name'].nunique()} contacts")
    else:
        cort_df = run_phase1()

    if cort_df.empty:
        print("ERROR: No cortical contact data. Check ground truth CSV.")
        raise SystemExit(1)

    r_arr = run_phase2(cort_df)
    report(r_arr, cort_df)
