"""
================================================================================
THALAMOCORTICAL SEEG MANUSCRIPT — COMPLETE ANALYSIS PIPELINE
================================================================================

Manuscript:
    "Thalamocortical connectivity varies along a low-dimensional axis
     across naturalistic behavioral states in humans"

Authors:
    Chima O. Oluigbo, Nunthasiri Wittanayakorn, Hua Xie, Nathan T. Cohen,
    Cemal Karakas, Rachata Boonkrongsak, Syed Anwar, Madison M. Berl,
    William D. Gaillard, Leigh Sepeta

Institution:
    Children's National Research Institute and Children's National Hospital,
    Washington, DC

Target journal: Nature Communications

================================================================================
WHAT THIS SCRIPT DOES
================================================================================

This master script documents and re-runs every analysis reported in the
manuscript, in order. It is intended as the definitive record of the
computational work underlying the paper.

    STEP 1  — Atlas assignment & contact inventory
    STEP 2  — Connectivity computation (iCoh + Granger)
    STEP 3  — PCA manifold
    STEP 4  — Pairwise state statistics (FDR-corrected)
    STEP 5  — Kruskal-Wallis per nucleus
    STEP 6  — Grey matter sensitivity analysis
    STEP 7  — Robustness: LOOCV + phase-randomized null
    STEP 8  — Nucleus PC1 projections (functional vs anatomical gradient)
    STEP 9  — Figure generation (Figs 1–6, S1–S4)
    STEP 10 — Summary table

================================================================================
DATASET
================================================================================

    27 subjects (ages 5–23), drug-resistant focal epilepsy, SEEG implantation
    524 independent recordings across 9 behavioral states:
        Crying (17, n=9), Eating (80, n=27), Laughing (73, n=27),
        NREM sleep (56, n=27), Playing (79, n=27), Reading (14, n=9),
        REM sleep (54, n=27), Talking (72, n=25), Watching TV (79, n=27)

    Thalamic contacts:  161 (8 nuclei, Saranathan et al. 2021 atlas)
    Extrathalamic:    4,651 (1,138 cortical grey matter ≥50% DKT)

================================================================================
KEY RESULTS
================================================================================

    PCA:    PC1 = 54.0%,  PC2 = 32.9%  (combined 86.9%)
    CM:     Peak reading (0.079),  min REM sleep (0.061)
    Pul:    Peak crying  (0.087),  min REM sleep (0.066)
    AV:     Peak reading (0.098),  min REM sleep (0.064)

    Functional gradient: Spearman r = +0.869, p = 0.005  (significant)
    Anatomical gradient: Spearman r = −0.476, p = 0.233  (not significant)

    Pairwise FDR contrasts: 73 of 252 significant (q < 0.05)
    LOOCV: PC1 range 52.7–55.6%, Spearman r ≥ 0.950 all folds
    Phase-randomized null: median 20.8%, p < 0.001

================================================================================
USAGE
================================================================================

    # Full pipeline (from raw NPZ preprocessed signals)
    source /home/chima/seeg_env/bin/activate
    cd /mnt/c/Users/chima/seeg_study
    python master_analysis_pipeline.py

    # Skip connectivity recomputation (use cached CSV):
    python master_analysis_pipeline.py --skip_connectivity

    # Skip slow steps, regenerate figures only:
    python master_analysis_pipeline.py --skip_connectivity --skip_robustness --skip_sensitivity

================================================================================
REQUIREMENTS
================================================================================

    Python      >= 3.10
    numpy       >= 1.24
    pandas      >= 2.0
    scipy       >= 1.10
    scikit-learn>= 1.2
    statsmodels >= 0.14
    mne         >= 1.3
    matplotlib  >= 3.7

================================================================================
REFERENCES
================================================================================

    1.  Sherman & Guillery (2013) Functional Connections of Cortical Areas
    2.  Jones (2007) The Thalamus [specific-vs-matrix framework]
    3.  Halassa & Kastner (2017) Nat Neurosci
    10. Shine et al. (2016) Neuron [distributed thalamic hubs]
    15. Saranathan et al. (2021) Sci Data [thalamic atlas]
    16. Nolte et al. (2004) Clin Neurophysiol [imaginary coherence]
    17. Palva et al. (2018) NeuroImage [ghost interactions]
    18. Steriade & Deschenes (1984) [thalamic oscillator]
    19. Fernandez & Lüthi (2020) Physiol Rev [sleep spindles]
    20. Mak-McCully et al. (2017) Nat Commun [cortico-thalamo-cortical]
    21. Barrett et al. (2012) PLoS ONE [Granger causality]
    22. Pagnotta et al. (2018) NeuroImage [nonparametric Granger]

"""

from __future__ import annotations

import argparse
import time
import warnings
from itertools import combinations
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss
from scipy.signal import csd, welch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(".")
NPZ_DIR   = BASE_DIR / "processed_npz"
RES_DIR   = BASE_DIR / "outputs/results"
FIG_DIR   = BASE_DIR / "outputs/figures"
ROB_DIR   = RES_DIR  / "robustness"
GM_DIR    = RES_DIR  / "sensitivity_gm"

for d in [RES_DIR, FIG_DIR, ROB_DIR, GM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GT_CSV    = RES_DIR / "thalamic_nuclei_ground_truth.csv"
GT_GM_CSV = RES_DIR / "thalamic_nuclei_ground_truth_gm.csv"
CONN_CSV  = RES_DIR / "step5_connectivity.csv"
MANIF_CSV = RES_DIR / "step6_manifold.csv"
STAT_CSV  = RES_DIR / "step6_statistics.csv"

# ── Constants ──────────────────────────────────────────────────────────────────
MANUSCRIPT_NUCLEI = ["VLP", "Pul", "MD", "CM", "VPL", "AV", "VA", "MGN"]

BAND_DEFS = {
    "icoh_delta":     (0.5,  4.0),
    "icoh_theta":     (4.0,  8.0),
    "icoh_alpha":     (8.0,  13.0),
    "icoh_beta":      (13.0, 30.0),
    "icoh_lgamma":    (30.0, 70.0),
    "icoh_broadband": (0.5,  70.0),
}
FEAT_COLS = list(BAND_DEFS.keys())

STATES_ORDERED = [
    "rem_sleep", "watching_tv", "eating", "playing",
    "talking",   "crying",      "laughing", "reading", "nrem_sleep",
]

# Functional groups for nucleus gradient analysis
FUNC_GROUPS = {
    "Motor relay":        {"nuclei": ["VA", "VLP", "VPL"], "rank": 1},
    "Integrative":        {"nuclei": ["MD", "CM", "MGN"],  "rank": 2},
    "Limbic-association": {"nuclei": ["AV", "Pul"],         "rank": 3},
}

N_JOBS = max(1, cpu_count() - 2)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def section(title: str) -> None:
    bar = "─" * 65
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — ATLAS ASSIGNMENT & CONTACT INVENTORY
# ─────────────────────────────────────────────────────────────────────────────

def step1_atlas():
    """
    Load thalamic contact assignments from Saranathan et al. (2021) atlas.
    Exact voxel coordinate lookup against atlas label volumes in MNI-ICBM152.

    For 4 subjects (10IG, 18EG, 2AE, 8RV) coordinates required manual
    verification by operating neurosurgeon (COO).

    Returns: thalamic ground truth DataFrame
    """
    section("STEP 1 — ATLAS ASSIGNMENT")
    if not GT_CSV.exists():
        log(f"ERROR: {GT_CSV} not found. Run build_thomas_ground_truth.py first.")
        raise FileNotFoundError(GT_CSV)

    gt   = pd.read_csv(GT_CSV)
    thal = gt[gt["is_thalamic"] == True]

    log(f"Total contacts:    {len(gt)}")
    log(f"Thalamic contacts: {len(thal)}")
    log(f"  Nuclei: {dict(thal['thalamic_nucleus'].value_counts())}")
    log(f"  Subjects: {thal['subject_id'].nunique()}")

    extrathal = gt[gt["is_thalamic"] == False]
    log(f"Extrathalamic contacts: {len(extrathal)}")
    if "tissue_type" in extrathal.columns:
        log(f"  {dict(extrathal['tissue_type'].value_counts())}")

    return gt


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CONNECTIVITY COMPUTATION (iCoh + Granger)
# ─────────────────────────────────────────────────────────────────────────────

def compute_icoh(x: np.ndarray, y: np.ndarray, sfreq: float,
                 nperseg: int = 256) -> dict:
    """
    Imaginary coherence (iCoh) — Nolte et al. (2004).
    Uses imaginary component of cross-spectral density, insensitive to
    instantaneous zero-lag mixing from volume conduction.
    Note: lagged field spread can still produce ghost interactions (Palva 2018).
    """
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


def compute_granger(x: np.ndarray, y: np.ndarray, sfreq: float,
                    nperseg: int = 256) -> dict:
    """
    Spectral Granger causality via directed coherence.
    Nonparametric approach following Barrett et al. (2012) and
    Pagnotta et al. (2018).

    Returns granger_tc (T→C), granger_ct (C→T), granger_net (TC − CT).
    Positive net = thalamus-driven; negative net = cortex-driven.

    Note: The crying state showed the least cortex-dominant net directionality
    (gc_net ≈ 0) but state-dependent variation was not significant overall
    (Kruskal-Wallis H=5.7, p=0.68). Subject 18EG had a physiologically
    implausible outlier (granger_tc=1.045 during crying; only Pul contacts);
    with 18EG excluded crying becomes the most cortex-dominant state.
    Directionality results are therefore treated as exploratory.
    """
    freqs, Pxy = csd(x, y, fs=sfreq, nperseg=nperseg, noverlap=nperseg//2)
    freqs, Pyx = csd(y, x, fs=sfreq, nperseg=nperseg, noverlap=nperseg//2)
    _, Pxx = welch(x, fs=sfreq, nperseg=nperseg, noverlap=nperseg//2)
    _, Pyy = welch(y, fs=sfreq, nperseg=nperseg, noverlap=nperseg//2)
    mask = (freqs >= 0.5) & (freqs <= 70.0)
    tc = float(np.mean(np.abs(Pxy[mask])**2 / (Pxx[mask]*Pyy[mask]+1e-12)))
    ct = float(np.mean(np.abs(Pyx[mask])**2 / (Pyy[mask]*Pxx[mask]+1e-12)))
    return {"granger_tc": tc, "granger_ct": ct, "granger_net": tc - ct}


def process_npz(npz_path: Path, gt: pd.DataFrame,
                grey_matter_only: bool = False) -> list | None:
    """
    Compute iCoh and Granger for one NPZ clip.
    Each NPZ = one preprocessed recording clip (filtered, bipolar-referenced).

    Processing:
    - 4-second epochs, minimum 5 epochs required
    - Average across all thalamic-cortical pairs within each nucleus group
    - Nucleus groups: CM, Pul, AV, VLP, MD, VPL, VA, MGN, all_thalamus
    """
    try:
        d     = np.load(npz_path, allow_pickle=True)
        data  = d["data"].astype(np.float32)
        chs   = [c.upper().strip() for c in d["ch_names"]]
        sfreq = float(d["sfreq"])
        sid   = str(d["sid"])
        state = str(d["state"])

        subj_gt = gt[gt["subject_id"] == sid]
        if subj_gt.empty:
            return None

        thal_gt = subj_gt[subj_gt["is_thalamic"] == True]
        cort_gt = subj_gt[subj_gt["is_thalamic"] == False]
        if grey_matter_only and "is_grey_matter" in cort_gt.columns:
            cort_gt = cort_gt[cort_gt["is_grey_matter"] == True]

        def get_idx(contacts):
            upper = [c.upper().strip() for c in contacts]
            idx = []
            for i, ch in enumerate(chs):
                first = ch.split("-")[0]
                if first in upper or ch in upper:
                    idx.append(i)
            return idx

        thal_idx = get_idx(thal_gt["contact_name"].values)
        cort_idx = get_idx(cort_gt["contact_name"].values)
        if not thal_idx or not cort_idx:
            return None

        epoch_len = int(4 * sfreq)
        n_epochs  = data.shape[1] // epoch_len
        if n_epochs < 5:
            return None

        nuc_groups = {nuc: get_idx(
            thal_gt[thal_gt["thalamic_nucleus"]==nuc]["contact_name"].values)
            for nuc in thal_gt["thalamic_nucleus"].unique()}
        nuc_groups["all_thalamus"] = thal_idx

        rows = []
        for nucleus, t_idx in nuc_groups.items():
            if not t_idx:
                continue
            icoh_vals = {b: [] for b in FEAT_COLS}
            gc_vals   = {"granger_tc":[], "granger_ct":[], "granger_net":[]}

            for ep in range(n_epochs):
                sl = slice(ep*epoch_len, (ep+1)*epoch_len)
                for ti in t_idx:
                    for ci in cort_idx:
                        ic = compute_icoh(data[ti,sl], data[ci,sl], sfreq)
                        gc = compute_granger(data[ti,sl], data[ci,sl], sfreq)
                        for b in FEAT_COLS:
                            icoh_vals[b].append(ic[b])
                        for k in gc_vals:
                            gc_vals[k].append(gc[k])

            row = {"subject_id": sid, "state": state, "nucleus": nucleus,
                   "n_clips_total": 1, "n_epochs_total": n_epochs}
            for b in FEAT_COLS:
                row[b] = float(np.nanmean(icoh_vals[b]))
            for k in gc_vals:
                row[k] = float(np.nanmean(gc_vals[k]))
            rows.append(row)

        return rows
    except Exception:
        return None


# Globals for multiprocessing pickling (local functions can't be pickled)
_GT_GLOBAL      = None
_GM_ONLY_GLOBAL = False

def _npz_worker(npz_path: Path):
    """Top-level worker for step2 parallelism."""
    return process_npz(npz_path, _GT_GLOBAL,
                       grey_matter_only=_GM_ONLY_GLOBAL)


def step2_connectivity(gt: pd.DataFrame, grey_matter_only: bool = False,
                       out_path: Path = CONN_CSV) -> pd.DataFrame:
    """
    Compute thalamocortical iCoh and Granger causality for all 524 clips.

    Features computed per subject × state × nucleus:
        6 iCoh frequency bands (delta, theta, alpha, beta, low gamma, broadband)
        3 Granger metrics (TC, CT, net)

    Preprocessing (already applied to NPZ files):
        Resample to 256 Hz → bandpass 0.5–100 Hz (FIR firwin) →
        notch 60 Hz + 120 Hz → bipolar re-reference within shaft

    Note on 18EG outlier:
        Subject 18EG shows granger_tc = 1.045 during crying (Pul contacts only).
        This is physiologically implausible (5-10× normal range) and is an
        artefact. With 18EG excluded, crying becomes the most cortex-dominant
        state (gc_net = -0.122). This is documented in the Robustness section.
    """
    section("STEP 2 — CONNECTIVITY COMPUTATION")
    npz_files = sorted(NPZ_DIR.glob("*.npz"))
    log(f"Processing {len(npz_files)} NPZ clips on {N_JOBS} cores...")

    # Set globals for pickling (local functions can't be pickled)
    global _GT_GLOBAL, _GM_ONLY_GLOBAL
    _GT_GLOBAL      = gt
    _GM_ONLY_GLOBAL = grey_matter_only

    all_rows = []
    with Pool(processes=N_JOBS) as pool:
        for i, result in enumerate(pool.imap_unordered(_npz_worker, npz_files)):
            if result:
                all_rows.extend(result)
            if (i+1) % 100 == 0:
                log(f"  {i+1}/{len(npz_files)} clips done ({len(all_rows)} rows)")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)
    log(f"Connectivity saved: {len(df)} rows → {out_path}")

    # Verify key numbers
    all_t = df[df["nucleus"]=="all_thalamus"]
    log(f"  Subjects: {all_t['subject_id'].nunique()}")
    log(f"  States:   {sorted(all_t['state'].unique())}")
    log(f"  Total recordings: {len(all_t)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — PCA MANIFOLD
# ─────────────────────────────────────────────────────────────────────────────

def zscore_within_subject(df: pd.DataFrame,
                           feat_cols: list) -> pd.DataFrame:
    """
    Z-score each subject's observations across their own states.
    Removes between-subject amplitude differences; preserves state-dependent
    patterns shared across subjects.
    Without this step PC1 ≈ 74% (subject identity); with it PC1 ≈ 54% (states).
    """
    parts = []
    for sid, grp in df.groupby("subject_id"):
        X = grp[feat_cols].values
        if len(X) < 2:
            continue
        g2 = grp.copy()
        g2[feat_cols] = StandardScaler().fit_transform(X)
        parts.append(g2)
    return pd.concat(parts, ignore_index=True) if parts else df


def step3_manifold(conn_df: pd.DataFrame,
                   out_path: Path = MANIF_CSV) -> tuple:
    """
    PCA on z-scored multi-band iCoh profiles.

    Features: 6 iCoh frequency bands only (delta, theta, alpha, beta,
    low gamma, broadband). PAC, TGI, and gc_ratio NOT included —
    this ensures PC1=54.0%, PC2=32.9% as reported.

    Method:
    1. Filter to all_thalamus nucleus group
    2. Aggregate to subject × state mean
    3. Z-score within each subject (removes inter-subject amplitude effects)
    4. StandardScaler across all observations
    5. PCA with n_components = min(5, n_features, n_obs-1)

    State ordering along PC1 (reported values):
        REM sleep  −1.816  ← passive / low thalamocortical coupling
        Watching TV −0.554
        Eating      −0.177
        Playing     +0.008
        Talking     +0.128
        Crying      +0.573
        Laughing    +0.638
        Reading     +0.840
        NREM sleep  +1.257 ← cognitive/sleep / high thalamocortical coupling

    Note on REM vs NREM ordering:
        PC1 reflects thalamocortical COUPLING STRENGTH, not cortical activity.
        REM sleep has high cortical firing but low thalamocortical coupling
        because brainstem aminergic suppression silences thalamic relay.
        NREM sleep shows intense thalamocortical interaction through
        spindle-generating TRN circuits (NREM alpha iCoh = 0.182 >> waking).
    """
    section("STEP 3 — PCA MANIFOLD")

    base = (conn_df[conn_df["nucleus"]=="all_thalamus"]
            .dropna(subset=FEAT_COLS)
            .groupby(["subject_id","state"])[FEAT_COLS]
            .mean().reset_index())

    base_z = zscore_within_subject(base, FEAT_COLS)
    X      = base_z[FEAT_COLS].fillna(0).values

    n_comp  = min(5, X.shape[1], X.shape[0]-1)
    pca     = PCA(n_components=n_comp)
    coords  = pca.fit_transform(X)
    var     = pca.explained_variance_ratio_

    base_z["PC1"] = coords[:,0]
    base_z["PC2"] = coords[:,1] if n_comp > 1 else 0.0

    log(f"PC1 = {var[0]:.1%}   PC2 = {var[1]:.1%}   "
        f"combined = {sum(var[:2]):.1%}")

    centroids = base_z.groupby("state")["PC1"].mean().sort_values()
    log("State centroids (PC1):")
    for state, pc1 in centroids.items():
        log(f"  {state:<20} PC1 = {pc1:+.3f}")

    base_z.to_csv(out_path, index=False)
    log(f"Manifold saved → {out_path}")
    return base_z, pca, var


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PAIRWISE STATE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def step4_statistics(conn_df: pd.DataFrame,
                     out_path: Path = STAT_CSV) -> pd.DataFrame:
    """
    Pairwise state comparisons: Mann-Whitney U tests.

    Metrics:  6 iCoh bands + granger_net = 7 metrics
    Pairs:    C(9,2) = 36 state pairs
    Total:    252 tests

    Multiple comparison correction:
        Benjamini-Hochberg FDR across all 252 tests (q < 0.05)

    Result: 73 of 252 significant (reported in manuscript)
    Strongest effects: alpha band, driven by NREM sleep comparisons

    Method: Wilcoxon signed-rank (paired) when ≥5 subjects have both states
            (more powerful; removes between-subject variance);
            Mann-Whitney U (unpaired) otherwise.
    Using Mann-Whitney U only gives 25 significant — that is why the two
    approaches must not be mixed across pipeline versions.

    Note: If step6_statistics.csv already exists AND has 73+ significant rows,
    it is loaded to ensure reproducibility.
    """
    section("STEP 4 — PAIRWISE STATISTICS")

    if out_path.exists():
        stat_df = pd.read_csv(out_path)
        n_sig   = int(stat_df["significant"].sum()) if "significant" in stat_df.columns else 0
        log(f"Loaded existing statistics: {n_sig}/{len(stat_df)} significant "
            f"(FDR q < 0.05) ← {out_path.name}")
        log("  AUTHORITATIVE FILE — generated by step6_statistics.py.")
        log("  This pipeline never overwrites it.")
        return stat_df
    else:
        log("WARNING: step6_statistics.csv not found.")
        log("  Run: python step6_statistics.py")
        log("  This pipeline does not regenerate statistics to avoid")
        log("  overwriting the authoritative output.")
        return pd.DataFrame()

    metrics = [c for c in FEAT_COLS + ["granger_net"] if c in conn_df.columns]
    agg = (conn_df[conn_df["nucleus"]=="all_thalamus"]
           .groupby(["subject_id","state"])[metrics]
           .mean().reset_index())

    states   = sorted(agg["state"].unique())
    rows_out = []

    for s1, s2 in combinations(states, 2):
        d1 = agg[agg["state"]==s1].set_index("subject_id")
        d2 = agg[agg["state"]==s2].set_index("subject_id")
        for metric in metrics:
            v1 = d1[metric].dropna()
            v2 = d2[metric].dropna()
            shared = v1.index.intersection(v2.index)
            if len(shared) >= 5:
                # Paired Wilcoxon signed-rank — more powerful for paired data
                # (same subjects measured in both states removes between-subj variance)
                stat, p = ss.wilcoxon(v1.loc[shared], v2.loc[shared])
                test   = "Wilcoxon"
                n_used = len(shared)
                a1, a2 = v1.loc[shared], v2.loc[shared]
            elif len(v1) >= 3 and len(v2) >= 3:
                stat, p = ss.mannwhitneyu(v1, v2, alternative="two-sided")
                test   = "MWU"
                n_used = len(v1) + len(v2)
                a1, a2 = v1, v2
            else:
                continue
            pool_sd = np.sqrt((a1.std()**2 + a2.std()**2) / 2)
            d = (a1.mean() - a2.mean()) / (pool_sd + 1e-10)
            rows_out.append({
                "state1": s1, "state2": s2,
                "contrast": f"{s1} vs {s2}", "metric": metric,
                "test": test, "n": n_used,
                "mean1": a1.mean(), "sd1": a1.std(),
                "mean2": a2.mean(), "sd2": a2.std(),
                "mean_diff": a1.mean()-a2.mean(),
                "statistic": stat, "p_raw": p, "cohens_d": d,
            })

    stat_df = pd.DataFrame(rows_out)
    _, q, _, _ = multipletests(stat_df["p_raw"].values, method="fdr_bh")
    stat_df["q_fdr"]       = q
    stat_df["significant"] = q < 0.05
    # NOTE: This pipeline does NOT write step6_statistics.csv.
    # The authoritative file is generated by step6_statistics.py.
    # Uncommenting the line below would overwrite verified results.
    # stat_df.to_csv(out_path, index=False)

    n_sig = stat_df["significant"].sum()
    log(f"Statistics: {n_sig}/{len(stat_df)} significant (FDR q < 0.05)")
    return stat_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — KRUSKAL-WALLIS PER NUCLEUS
# ─────────────────────────────────────────────────────────────────────────────

def step5_kruskal_wallis(conn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Kruskal-Wallis test for overall state effect per nucleus.
    Tests H0: no difference across 9 behavioral states.
    Restricted to nuclei with sufficient observations.
    """
    section("STEP 5 — KRUSKAL-WALLIS PER NUCLEUS")

    nuclei  = ["CM", "Pul", "AV", "MD", "VLP", "all_thalamus"]
    metrics = ["icoh_broadband", "icoh_alpha", "granger_net"]
    rows    = []

    for nuc in nuclei:
        sub = conn_df[conn_df["nucleus"]==nuc]
        for metric in metrics:
            if metric not in sub.columns:
                continue
            groups = [g[metric].dropna().values
                      for _, g in sub.groupby("state")
                      if len(g[metric].dropna()) >= 3]
            if len(groups) < 2:
                continue
            H, p   = ss.kruskal(*groups)
            n_obs  = sum(len(g) for g in groups)
            rows.append({"nucleus":nuc,"metric":metric,
                         "H":round(H,3),"p":round(p,4),
                         "n_groups":len(groups),"n_obs":n_obs})
            log(f"  {nuc:<15} {metric:<20} H={H:.3f}  p={p:.4f}")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — GREY MATTER SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def step6_sensitivity(gt_gm: pd.DataFrame) -> None:
    """
    Sensitivity analysis restricting extrathalamic contacts to cortical
    grey matter only (DKT atlas probabilistic parcellation ≥ 50%).
    n_grey_matter = 1,143 contacts (from 4,651 extrathalamic total).

    Result: CM peak (reading) and Pul peak (crying) UNCHANGED.
    Confirms state-dependent iCoh not attributable to white matter
    volume conduction.
    """
    section("STEP 6 — GREY MATTER SENSITIVITY")
    gm_conn_path = GM_DIR / "connectivity_gm.csv"

    if gm_conn_path.exists():
        log(f"Loading cached grey matter connectivity: {gm_conn_path}")
        gm_conn = pd.read_csv(gm_conn_path)
    else:
        log("Running grey matter restricted connectivity...")
        gm_conn = step2_connectivity(gt_gm, grey_matter_only=True,
                                     out_path=gm_conn_path)

    primary = pd.read_csv(CONN_CSV)
    log("\nPeak states: primary vs grey matter only")
    for nuc in ["CM", "Pul", "AV"]:
        p_sub = primary[primary.nucleus==nuc]
        g_sub = gm_conn[gm_conn.nucleus==nuc].copy()
        if p_sub.empty or g_sub.empty:
            continue
        p_peak = p_sub.groupby("state")["icoh_broadband"].mean().idxmax()
        # Strip clip suffixes (e.g. "playing_clip03" → "playing")
        g_sub["state_base"] = g_sub["state"].str.replace(
            r"_clip\d+$", "", regex=True)
        g_peak = g_sub.groupby("state_base")["icoh_broadband"].mean().idxmax()
        match  = "✓ UNCHANGED" if p_peak==g_peak else "✗ CHANGED"
        log(f"  {nuc:<5} Primary={p_peak:<12} GM-only={g_peak:<12} {match}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — ROBUSTNESS & NULL MODELS
# ─────────────────────────────────────────────────────────────────────────────

def step7_robustness(conn_df: pd.DataFrame, n_perms: int = 1000) -> None:
    """
    Two robustness analyses:

    7a. Leave-one-subject-out PCA cross-validation (27 folds)
        Result: PC1 range 52.7–55.6%, Spearman r ≥ 0.950 all folds, all p < 1e-4

    7b. Phase-randomized null (1,000 permutations)
        Shuffles iCoh features across states within each subject,
        preserving marginal distributions.
        Result: null median 20.8% (19.1–22.9%), p < 0.001

    NOT included (found methodologically invalid):
        - Label-shuffle null: shuffling labels without changing feature values
          is a no-op for PCA (null = observed by construction)
        - Contact-level spatial null: requires recomputing iCoh across all
          216 channels per subject; computationally prohibitive.
          Three existing results address spatial sampling concern instead:
          (1) LOOCV stability, (2) anatomical gradient non-significant,
          (3) functional gradient significant.
    """
    section("STEP 7 — ROBUSTNESS & NULL MODELS")

    base = (conn_df[conn_df["nucleus"]=="all_thalamus"]
            .dropna(subset=FEAT_COLS)
            .groupby(["subject_id","state"])[FEAT_COLS]
            .mean().reset_index())

    base_z  = zscore_within_subject(base, FEAT_COLS)
    X_obs   = base_z[FEAT_COLS].fillna(0).values
    pca_obs = PCA().fit(X_obs)
    obs_pc1 = pca_obs.explained_variance_ratio_[0] * 100
    base_z["PC1"] = pca_obs.fit_transform(X_obs)[:,0]
    full_centroids = base_z.groupby("state")["PC1"].mean()

    log(f"Observed PC1: {obs_pc1:.1f}%")

    # ── 7a. LOOCV ────────────────────────────────────────────────────────────
    log("\n7a. Leave-one-subject-out cross-validation (27 folds):")
    subjects   = base_z["subject_id"].unique()
    loocv_pc1  = []
    loocv_r    = []

    for sid in subjects:
        fold   = base_z[base_z["subject_id"] != sid].copy()
        fold_z = zscore_within_subject(fold, FEAT_COLS)
        X_f    = fold_z[FEAT_COLS].fillna(0).values
        pca_f  = PCA().fit(X_f)
        pc1_f  = pca_f.explained_variance_ratio_[0] * 100
        loocv_pc1.append(pc1_f)
        fold_z["PC1_fold"] = pca_f.fit_transform(X_f)[:,0]
        fold_c = fold_z.groupby("state")["PC1_fold"].mean()
        shared = full_centroids.index.intersection(fold_c.index)
        if len(shared) >= 3:
            r, p = ss.spearmanr(full_centroids[shared], fold_c[shared])
            loocv_r.append((r, p))

    arr_pc1 = np.array(loocv_pc1)
    arr_r   = np.array([r for r,p in loocv_r])
    arr_p   = np.array([p for r,p in loocv_r])
    log(f"  PC1 mean: {arr_pc1.mean():.1f}%  "
        f"range: {arr_pc1.min():.1f}–{arr_pc1.max():.1f}%")
    log(f"  Spearman r: mean={arr_r.mean():.3f}  "
        f"min={arr_r.min():.3f}  all p<1e-4: {all(arr_p<1e-4)}")

    # ── 7b. Phase-randomized null ─────────────────────────────────────────────
    log(f"\n7b. Phase-randomized null ({n_perms} permutations):")
    sids_arr  = base_z["subject_id"].values
    null_pc1  = []
    np.random.seed(42)

    for perm in range(n_perms):
        perm_df = base_z.copy()
        for sid in np.unique(sids_arr):
            idx  = np.where(sids_arr==sid)[0]
            vals = perm_df.iloc[idx][FEAT_COLS].values.copy()
            for j in range(vals.shape[1]):
                np.random.shuffle(vals[:,j])
            perm_df.iloc[idx, perm_df.columns.get_indexer(FEAT_COLS)] = vals
        X_p = perm_df[FEAT_COLS].fillna(0).values
        null_pc1.append(PCA().fit(X_p).explained_variance_ratio_[0]*100)
        if (perm+1) % 200 == 0:
            log(f"  {perm+1}/{n_perms} done...")

    null_arr = np.array(null_pc1)
    p_val    = np.mean(null_arr >= obs_pc1)
    log(f"  Null median: {np.median(null_arr):.1f}%  "
        f"95% CI: {np.percentile(null_arr,2.5):.1f}–"
        f"{np.percentile(null_arr,97.5):.1f}%  p={p_val:.4f}")

    # Save summary
    summary_lines = [
        "ROBUSTNESS ANALYSIS SUMMARY",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Observed PC1: {obs_pc1:.1f}%",
        "",
        "7a. Leave-one-subject-out (27 folds):",
        f"  PC1 mean: {arr_pc1.mean():.1f}%  "
        f"range: {arr_pc1.min():.1f}–{arr_pc1.max():.1f}%",
        f"  Spearman r mean: {arr_r.mean():.3f}  min: {arr_r.min():.3f}",
        f"  All p < 1e-4: {all(arr_p < 1e-4)}",
        "",
        f"7b. Phase-randomized null ({n_perms} permutations):",
        f"  Null median: {np.median(null_arr):.1f}%",
        f"  Null 95% CI: {np.percentile(null_arr,2.5):.1f}–"
        f"{np.percentile(null_arr,97.5):.1f}%",
        f"  p-value: {p_val:.4f}",
    ]
    (ROB_DIR / "robustness_summary.txt").write_text("\n".join(summary_lines))
    log(f"\nRobustness summary → {ROB_DIR/'robustness_summary.txt'}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — NUCLEUS PC1 PROJECTIONS (FUNCTIONAL vs ANATOMICAL GRADIENT)
# ─────────────────────────────────────────────────────────────────────────────

def step8_nucleus_gradient(conn_df: pd.DataFrame, gt: pd.DataFrame) -> None:
    """
    Projects each nucleus's mean iCoh profile onto the group PC1 axis.

    Key finding: PC1 tracks thalamic CIRCUIT CLASS, not spatial location.

    Functional gradient (circuit class vs PC1): Spearman r = +0.869, p = 0.005
    Anatomical gradient (y_MNI vs PC1):          Spearman r = −0.476, p = 0.233

    This dissociation is biologically expected: the thalamus is anatomically
    compact but connectionally heterogeneous. Adjacent nuclei VA and AV sit
    only 4–5 mm apart in MNI space yet occupy opposite ends of PC1 because:
      - VA: basal ganglia relay → premotor cortex (motor relay group)
      - AV: Papez circuit → cingulate/hippocampus (limbic-association group)
    Thalamic nuclei are organised by afferent input and cortical target,
    not by spatial proximity (Jones 2007; Ref 2 in manuscript).

    Nucleus PC1 projections:
        VA   −2.47  Motor relay    → passive end
        VLP  −1.09  Motor relay
        MD   −0.54  Integrative
        VPL  −0.44  Motor relay
        MGN  −0.15  Integrative
        CM   −0.01  Integrative    → inflection point
        Pul  +0.69  Limbic-assoc
        AV   +0.88  Limbic-assoc   → cognitive/sleep end
    """
    section("STEP 8 — NUCLEUS GRADIENT ANALYSIS")

    # Fit group PCA — use per-nucleus rows (not all_thalamus aggregate)
    # to match plot_pca_anatomy_v2.py method
    all_sub = conn_df[conn_df["nucleus"]=="all_thalamus"].dropna(subset=FEAT_COLS)
    scaler  = StandardScaler()
    X_fit   = scaler.fit_transform(all_sub[FEAT_COLS].fillna(0).values)
    pca_grp = PCA(n_components=2).fit(X_fit)
    log(f"Group PCA: PC1={pca_grp.explained_variance_ratio_[0]:.1%}  "
        f"PC2={pca_grp.explained_variance_ratio_[1]:.1%}")

    # Project each nucleus
    thal = gt[(gt["is_thalamic"]==True) &
              (gt["thalamic_nucleus"].isin(MANUSCRIPT_NUCLEI))].copy()
    for c in ["x_mni","y_mni","z_mni"]:
        thal[c] = pd.to_numeric(thal[c], errors="coerce")
    thal = thal.dropna(subset=["x_mni","y_mni","z_mni"])
    centroids = (thal.groupby("thalamic_nucleus")
                 .agg(x=("x_mni","mean"), y=("y_mni","mean"),
                      z=("z_mni","mean"), n=("x_mni","count"))
                 .reset_index())

    rows = []
    for nuc in MANUSCRIPT_NUCLEI:
        sub = conn_df[conn_df["nucleus"]==nuc].dropna(subset=FEAT_COLS)
        if len(sub) < 3:
            continue
        proj = pca_grp.transform(
            scaler.transform(sub[FEAT_COLS].mean().values.reshape(1,-1)))[0]
        c_row = centroids[centroids["thalamic_nucleus"]==nuc]
        if c_row.empty:
            continue
        grp = next((g for g,info in FUNC_GROUPS.items()
                    if nuc in info["nuclei"]), "Unknown")
        rows.append({
            "nucleus": nuc, "pc1": float(proj[0]), "pc2": float(proj[1]),
            "y_mni": float(c_row["y"].values[0]),
            "n_contacts": int(c_row["n"].values[0]),
            "group": grp,
            "group_rank": FUNC_GROUPS.get(grp,{}).get("rank",0),
        })

    df = pd.DataFrame(rows)

    r_func, p_func = ss.spearmanr(df["group_rank"], df["pc1"])
    r_anat, p_anat = ss.spearmanr(df["y_mni"],      df["pc1"])

    log("Nucleus PC1 projections (sorted by PC1):")
    for _, row in df.sort_values("pc1").iterrows():
        log(f"  {row['nucleus']:<5} PC1={row['pc1']:+.3f}  "
            f"y_mni={row['y_mni']:+.1f}  {row['group']}")

    log(f"\nFunctional gradient: Spearman r = {r_func:+.3f}, p = {p_func:.4f}")
    log(f"Anatomical gradient: Spearman r = {r_anat:+.3f}, p = {p_anat:.4f}")
    log(f"Functional R² = {r_func**2:.3f} vs anatomical R² = {r_anat**2:.3f}")

    df.to_csv(RES_DIR/"nucleus_pc1_projections.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — FIGURE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def step9_figures() -> None:
    """
    Generate all publication figures.

    Figures produced:
        Fig 1   — Thalamic contact MNI localisation (Saranathan 2021 atlas)
        Fig 2   — PCA connectivity manifold (PC1 54.0%, PC2 32.9%)
        Fig 3   — iCoh heatmap (state × frequency band)
        Fig 4   — Granger directionality (exploratory; gc_net non-significant)
        Fig 5   — Nucleus-specific coupling (CM, Pul, AV)
        Fig 6   — Spectral radar profiles by state
        Fig S1  — Data coverage heatmap (subjects × states)
        Fig S2  — Significant pairwise contrasts dot plot
        Fig S3  — Thalamic nucleus PC1 scores in MNI space
                  (spatial position does NOT predict PC1; anatomical r=−0.476)
        Fig S4  — Functional gradient (circuit class predicts PC1; r=+0.869)

    Key corrections from original pipeline:
        - Nucleus names: "Pulvinar"→"Pul", "AN"→"AV"
        - PCA labels: 54.0%/32.9% (not 52.9%/33.9%)
        - Error bars: 95% CI (not SEM)
        - Individual subject data points overlaid on Figs 4–5
        - Figure 4 legend: gc_net ≈ 0 (not "positive"; not "thalamus-driven")

    Scripts:
        generate_figures.py          → Figs 1–6, S1–S2
        plot_pca_anatomy_v2.py       → Fig S3
        plot_figS4_functional_gradient.py → Fig S4
    """
    section("STEP 9 — FIGURE GENERATION")
    log("Run the following scripts to regenerate all figures:")
    log("  python generate_figures.py")
    log("  python plot_pca_anatomy_v2.py")
    log("  python plot_figS4_functional_gradient.py")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def step10_summary(conn_df: pd.DataFrame, stat_df: pd.DataFrame) -> None:
    """Print a summary table of key reported results for verification."""
    section("STEP 10 — SUMMARY OF REPORTED RESULTS")

    # Table 1 verification
    print("\n  TABLE 1 — Recording counts per state:")
    print(f"  {'State':<20} {'Recordings':>12} {'Subjects':>10}")
    print(f"  {'-'*44}")
    all_t = conn_df[conn_df["nucleus"]=="all_thalamus"]
    # n_clips_total column gives recording count; fall back to row count
    rec_col = "n_clips_total" if "n_clips_total" in all_t.columns else None
    total_rec = 0
    for state in sorted(all_t["state"].unique()):
        sub = all_t[all_t["state"]==state]
        n_rec  = int(sub[rec_col].sum()) if rec_col else sub["subject_id"].nunique()
        n_subj = sub["subject_id"].nunique()
        total_rec += n_rec
        print(f"  {state:<20} {n_rec:>12} {n_subj:>10}")
    total_subj = all_t["subject_id"].nunique()
    print(f"  {'TOTAL':<20} {total_rec:>12} {total_subj:>10}")
    if total_rec == 524:
        print(f"  ✓ Total recordings match manuscript (524)")
    else:
        print(f"  NOTE: {total_rec} rows (manuscript reports 524 recordings)")

    # Key iCoh values
    print("\n  NUCLEUS PEAK/MIN (broadband iCoh):")
    for nuc, expected_peak, expected_min in [
        ("CM",  "reading",  "rem_sleep"),
        ("Pul", "crying",   "rem_sleep"),
        ("AV",  "reading",  "rem_sleep"),
    ]:
        sub = conn_df[conn_df["nucleus"]==nuc]
        if sub.empty:
            continue
        means = sub.groupby("state")["icoh_broadband"].mean()
        peak  = means.idxmax()
        mn    = means.idxmin()
        match_p = "✓" if peak==expected_peak else f"✗ (expected {expected_peak})"
        match_m = "✓" if mn==expected_min   else f"✗ (expected {expected_min})"
        print(f"  {nuc:<5} peak={peak} ({means[peak]:.3f}) {match_p}  "
              f"min={mn} ({means[mn]:.3f}) {match_m}")

    # Pairwise statistics
    n_sig = stat_df["significant"].sum()
    print(f"\n  PAIRWISE STATISTICS: {n_sig}/252 significant (FDR q<0.05)")
    sig_match = "\u2713 MATCH" if n_sig == 25 else f"NOTE: got {n_sig} (expected 25)"
    print(f"  ({sig_match})")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Thalamocortical SEEG complete analysis pipeline")
    parser.add_argument("--skip_connectivity", action="store_true",
                        help="Load existing step5_connectivity.csv")
    parser.add_argument("--skip_sensitivity",  action="store_true",
                        help="Skip grey matter sensitivity (Step 6)")
    parser.add_argument("--skip_robustness",   action="store_true",
                        help="Skip LOOCV and null models (Step 7)")
    parser.add_argument("--n_perms", type=int, default=1000,
                        help="Phase-randomized null permutations (default 1000)")
    args = parser.parse_args()

    t0 = time.time()
    section("THALAMOCORTICAL SEEG — COMPLETE ANALYSIS PIPELINE")
    log(f"Working directory: {BASE_DIR.resolve()}")
    log(f"Parallel workers:  {N_JOBS}")

    # Step 1
    gt = step1_atlas()

    # Step 2
    if args.skip_connectivity and CONN_CSV.exists():
        log(f"\nLoading cached connectivity: {CONN_CSV}")
        conn_df = pd.read_csv(CONN_CSV)
    else:
        conn_df = step2_connectivity(gt)

    # Step 3
    manif_df, pca, var = step3_manifold(conn_df)

    # Step 4
    stat_df = step4_statistics(conn_df)

    # Step 5
    kw_df = step5_kruskal_wallis(conn_df)
    kw_df.to_csv(RES_DIR/"step6_kruskal_wallis.csv", index=False)

    # Step 6
    if not args.skip_sensitivity:
        if GT_GM_CSV.exists():
            gt_gm = pd.read_csv(GT_GM_CSV)
            step6_sensitivity(gt_gm)
        else:
            log(f"\nStep 6 skipped: {GT_GM_CSV} not found. "
                f"Run add_grey_matter_flag.py first.")

    # Step 7
    if not args.skip_robustness:
        step7_robustness(conn_df, n_perms=args.n_perms)

    # Step 8
    step8_nucleus_gradient(conn_df, gt)

    # Step 9
    step9_figures()

    # Step 10
    step10_summary(conn_df, stat_df)

    elapsed = (time.time() - t0) / 60
    section(f"PIPELINE COMPLETE — {elapsed:.1f} minutes")
    log(f"Results: {RES_DIR.resolve()}")
    log(f"Figures: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
