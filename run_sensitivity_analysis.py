#!/usr/bin/env python3
"""
run_sensitivity_analysis.py
═══════════════════════════════════════════════════════════════════════════════
Grey-matter-only sensitivity analysis for thalamocortical iCoh.

This script re-runs connectivity and statistics using only cortical grey
matter contacts (DKT probability ≥50%) as the cortical reference set,
comparing results against the primary analysis (all non-thalamic contacts).

Rationale:
    The primary analysis labels all non-thalamic contacts as "cortical",
    but 53.7% are in white matter. iCoh from white matter bipolar pairs
    is expected to be near-zero (volume conduction → zero imaginary
    component), diluting the primary estimates. This sensitivity analysis
    tests whether restricting to verified grey matter changes the pattern
    of findings.

    If CM still peaks during reading and Pul still peaks during crying
    with grey-matter-only contacts, the primary findings are robust.

Inputs (must be in seeg_study/ or outputs/results/):
    thalamic_nuclei_ground_truth_gm.csv   (from add_grey_matter_flag.py)
    outputs/results/step5_connectivity.csv (primary analysis, for comparison)
    processed_npz/                         (preprocessed .npz files)

Outputs → outputs/results/sensitivity_gm/:
    connectivity_gm.csv      iCoh/PAC/GC with grey-matter-only cortical ref
    statistics_gm.csv        FDR-corrected state contrasts
    manifold_gm.csv          PCA embedding
    comparison_report.txt    side-by-side vs primary analysis

USAGE
──────────────────────────────────────────────────────────────────────────────
    source /home/chima/seeg_env/bin/activate
    cd /mnt/c/Users/chima/seeg_study/
    python run_sensitivity_analysis.py
"""

import os, sys, time, warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────
DATA_DIR   = Path("/mnt/c/Users/chima/seeg_study")
PROC_DIR   = DATA_DIR / "processed_npz"
OUT_DIR    = DATA_DIR / "outputs" / "results" / "sensitivity_gm"
PRIMARY_CSV= DATA_DIR / "outputs" / "results" / "step5_connectivity.csv"
GT_GM_CSV  = DATA_DIR / "thalamic_nuclei_ground_truth_gm.csv"

SFREQ      = 256
EPOCH_SEC  = 4.0
MIN_EPOCHS = 3
MAX_THAL_CH= 8
MAX_CORT_CH= 20

FREQ_BANDS = {
    "delta":     (0.5,  4),
    "theta":     (4,    8),
    "alpha":     (8,   13),
    "beta":      (13,  30),
    "lgamma":    (30,  70),
    "broadband": (0.5, 70),
}

PRIMARY_NUCLEI = ["CM", "Pul", "AV"]

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Spectral helpers (identical to primary pipeline) ──────────────────────
def _band_mask(freqs, flo, fhi):
    return (freqs >= flo) & (freqs <= fhi)

def _compute_icoh(data, sfreq, thal_idx, cort_idx, epoch_sec=4.0, min_epochs=3):
    from scipy.signal import csd, welch
    ti = thal_idx[:MAX_THAL_CH]
    ci = cort_idx[:MAX_CORT_CH]
    if not ti or not ci: return None
    step     = int(sfreq * epoch_sec)
    nperseg  = max(16, min(int(sfreq), data.shape[1] // 4))
    n_times  = data.shape[1]
    if (n_times - step) // step + 1 < min_epochs: return None
    icoh_accum = {k: [] for k in FREQ_BANDS}
    icoh_rev   = []
    for t in ti:
        for c in ci:
            x, y = data[t], data[c]
            Pxy_l, Pxx_l, Pyy_l = [], [], []
            for s in range(0, n_times - step + 1, step):
                f, pxy = csd(x[s:s+step], y[s:s+step], fs=sfreq, nperseg=nperseg)
                _, pxx = __import__('scipy').signal.welch(x[s:s+step], fs=sfreq, nperseg=nperseg)
                _, pyy = __import__('scipy').signal.welch(y[s:s+step], fs=sfreq, nperseg=nperseg)
                Pxy_l.append(pxy); Pxx_l.append(pxx); Pyy_l.append(pyy)
            if len(Pxy_l) < min_epochs: continue
            Pxy = np.mean(Pxy_l, axis=0)
            Pxx = np.mean(Pxx_l, axis=0)
            Pyy = np.mean(Pyy_l, axis=0)
            denom = np.sqrt(Pxx * Pyy) + 1e-12
            ic = np.abs(np.imag(Pxy)) / denom
            for band, (flo, fhi) in FREQ_BANDS.items():
                m = _band_mask(f, flo, fhi)
                if m.any(): icoh_accum[band].append(ic[m].mean())
            _, pxy_r = csd(y, x, fs=sfreq, nperseg=nperseg)
            ic_r = np.abs(np.imag(pxy_r)) / denom
            m_a = _band_mask(f, 8, 13)
            if m_a.any(): icoh_rev.append(ic_r[m_a].mean())
    out = {f"icoh_{b}": float(np.mean(v)) if v else np.nan
           for b, v in icoh_accum.items()}
    tc = out.get("icoh_alpha", np.nan)
    ct = float(np.mean(icoh_rev)) if icoh_rev else tc
    out["gc_tc"] = tc; out["gc_ct"] = ct
    out["gc_ratio"] = tc / (ct + 1e-10)
    return out

# ── Connectivity worker ───────────────────────────────────────────────────
def _worker(args_tuple):
    sid, state, npz_path, el_sid = args_tuple
    rows = []
    try:
        d     = np.load(npz_path, allow_pickle=True)
        data  = d["data"].astype(np.float64)
        chs   = list(d["ch_names"])
        sfreq = float(d["sfreq"])
    except Exception:
        return rows
    if el_sid is None: return rows

    el = el_sid
    thal_contacts = set(el[el["is_thalamic"]==True]["contact_name"].tolist())

    # KEY DIFFERENCE: only use verified grey matter contacts as cortical ref
    gm_contacts   = set(el[
        (el["is_thalamic"]==False) & (el["is_grey_matter"]==True)
    ]["contact_name"].tolist())

    def _idx(cs):
        return [i for i, ch in enumerate(chs) if ch.split("-")[0] in cs]

    thal_idx = _idx(thal_contacts)
    cort_idx = _idx(gm_contacts)      # grey matter only

    if not thal_idx or not cort_idx:
        log(f"  SKIP {sid}/{state}: thal={len(thal_idx)} gm_cort={len(cort_idx)}")
        return rows

    # all_thalamus
    icoh = _compute_icoh(data, sfreq, thal_idx, cort_idx, EPOCH_SEC, MIN_EPOCHS)
    if icoh is None: return rows

    row_base = {"subject_id":sid,"state":state,"nucleus":"all_thalamus",
                "n_ch_thal":len(thal_idx),"n_ch_cort_gm":len(cort_idx)}
    row_base.update(icoh)
    rows.append(row_base)

    # Per-nucleus
    nuc_col = "thalamic_nucleus"
    for nuc in el[nuc_col].dropna().unique():
        if not nuc or nuc == "background": continue
        nidx = _idx(set(el[el[nuc_col]==nuc]["contact_name"].tolist()))
        if not nidx: continue
        icoh_n = _compute_icoh(data, sfreq, nidx, cort_idx, EPOCH_SEC, MIN_EPOCHS)
        if icoh_n is None: continue
        row = {"subject_id":sid,"state":state,"nucleus":str(nuc),
               "n_ch_thal":len(nidx),"n_ch_cort_gm":len(cort_idx)}
        row.update(icoh_n)
        rows.append(row)

    return rows

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    for f in [GT_GM_CSV, PROC_DIR]:
        if not Path(f).exists():
            log(f"ERROR: not found: {f}"); sys.exit(1)

    # Load grey-matter ground truth
    log("Loading grey matter ground truth...")
    gt = pd.read_csv(GT_GM_CSV)
    log(f"  {len(gt)} contacts | "
        f"thalamic={gt['is_thalamic'].sum()} | "
        f"grey_matter={gt['is_grey_matter'].sum()}")

    electrodes = {sid: grp.reset_index(drop=True)
                  for sid, grp in gt.groupby("subject_id")}

    # Discover .npz files
    npz_files = list(PROC_DIR.glob("*.npz"))
    if not npz_files:
        log(f"ERROR: no .npz files in {PROC_DIR}"); sys.exit(1)

    tasks = []
    for npz in sorted(npz_files):
        parts = npz.stem.split("_", 1)
        if len(parts) != 2: continue
        sid, state = parts
        if sid not in electrodes: continue
        tasks.append((sid, state, str(npz), electrodes[sid]))

    log(f"Dispatching {len(tasks)} subject×state tasks "
        f"(grey-matter cortical reference)...")

    n_jobs = min(4, cpu_count())
    all_rows = []
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            for rows in pool.imap_unordered(_worker, tasks):
                all_rows.extend(rows)
    else:
        for task in tasks:
            all_rows.extend(_worker(task))

    if not all_rows:
        log("ERROR: no results produced — check .npz paths and ground truth")
        sys.exit(1)

    df_gm = pd.DataFrame(all_rows)
    df_gm.to_csv(OUT_DIR / "connectivity_gm.csv", index=False)
    log(f"Connectivity (GM): {len(df_gm)} rows saved")

    # ── PCA ───────────────────────────────────────────────────────────────
    log("Running PCA...")
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        feat_cols = [c for c in df_gm.columns if c.startswith("icoh_")]
        agg = (df_gm.groupby(["subject_id","state","nucleus"])[feat_cols]
               .mean().reset_index())
        parts = []
        for sid, grp in agg.groupby("subject_id"):
            X = grp[feat_cols].values
            if len(X) < 2: continue
            g2 = grp.copy()
            g2[feat_cols] = StandardScaler().fit_transform(X)
            parts.append(g2)
        if parts:
            scaled = pd.concat(parts, ignore_index=True)
            X_all  = scaled[feat_cols].fillna(0).values
            pca    = PCA(n_components=min(5, X_all.shape[1], X_all.shape[0]-1),
                         random_state=42)
            coords = pca.fit_transform(X_all)
            scaled["PC1"] = coords[:,0]
            scaled["PC2"] = coords[:,1] if coords.shape[1] > 1 else 0.0
            var = pca.explained_variance_ratio_
            log(f"PCA (GM): PC1={var[0]:.1%}  PC2={var[1]:.1%}")
            centroids = scaled.groupby("state")[["PC1","PC2"]].mean()
            log("State centroids on PC1 (GM):")
            for s, r in centroids.sort_values("PC1").iterrows():
                log(f"  {s:<20} PC1={r['PC1']:+.3f}")
            scaled.to_csv(OUT_DIR / "manifold_gm.csv", index=False)
    except ImportError:
        log("scikit-learn not installed — skipping PCA")

    # ── Statistics ────────────────────────────────────────────────────────
    log("Running statistics (GM)...")
    try:
        from scipy import stats as ss
        from statsmodels.stats.multitest import multipletests
        from itertools import combinations

        metrics  = [c for c in df_gm.columns if c.startswith("icoh_")]
        agg_stat = (df_gm[df_gm["nucleus"]=="all_thalamus"]
                    .groupby(["subject_id","state"])[metrics].mean().reset_index())
        states   = sorted(agg_stat["state"].unique())
        rows_out = []
        for s1, s2 in combinations(states, 2):
            d1 = agg_stat[agg_stat["state"]==s1].set_index("subject_id")
            d2 = agg_stat[agg_stat["state"]==s2].set_index("subject_id")
            for metric in metrics:
                v1 = d1[metric].dropna(); v2 = d2[metric].dropna()
                shared = v1.index.intersection(v2.index)
                if len(shared) >= 5:
                    stat, p = ss.wilcoxon(v1.loc[shared], v2.loc[shared])
                    a1, a2 = v1.loc[shared], v2.loc[shared]
                elif len(v1) >= 3 and len(v2) >= 3:
                    stat, p = ss.mannwhitneyu(v1, v2, alternative="two-sided")
                    a1, a2 = v1, v2
                else: continue
                pool_sd = np.sqrt((a1.std()**2 + a2.std()**2)/2)
                d = (a1.mean()-a2.mean())/(pool_sd+1e-10)
                rows_out.append({"state1":s1,"state2":s2,"metric":metric,
                                 "mean1":a1.mean(),"mean2":a2.mean(),
                                 "cohens_d":d,"p_raw":p})
        if rows_out:
            stat_df = pd.DataFrame(rows_out)
            _, q, _, _ = multipletests(stat_df["p_raw"].values, method="fdr_bh")
            stat_df["q_fdr"] = q
            stat_df["significant"] = q < 0.05
            stat_df.to_csv(OUT_DIR / "statistics_gm.csv", index=False)
            log(f"Statistics (GM): {stat_df['significant'].sum()} significant (FDR q<0.05)")
    except ImportError:
        log("scipy/statsmodels not installed — skipping statistics")

    # ── Comparison report ─────────────────────────────────────────────────
    log("\nGenerating comparison report...")
    report_lines = []
    report_lines.append("SENSITIVITY ANALYSIS — GREY MATTER ONLY vs PRIMARY")
    report_lines.append("="*65)
    report_lines.append(f"Primary:     all non-thalamic contacts")
    report_lines.append(f"Sensitivity: grey matter only (DKT ≥50%)")
    report_lines.append("")

    if PRIMARY_CSV.exists():
        df_primary = pd.read_csv(PRIMARY_CSV)
        for nuc in PRIMARY_NUCLEI + ["all_thalamus"]:
            report_lines.append(f"\n{nuc} — broadband iCoh by state:")
            report_lines.append(f"  {'State':<18} {'Primary':>10} {'GM-only':>10} {'Diff':>8}")
            report_lines.append(f"  {'-'*50}")
            p_sub = df_primary[df_primary["nucleus"]==nuc].groupby("state")["icoh_broadband"].mean()
            g_sub = df_gm[df_gm["nucleus"]==nuc].groupby("state")["icoh_broadband"].mean()
            all_states = sorted(set(p_sub.index) | set(g_sub.index))
            for s in all_states:
                pv = p_sub.get(s, float("nan"))
                gv = g_sub.get(s, float("nan"))
                diff = gv - pv if not (np.isnan(pv) or np.isnan(gv)) else float("nan")
                report_lines.append(
                    f"  {s:<18} {pv:>10.4f} {gv:>10.4f} {diff:>+8.4f}")
            if nuc in ["CM","Pul","AV"]:
                pp = p_sub.idxmax() if len(p_sub) else "N/A"
                gp = g_sub.idxmax() if len(g_sub) else "N/A"
                report_lines.append(f"  Peak state → Primary: {pp}  |  GM-only: {gp}")
    else:
        report_lines.append("Primary connectivity CSV not found — comparison skipped.")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    (OUT_DIR / "comparison_report.txt").write_text(report_text)

    total = (time.time()-t0)/60
    log(f"\nSensitivity analysis complete in {total:.1f} min")
    log(f"Results → {OUT_DIR}")

if __name__ == "__main__":
    main()
