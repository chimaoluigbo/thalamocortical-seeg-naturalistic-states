"""
build_thomas_ground_truth.py
══════════════════════════════════════════════════════════════════════════════
Rebuilds thalamic_nuclei_ground_truth.csv using THOMAS MNI atlas assignments.

Reference:
  Saranathan M, Iglehart C, Monti M, Tourdias T, Rutt B.
  In vivo high-resolution structural MRI-based atlas of human thalamic nuclei.
  Sci Data. 2021;8:275. https://doi.org/10.1038/s41597-021-01062-y

CONTEXT
──────────────────────────────────────────────────────────────────────────────
The original ground truth (112 contacts) used neurosurgeon-verified manual
labelling for all contacts. The THOMAS update:
  • Keeps neurosurgeon labels for 4 patients (10IG, 18EG, 2AE, 8RV) whose
    MNI coordinate transformation errors were manually corrected.
  • Applies THOMAS atlas labels for the remaining 23 patients.
  • Expands the contact count from 112 → 161 because THOMAS labels ALL
    contacts whose MNI coordinates fall within thalamic nuclei, not just
    the electrode tip contacts.
  • Expands nucleus categories from {CM, AN, Pulvinar} to 8 THOMAS nuclei.

NUCLEUS LABEL MAPPING (old → THOMAS)
  CM       → CM     (centromedian, unchanged)
  AN       → AV     (anterior nucleus → anterior ventral)
  Pulvinar → Pul    (same structure, THOMAS abbreviation)
  NEW:  MD (mediodorsal), MGN (medial geniculate), VA (ventroanterior),
        VLP (ventrolateral posterior), VPL (ventroposterolateral)

USAGE
──────────────────────────────────────────────────────────────────────────────
  source /home/chima/seeg_env/bin/activate
  cd /mnt/c/Users/chima/seeg_study/

  # Step A — Generate coordinate scaffold for THOMAS atlas lookup:
  python build_thomas_ground_truth.py --mode scaffold

  # Step B — After running THOMAS on the scaffold coordinates:
  python build_thomas_ground_truth.py --mode build \\
      --thomas_labels outputs/results/thomas_per_contact_labels.csv

  # Step C (optional QC only) — Approximate from per-patient TSV summary:
  python build_thomas_ground_truth.py --mode tsv_summary

EXPECTED FILES
  thalamic_nuclei_ground_truth.csv          (existing, in study root)
  per_patient_nucleus_summary_all27.tsv     (THOMAS per-patient counts)
  thalamic_contact_labelling_summary.json   (labelling source metadata)

OUTPUTS
  outputs/results/thalamic_nuclei_ground_truth_THOMAS.csv
  outputs/results/thalamic_nuclei_ground_truth_THOMAS_diff.csv
  outputs/results/thomas_mni_lookup_scaffold.csv  (Mode scaffold only)
"""

import argparse
import os
import sys
import json
import pandas as pd
import numpy as np


# ── Paths ─────────────────────────────────────────────────────────────────────
GT_PATH      = "thalamic_nuclei_ground_truth.csv"
TSV_PATH     = "per_patient_nucleus_summary_all27.tsv"
JSON_PATH    = "thalamic_contact_labelling_summary.json"
OUT_DIR      = "outputs/results"
OUT_GT       = os.path.join(OUT_DIR, "thalamic_nuclei_ground_truth_THOMAS.csv")
OUT_DIFF     = os.path.join(OUT_DIR, "thalamic_nuclei_ground_truth_THOMAS_diff.csv")
OUT_SCAFFOLD = os.path.join(OUT_DIR, "thomas_mni_lookup_scaffold.csv")

# ── Constants ─────────────────────────────────────────────────────────────────
NEUROSURGEON_PATIENTS = {"10IG", "18EG", "2AE", "8RV"}

OLD_TO_THOMAS = {
    "AN":       "AV",
    "Pulvinar": "Pul",
    "CM":       "CM",
}

NUCLEUS_FUNCTIONAL_GROUP = {
    "AV":  "anterior",
    "CM":  "intralaminar",
    "MD":  "mediodorsal",
    "MGN": "sensory_relay",
    "Pul": "posterior",
    "VA":  "motor_relay",
    "VLP": "motor_relay",
    "VPL": "sensory_relay",
}

PRIMARY_NUCLEI   = {"CM", "Pul", "AV"}
COMPOSITE_NUCLEI = {"VLP", "MD", "VA", "VPL", "MGN"}
THOMAS_NUCLEI    = ["AV", "CM", "MD", "MGN", "Pul", "VA", "VLP", "VPL"]

# Approximate MNI centroids (bilateral means, Saranathan et al. 2021 Table 1)
# Right hemisphere values; left hemisphere = sign-flipped X.
THOMAS_CENTROIDS_R = {
    "AV":  ( 7.5,  -5.5,  9.5),
    "CM":  (10.5, -20.5,  4.5),
    "MD":  ( 7.0, -14.0,  9.0),
    "MGN": (22.0, -30.0,  0.0),
    "Pul": (17.0, -28.0,  4.0),
    "VA":  (11.5,  -8.0,  8.0),
    "VLP": (18.0, -20.0,  8.0),
    "VPL": (22.0, -25.0,  6.0),
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def remap_old_nucleus(nuc):
    """Map old nucleus name to THOMAS canonical label."""
    if pd.isna(nuc) or str(nuc).strip() in ("", "none"):
        return None
    return OLD_TO_THOMAS.get(str(nuc).strip(), str(nuc).strip())


def analysis_group(nuc):
    if nuc in PRIMARY_NUCLEI:
        return nuc
    elif nuc in COMPOSITE_NUCLEI:
        return "all_thalamus"
    elif nuc:
        return "unclassified"
    return "cortex"


def centroid(nuc, hemi):
    """Return MNI centroid for a nucleus in a given hemisphere."""
    rx, ry, rz = THOMAS_CENTROIDS_R[nuc]
    if str(hemi).upper().startswith("L"):
        return (-rx, ry, rz)
    return (rx, ry, rz)


def mni_dist(x, y, z, nuc, hemi):
    cx, cy, cz = centroid(nuc, hemi)
    return np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)


def load_inputs(gt_path, tsv_path, json_path):
    for p in [gt_path]:
        if not os.path.exists(p):
            sys.exit(f"\nERROR: {p} not found.\nRun from: /mnt/c/Users/chima/seeg_study/\n")

    df = pd.read_csv(gt_path)
    print(f"Ground truth loaded:  {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"Existing thalamic contacts: {df['is_thalamic'].sum()}")

    tsv, jdata = None, None
    if os.path.exists(tsv_path):
        tsv = pd.read_csv(tsv_path, sep="\t")
        print(f"THOMAS TSV loaded:    {len(tsv)} patients, "
              f"{int(tsv['TOTAL'].sum())} total THOMAS contacts")
    if os.path.exists(json_path):
        with open(json_path) as f:
            jdata = json.load(f)

    return df, tsv, jdata


# ── MODE 1: Scaffold ──────────────────────────────────────────────────────────

def mode_scaffold(df):
    """
    Write the MNI coordinate scaffold for THOMAS atlas per-contact lookup.

    Every row in the existing ground truth is included with its MNI coords.
    Feed the x_mni / y_mni / z_mni columns through the THOMAS pipeline to
    get per-contact nucleus assignments, then pass the result to --mode build.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    sc = df[["subject_id", "contact_name", "electrode_shaft",
             "x_mni", "y_mni", "z_mni", "hemisphere",
             "is_thalamic", "thalamic_nucleus", "atlas_source"]].copy()

    sc["needs_thomas"] = ~sc["subject_id"].isin(NEUROSURGEON_PATIENTS)
    sc["old_nucleus_thomas_name"] = sc["thalamic_nucleus"].apply(remap_old_nucleus)
    sc["thomas_nucleus"] = None        # to be filled by THOMAS pipeline
    sc["thomas_distance_mm"] = None    # optional QC column

    sc.to_csv(OUT_SCAFFOLD, index=False)

    neuro_n  = sc["subject_id"].isin(NEUROSURGEON_PATIENTS).sum()
    thomas_n = sc["needs_thomas"].sum()
    print(f"\nScaffold written: {OUT_SCAFFOLD}")
    print(f"  Total contacts:            {len(sc)}")
    print(f"  Neurosurgeon-verified (retain label): {neuro_n}")
    print(f"  Need THOMAS lookup:        {thomas_n}")
    print(f"\nNEXT STEPS:")
    print(f"  1. Run THOMAS atlas segmentation on x_mni/y_mni/z_mni for rows")
    print(f"     where needs_thomas = True")
    print(f"  2. Fill in 'thomas_nucleus' column (use None/'' for non-thalamic)")
    print(f"  3. Save result as: outputs/results/thomas_per_contact_labels.csv")
    print(f"     Required columns: subject_id, contact_name, thomas_nucleus")
    print(f"  4. Run: python build_thomas_ground_truth.py --mode build \\")
    print(f"            --thomas_labels outputs/results/thomas_per_contact_labels.csv")


# ── MODE 2: Build from per-contact THOMAS labels ──────────────────────────────

def mode_build(df, thomas_labels_path):
    """
    Authoritative mode. Applies per-contact THOMAS segmentation output.
    """
    if not os.path.exists(thomas_labels_path):
        sys.exit(f"\nERROR: {thomas_labels_path} not found.")

    tl = pd.read_csv(thomas_labels_path)
    needed = {"subject_id", "contact_name", "thomas_nucleus"}
    missing = needed - set(tl.columns)
    if missing:
        sys.exit(f"\nERROR: thomas_labels file missing columns: {missing}")

    thomas_map = (tl.set_index(["subject_id", "contact_name"])["thomas_nucleus"]
                    .where(tl.set_index(["subject_id","contact_name"])["thomas_nucleus"].notna())
                    .to_dict())

    print(f"THOMAS per-contact labels loaded: {len(tl)} rows, "
          f"{tl['thomas_nucleus'].notna().sum()} non-null")

    os.makedirs(OUT_DIR, exist_ok=True)
    df_new = df.copy()

    for idx, row in df_new.iterrows():
        pid   = row["subject_id"]
        cname = row["contact_name"]

        if pid in NEUROSURGEON_PATIENTS:
            # Retain label; update to THOMAS naming convention
            if pd.notna(row["thalamic_nucleus"]) and row["thalamic_nucleus"] not in ("", None):
                tn = remap_old_nucleus(row["thalamic_nucleus"])
                df_new.at[idx, "thalamic_nucleus"]  = tn
                df_new.at[idx, "nucleus_gt"]         = tn
                df_new.at[idx, "atlas_source"]       = "neurosurgeon_verified"
                df_new.at[idx, "functional_group"]   = NUCLEUS_FUNCTIONAL_GROUP.get(tn, "none")
            continue

        t_nuc = thomas_map.get((pid, cname), None)
        if t_nuc and str(t_nuc).strip() not in ("", "none", "cortex", "nan"):
            t_nuc = str(t_nuc).strip()
            df_new.at[idx, "is_thalamic"]       = True
            df_new.at[idx, "thalamic_nucleus"]  = t_nuc
            df_new.at[idx, "nucleus_gt"]         = t_nuc
            df_new.at[idx, "functional_group"]  = NUCLEUS_FUNCTIONAL_GROUP.get(t_nuc, "none")
            df_new.at[idx, "atlas_source"]       = "THOMAS_atlas"
        else:
            df_new.at[idx, "is_thalamic"]       = False
            df_new.at[idx, "thalamic_nucleus"]  = None
            df_new.at[idx, "nucleus_gt"]         = None
            df_new.at[idx, "functional_group"]  = "none"
            df_new.at[idx, "atlas_source"]       = "THOMAS_atlas"

    df_new["analysis_group"] = df_new["thalamic_nucleus"].apply(
        lambda x: analysis_group(x) if pd.notna(x) else "cortex"
    )
    _save_and_summarise(df, df_new)


# ── MODE 3: TSV summary (approximate, QC only) ────────────────────────────────

def mode_tsv_summary(df, tsv):
    """
    Heuristic approximation. Uses per-patient nucleus counts from the THOMAS
    TSV and assigns contacts to nuclei by nearest MNI centroid within quota.

    This is appropriate for QC and pipeline testing. For the final manuscript
    use mode_build() with full per-contact THOMAS segmentation.
    """
    print("\n⚠  TSV-SUMMARY MODE: Centroid-proximity heuristic — not full THOMAS segmentation.")
    print("   Use for pipeline QC only. Run --mode build for the final manuscript.\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    df_new = df.copy()

    is_neuro = df_new["subject_id"].isin(NEUROSURGEON_PATIENTS)

    # Neurosurgeon patients: remap label names only
    for idx, row in df_new[is_neuro].iterrows():
        nuc = row.get("thalamic_nucleus", None)
        if pd.notna(nuc) and str(nuc).strip() not in ("", "none"):
            tn = remap_old_nucleus(nuc)
            df_new.at[idx, "thalamic_nucleus"]  = tn
            df_new.at[idx, "nucleus_gt"]         = tn
            df_new.at[idx, "atlas_source"]       = "neurosurgeon_verified"
            df_new.at[idx, "functional_group"]   = NUCLEUS_FUNCTIONAL_GROUP.get(tn, "none")

    # THOMAS patients: reset then reassign
    df_new.loc[~is_neuro, ["is_thalamic", "thalamic_nucleus",
                            "nucleus_gt", "functional_group"]] = [False, None, None, "none"]
    df_new.loc[~is_neuro, "atlas_source"] = "THOMAS_atlas_approx"

    tsv_idx = tsv.set_index("patient").to_dict("index")

    for pid, grp in df_new[~is_neuro].groupby("subject_id"):
        if pid not in tsv_idx:
            print(f"  SKIP: {pid} not in TSV")
            continue

        quota = {n: int(tsv_idx[pid].get(n, 0)) for n in THOMAS_NUCLEI}
        if sum(quota.values()) == 0:
            continue

        # Score every contact against every nucleus with non-zero quota
        scores = []
        for idx, row in grp.iterrows():
            for nuc in THOMAS_NUCLEI:
                if quota[nuc] == 0:
                    continue
                d = mni_dist(row["x_mni"], row["y_mni"], row["z_mni"],
                             nuc, row["hemisphere"])
                scores.append((d, idx, nuc))
        scores.sort()

        assigned_contacts = set()
        remaining = quota.copy()

        for d, idx, nuc in scores:
            if idx in assigned_contacts or remaining[nuc] <= 0:
                continue
            if d > 20.0:  # reject implausible assignments
                continue
            df_new.at[idx, "is_thalamic"]      = True
            df_new.at[idx, "thalamic_nucleus"]  = nuc
            df_new.at[idx, "nucleus_gt"]         = nuc
            df_new.at[idx, "functional_group"]  = NUCLEUS_FUNCTIONAL_GROUP.get(nuc, "none")
            assigned_contacts.add(idx)
            remaining[nuc] -= 1

        # Warn on unfilled quota
        shortfall = {n: v for n, v in remaining.items() if v > 0}
        if shortfall:
            print(f"  ⚠  {pid}: unassigned quota {shortfall} "
                  f"(no contacts within 20mm of centroid)")

    df_new["analysis_group"] = df_new["thalamic_nucleus"].apply(
        lambda x: analysis_group(x) if pd.notna(x) else "cortex"
    )
    _save_and_summarise(df, df_new)


# ── Shared save + summary ─────────────────────────────────────────────────────

def _save_and_summarise(df_old, df_new):
    thal_new = df_new[df_new["is_thalamic"] == True]
    thal_old = df_old[df_old["is_thalamic"] == True]

    # Validate against THOMAS TSV if available
    tsv = pd.read_csv(TSV_PATH, sep="\t") if os.path.exists(TSV_PATH) else None

    print(f"\n{'═'*62}")
    print("UPDATED GROUND TRUTH — FINAL COUNTS")
    print(f"{'═'*62}")
    print(f"  Total rows (all contacts):    {len(df_new)}")
    print(f"  Thalamic contacts (old):      {len(thal_old)}")
    print(f"  Thalamic contacts (new):      {len(thal_new)}")
    print(f"\n  Per-nucleus breakdown:")
    print(f"  {'Nucleus':<8} {'Contacts':>9} {'Subjects':>9} {'Group':<16} {'THOMAS TSV':>10}")
    print(f"  {'-'*56}")

    for nuc in sorted(THOMAS_NUCLEI):
        subset = df_new[(df_new["is_thalamic"]==True) & (df_new["thalamic_nucleus"]==nuc)]
        n_contacts = len(subset)
        n_subjects = subset["subject_id"].nunique()
        grp = NUCLEUS_FUNCTIONAL_GROUP.get(nuc, "—")
        tsv_count = ""
        if tsv is not None:
            tsv_count = str(int(tsv[nuc].sum()))
        match = "✓" if tsv_count and str(n_contacts) == tsv_count else "⚠" if tsv_count else ""
        print(f"  {nuc:<8} {n_contacts:>9} {n_subjects:>9} {grp:<16} {tsv_count:>8} {match}")

    neuro_contacts = df_new[
        (df_new["is_thalamic"]==True) & (df_new["atlas_source"]=="neurosurgeon_verified")
    ]
    thomas_contacts = df_new[
        (df_new["is_thalamic"]==True) & (df_new["atlas_source"].str.contains("THOMAS", na=False))
    ]
    print(f"\n  Labelling source:")
    print(f"    neurosurgeon_verified:  {len(neuro_contacts)} contacts")
    print(f"    THOMAS_atlas:           {len(thomas_contacts)} contacts")
    print(f"    (4 patients neurosurgeon; 23 patients THOMAS)")

    # Diff
    diff = df_old[["subject_id","contact_name","thalamic_nucleus","is_thalamic"]].merge(
        df_new[["subject_id","contact_name","thalamic_nucleus","is_thalamic","atlas_source"]],
        on=["subject_id","contact_name"], suffixes=("_old","_new"), how="outer"
    )
    changed = diff[
        (diff["thalamic_nucleus_old"].fillna("") != diff["thalamic_nucleus_new"].fillna("")) |
        (diff["is_thalamic_old"].fillna(False)   != diff["is_thalamic_new"].fillna(False))
    ]
    changed.to_csv(OUT_DIFF, index=False)
    df_new.to_csv(OUT_GT, index=False)

    print(f"\n  Contacts with changed assignments: {len(changed)}")
    print(f"\n  Files written:")
    print(f"    {OUT_GT}")
    print(f"    {OUT_DIFF}")
    print(f"\n{'═'*62}")
    print("PIPELINE NEXT STEPS")
    print(f"{'═'*62}")
    print("  1. Copy new ground truth to study root:")
    print(f"       cp {OUT_GT} thalamic_nuclei_ground_truth.csv")
    print("  2. Re-run Step 5 (connectivity):")
    print("       python step5_connectivity.py")
    print("  3. Re-run Step 6 (statistics):")
    print("       python step6_statistics.py")
    print("  4. Check that ✓ marks appear in all nucleus rows above.")
    print(f"{'═'*62}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Build THOMAS-updated thalamic nuclei ground truth CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=["scaffold", "build", "tsv_summary"],
                   default="scaffold",
                   help="scaffold → generate MNI lookup table for THOMAS;\n"
                        "build    → apply per-contact THOMAS results;\n"
                        "tsv_summary → approximate from per-patient counts (QC only)")
    p.add_argument("--gt",            default=GT_PATH)
    p.add_argument("--tsv",           default=TSV_PATH)
    p.add_argument("--json",          default=JSON_PATH)
    p.add_argument("--thomas_labels", default=None,
                   help="(--mode build) CSV with subject_id, contact_name, thomas_nucleus")
    args = p.parse_args()

    df, tsv, jdata = load_inputs(args.gt, args.tsv, args.json)

    if args.mode == "scaffold":
        mode_scaffold(df)
    elif args.mode == "build":
        if not args.thomas_labels:
            p.error("--thomas_labels required when --mode build")
        mode_build(df, args.thomas_labels)
    elif args.mode == "tsv_summary":
        if tsv is None:
            p.error(f"THOMAS TSV not found at {args.tsv}")
        mode_tsv_summary(df, tsv)


if __name__ == "__main__":
    main()
