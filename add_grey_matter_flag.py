#!/usr/bin/env python3
"""
add_grey_matter_flag.py
Merges grey matter classification into thalamic_nuclei_ground_truth.csv.

USAGE:
  cd /mnt/c/Users/chima/seeg_study/
  python add_grey_matter_flag.py

New columns added:
  tissue_class   : cortical_grey | white_matter | subcortical | outside_brain | thalamus | unknown
  dkt_region     : DKT atlas label string
  dkt_prob       : integer probability 0-100
  is_grey_matter : True if cortical_grey AND dkt_prob >= 50
"""

import argparse, os, sys
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt",        default="thalamic_nuclei_ground_truth.csv")
    p.add_argument("--gm",        default="grey_matter_contacts_all27.csv")
    p.add_argument("--out",       default="thalamic_nuclei_ground_truth_gm.csv")
    p.add_argument("--threshold", type=int, default=50)
    return p.parse_args()

def main():
    args = parse_args()

    for f in [args.gt, args.gm]:
        if not os.path.exists(f):
            print(f"ERROR: not found: {f}"); sys.exit(1)

    print(f"Loading: {args.gt}")
    gt = pd.read_csv(args.gt)
    print(f"  {len(gt)} rows")

    print(f"Loading: {args.gm}")
    gm = pd.read_csv(args.gm)
    print(f"  {len(gm)} grey matter contacts")

    # Normalise join keys
    gt['_s'] = gt['subject_id'].astype(str).str.strip().str.upper()
    gt['_c'] = gt['contact_name'].astype(str).str.strip().str.upper()
    gm['_s'] = gm['subject_id'].astype(str).str.strip().str.upper()
    gm['_c'] = gm['contact_name'].astype(str).str.strip().str.upper()

    gm_slim = gm[['_s','_c','tissue_class','dkt_region','dkt_prob']].copy()

    merged = gt.merge(gm_slim, on=['_s','_c'], how='left')

    # Thalamic contacts not in GM list -> label thalamus
    thal = merged['is_thalamic'] == True
    merged.loc[thal & merged['tissue_class'].isna(), 'tissue_class'] = 'thalamus'

    # Non-thalamic not matched -> unknown (white matter / subcortical / outside)
    unmatched = (~thal) & merged['tissue_class'].isna()
    merged.loc[unmatched, 'tissue_class'] = 'unknown'

    merged['dkt_region'] = merged['dkt_region'].fillna('')
    merged['dkt_prob']   = merged['dkt_prob'].fillna(0).astype(int)

    # Grey matter flag: cortical grey AND >= threshold probability
    merged['is_grey_matter'] = (
        (merged['tissue_class'] == 'cortical_grey') &
        (merged['dkt_prob'] >= args.threshold)
    )

    merged = merged.drop(columns=['_s','_c'])
    merged.to_csv(args.out, index=False)

    # ── Report ────────────────────────────────────────────────────────────
    n       = len(merged)
    n_thal  = int(merged['is_thalamic'].sum())
    n_nt    = n - n_thal
    n_gm    = int(merged['is_grey_matter'].sum())
    n_wm    = int((merged['tissue_class']=='white_matter').sum())
    n_unk   = int((merged['tissue_class']=='unknown').sum())

    print(f"\n{'='*60}")
    print("MERGE COMPLETE")
    print(f"{'='*60}")
    print(f"Total contacts:                      {n}")
    print(f"  Thalamic (atlas-assigned):         {n_thal}")
    print(f"  Non-thalamic total:                {n_nt}")
    print(f"\nTissue class breakdown:")
    for tc, cnt in merged['tissue_class'].value_counts().items():
        print(f"  {tc:<28} {cnt:>5}  ({cnt/n*100:.1f}%)")
    print(f"\nis_grey_matter == True (>={args.threshold}% DKT): {n_gm}")
    print(f"White matter:                        {n_wm}")
    print(f"Unknown / unmatched:                 {n_unk}")
    if n_unk > 0:
        print(f"\n  Unmatched by subject:")
        for sid, cnt in merged[merged['tissue_class']=='unknown']['subject_id'].value_counts().items():
            print(f"    {sid}: {cnt}")
    print(f"\n{'='*60}")
    print("SENSITIVITY ANALYSIS SCOPE")
    print(f"{'='*60}")
    print(f"  Primary:     {n_nt} non-thalamic contacts (current pipeline)")
    print(f"  Sensitivity: {n_gm} cortical grey matter contacts (>={args.threshold}% DKT)")
    print(f"  Excluded:    {n_nt-n_gm} contacts ({(n_nt-n_gm)/n_nt*100:.1f}% of non-thalamic)")
    print(f"\nOutput: {args.out}")
    print(f"\nNEXT STEP:")
    print(f"  cp {args.out} outputs/results/{args.out}")
    print(f"  python run_sensitivity_analysis.py")

if __name__ == "__main__":
    main()
