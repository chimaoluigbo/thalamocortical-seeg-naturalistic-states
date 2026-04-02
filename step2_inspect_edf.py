#!/usr/bin/env python3
"""
STEP 2 — Inspect a single EDF file
====================================
Loads ONE EDF file and reports:
  - Channel names (raw, before any renaming)
  - Sample rate
  - Duration
  - Signal amplitude ranges (to check for flat/saturated channels)
  - Which channels match the thalamic ground truth contacts

Does NOT filter. Does NOT save processed data. Does NOT modify anything.

Run:
    cd /mnt/c/Users/chima/seeg_study
    python3 step2_inspect_edf.py

To inspect a different file, change SUBJECT and STATE below.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─── Choose which file to inspect ────────────────────────────────────────────
# Change these to inspect a different subject or state

SUBJECT = "11PH"    # subject to inspect
STATE   = "eating"  # state folder to inspect (first clip will be loaded)

DATA_DIR = Path(".")
EDF_DIR  = DATA_DIR / "edf"
GT_CSV   = DATA_DIR / "outputs/results/thalamic_nuclei_ground_truth.csv"

# ─── Find the first EDF for this subject + state ─────────────────────────────

STATE_MAP = {
    "eating":     ["EATING","EAT","DRINKING"],
    "playing":    ["PLAYING","PLAY"],
    "reading":    ["READING","READ"],
    "talking":    ["TALKING","TALK"],
    "laughing":   ["SMILING","SMILE","LAUGHING"],
    "watching_tv":["WATCHING","WATCH","SCREEN"],
    "crying":     ["PAIN","UPSET","CRYING"],
    "nrem_sleep": ["NONEREM","NONREM","NREM","NON-REM"],
    "rem_sleep":  ["REM"],
}

def matches_state(folder_name, target_state):
    u = folder_name.upper()
    for kw in STATE_MAP.get(target_state, []):
        if kw in u:
            return True
    return False

sid_dir = EDF_DIR / SUBJECT
if not sid_dir.exists():
    print(f"ERROR: {sid_dir} does not exist")
    exit(1)

target_edf = None
for edf_path in sorted(sid_dir.rglob("*.edf")):
    if matches_state(edf_path.parent.name, STATE) or \
       matches_state(edf_path.parent.parent.name, STATE):
        target_edf = edf_path
        break

if target_edf is None:
    print(f"ERROR: No EDF found for subject={SUBJECT}, state={STATE}")
    exit(1)

print(f"Loading: {target_edf}")
print()

# ─── Load with MNE ────────────────────────────────────────────────────────────
import mne

raw = mne.io.read_raw_edf(str(target_edf), preload=True, verbose=False)

# ─── Basic info ───────────────────────────────────────────────────────────────
print("=" * 65)
print("BASIC INFO")
print("=" * 65)
print(f"  Subject:        {SUBJECT}")
print(f"  State:          {STATE}")
print(f"  File:           {target_edf.name}")
print(f"  Duration:       {raw.times[-1]/60:.2f} minutes ({raw.times[-1]:.0f} seconds)")
print(f"  Sample rate:    {raw.info['sfreq']} Hz")
print(f"  Total channels: {len(raw.ch_names)}")

# ─── Channel names (raw, before any renaming) ─────────────────────────────────
print()
print("=" * 65)
print("RAW CHANNEL NAMES (first 30)")
print("=" * 65)
for i, ch in enumerate(raw.ch_names[:30]):
    print(f"  {i:3d}  {repr(ch)}")
if len(raw.ch_names) > 30:
    print(f"  ... and {len(raw.ch_names)-30} more")

# Check for POL prefix
pol_channels = [ch for ch in raw.ch_names if ch.strip().upper().startswith("POL ")]
non_pol      = [ch for ch in raw.ch_names if not ch.strip().upper().startswith("POL ")]
print(f"\n  Channels with 'POL ' prefix: {len(pol_channels)}")
print(f"  Channels without prefix:     {len(non_pol)}")

# Status/off channels
status_chs = [ch for ch in raw.ch_names
              if any(kw in ch.upper()
                     for kw in ["OFF","MKR","STATUS","TRIGGER","ANNOT","EDF"])]
print(f"  Status/marker channels:      {len(status_chs)}")
if status_chs:
    print(f"  Status channel names: {status_chs}")

# ─── After stripping POL prefix: what do channel names become? ───────────────
print()
print("=" * 65)
print("CHANNEL NAMES AFTER STRIPPING 'POL ' PREFIX")
print("=" * 65)
stripped = []
for ch in raw.ch_names:
    s = ch.strip()
    if s.upper().startswith("POL "):
        stripped.append(s[4:].strip())
    else:
        stripped.append(s)
for i, ch in enumerate(stripped[:30]):
    print(f"  {i:3d}  {ch}")

# ─── Match against ground truth atlas ────────────────────────────────────────
print()
print("=" * 65)
print("MATCH AGAINST GROUND TRUTH ATLAS")
print("=" * 65)

if GT_CSV.exists():
    gt = pd.read_csv(GT_CSV)
    gt_sub = gt[gt["subject_id"] == SUBJECT]
    thal   = gt_sub[gt_sub["is_thalamic"] == True]

    print(f"  Ground truth thalamic contacts for {SUBJECT}: {len(thal)}")
    if len(thal):
        print(f"  Nuclei: {thal['thalamic_nucleus'].value_counts().to_dict()}")

    # Check which thalamic contacts appear in stripped channel names
    stripped_set = set(stripped)
    found  = [c for c in thal["contact_name"].tolist() if c in stripped_set]
    missing = [c for c in thal["contact_name"].tolist() if c not in stripped_set]

    print(f"\n  Thalamic contacts FOUND in this EDF:   {len(found)}")
    if found:
        for c in found:
            nuc = thal[thal["contact_name"]==c]["thalamic_nucleus"].values[0]
            print(f"    {c}  [{nuc}]")

    print(f"\n  Thalamic contacts MISSING from this EDF: {len(missing)}")
    if missing:
        for c in missing:
            nuc = thal[thal["contact_name"]==c]["thalamic_nucleus"].values[0]
            print(f"    {c}  [{nuc}]")
else:
    print(f"  Ground truth CSV not found at {GT_CSV}")

# ─── Signal amplitude ranges ─────────────────────────────────────────────────
print()
print("=" * 65)
print("SIGNAL AMPLITUDE CHECK (microvolts)")
print("=" * 65)
data = raw.get_data() * 1e6  # convert to µV

flat_threshold     = 0.5    # µV — peak-to-peak below this = flat
saturated_threshold = 5000  # µV — peak-to-peak above this = likely saturated

pp = np.ptp(data, axis=1)  # peak-to-peak per channel

flat_chs      = [stripped[i] for i, v in enumerate(pp) if v < flat_threshold
                 and stripped[i] not in [s for s in status_chs]]
saturated_chs = [stripped[i] for i, v in enumerate(pp) if v > saturated_threshold]

print(f"  Median peak-to-peak amplitude: {np.median(pp):.1f} µV")
print(f"  Min peak-to-peak:              {pp.min():.2f} µV")
print(f"  Max peak-to-peak:              {pp.max():.1f} µV")
print(f"\n  Flat channels (< {flat_threshold} µV p-p):        {len(flat_chs)}")
if flat_chs:
    print(f"    {flat_chs}")
print(f"  Saturated channels (> {saturated_threshold} µV p-p): {len(saturated_chs)}")
if saturated_chs:
    print(f"    {saturated_chs}")

# ─── Summary of what Step 3 will need to do ──────────────────────────────────
print()
print("=" * 65)
print("SUMMARY — what Step 3 (preprocessing) will need to do")
print("=" * 65)
print(f"  1. Strip 'POL ' prefix from channel names")
print(f"  2. Drop {len(status_chs)} status/marker channels")
if flat_chs:
    print(f"  3. Flag {len(flat_chs)} flat channels for exclusion")
print(f"  4. Resample from {raw.info['sfreq']} Hz → 256 Hz"
      if raw.info['sfreq'] != 256 else
      f"  4. Already at 256 Hz — no resampling needed")
print(f"  5. Bandpass filter 0.5–100 Hz + notch at 60 Hz")
print(f"  6. Bipolar re-reference within electrode shafts")
print()
print("Review the above and confirm before Step 3 is written.")
