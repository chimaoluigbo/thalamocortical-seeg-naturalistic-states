#!/usr/bin/env python3
"""
STEP 3 — Preprocess ONE EDF clip and verify the result
=======================================================
Processes a single EDF file and shows you what comes out.
Does NOT process all subjects. Does NOT save to processed_npz/.
Saves ONE verification file: outputs/results/step3_verification.npz

Steps performed (in order):
  1. Load EDF
  2. Drop non-SEEG channels (EKG, DC, E, status channels)
  3. Strip POL prefix from channel names
  4. Keep only channels that are valid SEEG contacts (letters+digits pattern)
  5. Resample to 256 Hz if needed
  6. Bandpass filter 0.5–100 Hz
  7. Notch filter 60 Hz + 120 Hz
  8. Bipolar re-reference within electrode shafts
  9. Report what came out

Run:
    cd /mnt/c/Users/chima/seeg_study
    python3 step3_preprocess_one_clip.py
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

SUBJECT  = "11PH"
STATE    = "eating"
DATA_DIR = Path(".")
GT_CSV   = DATA_DIR / "outputs/results/thalamic_nuclei_ground_truth.csv"
SFREQ    = 256     # target sample rate
NOTCH    = 60.0    # Hz

# ─── Non-SEEG channel drop rules ─────────────────────────────────────────────
# A channel is dropped if its STRIPPED name (after removing POL prefix) matches
# any of these rules. Confirmed with you in Step 2.

def is_non_seeg(stripped_name):
    s = stripped_name.strip().upper()
    if s == "E":                    return True   # single letter
    if s.startswith("EKG"):        return True   # cardiac
    if s.startswith("DC"):         return True   # DC input channels
    if s.startswith("MKR"):        return True   # marker
    if "OFF" in s:                 return True   # POL OFF status
    if "STATUS" in s:              return True
    if "TRIGGER" in s:             return True
    if "ANNOT" in s:               return True
    if "EDF" in s:                 return True
    return False

# A valid SEEG contact name: one or more capital letters followed by digits
# e.g. LMFC1, RIAT14, LSAM10
SEEG_PATTERN = re.compile(r'^[A-Za-z]+\d+$')

def is_valid_seeg(stripped_name):
    return bool(SEEG_PATTERN.match(stripped_name.strip()))

# ─── Find first EDF for this subject + state ─────────────────────────────────

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

def matches_state(folder, target):
    u = folder.upper()
    for kw in STATE_MAP.get(target, []):
        if kw in u: return True
    return False

sid_dir = DATA_DIR / "edf" / SUBJECT
target_edf = None
for edf in sorted(sid_dir.rglob("*.edf")):
    if matches_state(edf.parent.name, STATE) or \
       matches_state(edf.parent.parent.name, STATE):
        target_edf = edf
        break

if not target_edf:
    print(f"ERROR: No EDF for {SUBJECT} / {STATE}"); exit(1)

print(f"\nProcessing: {target_edf}")

# ─── STEP 1: Load ─────────────────────────────────────────────────────────────
import mne
raw = mne.io.read_raw_edf(str(target_edf), preload=True, verbose=False)
print(f"\n[1] Loaded:  {len(raw.ch_names)} channels, "
      f"{raw.info['sfreq']} Hz, "
      f"{raw.times[-1]/60:.2f} min")

# ─── STEP 2: Strip POL prefix and classify channels ──────────────────────────
rename_map = {}
for ch in raw.ch_names:
    s = ch.strip()
    if s.upper().startswith("POL "):
        rename_map[ch] = s[4:].strip()
    else:
        rename_map[ch] = s
raw.rename_channels(rename_map)

# Identify channels to drop
drop_non_seeg  = [ch for ch in raw.ch_names if is_non_seeg(ch)]
drop_not_valid = [ch for ch in raw.ch_names
                  if ch not in drop_non_seeg and not is_valid_seeg(ch)]

print(f"\n[2] Channel classification after stripping POL prefix:")
print(f"    Total channels:         {len(raw.ch_names)}")
print(f"    Non-SEEG (EKG/DC/E):   {len(drop_non_seeg)}  → {drop_non_seeg}")
print(f"    Invalid pattern:        {len(drop_not_valid)}  → {drop_not_valid}")

# ─── STEP 3: Drop non-SEEG channels ──────────────────────────────────────────
to_drop = drop_non_seeg + drop_not_valid
if to_drop:
    raw.drop_channels(to_drop)
print(f"\n[3] After dropping:  {len(raw.ch_names)} SEEG channels remain")

# ─── STEP 4: Resample ─────────────────────────────────────────────────────────
original_sfreq = raw.info["sfreq"]
if original_sfreq != SFREQ:
    print(f"\n[4] Resampling {original_sfreq} Hz → {SFREQ} Hz ...")
    raw.resample(SFREQ, npad="auto", verbose=False)
    print(f"    Done. New duration: {raw.times[-1]/60:.2f} min")
else:
    print(f"\n[4] Already at {SFREQ} Hz — no resampling needed")

# ─── STEP 5: Filter ───────────────────────────────────────────────────────────
print(f"\n[5] Bandpass filtering 0.5–100 Hz ...")
raw.filter(l_freq=0.5, h_freq=100.0,
           method="fir", fir_design="firwin", verbose=False)
print(f"    Notch filtering {NOTCH} Hz + {NOTCH*2} Hz ...")
raw.notch_filter(freqs=[NOTCH, NOTCH*2], verbose=False)
print(f"    Done.")

# ─── STEP 6: Bipolar re-reference within shafts ───────────────────────────────
print(f"\n[6] Bipolar re-referencing within electrode shafts ...")

# Load ground truth to get shaft assignments
if GT_CSV.exists():
    gt = pd.read_csv(GT_CSV)
    gt_sub = gt[gt["subject_id"] == SUBJECT].copy()
    gt_sub["contact_num"] = gt_sub["contact_name"].apply(
        lambda x: int(m.group()) if (m := re.search(r'\d+$', str(x))) else 0)
else:
    print("    WARNING: Ground truth CSV not found — using auto-detected shafts")
    gt_sub = None

# Build bipolar pairs
current_chs = set(raw.ch_names)
anodes, cathodes, bip_names = [], [], []

if gt_sub is not None:
    for shaft, grp in gt_sub.groupby("electrode_shaft"):
        shaft_chs = (grp.sort_values("contact_num")["contact_name"].tolist())
        for i in range(len(shaft_chs) - 1):
            a, c = shaft_chs[i], shaft_chs[i+1]
            if a in current_chs and c in current_chs:
                anodes.append(a)
                cathodes.append(c)
                bip_names.append(f"{a}-{c}")
else:
    # Auto-detect shafts from channel names
    shaft_map = {}
    for ch in raw.ch_names:
        m = re.match(r'^([A-Za-z]+)(\d+)$', ch)
        if m:
            shaft = m.group(1)
            shaft_map.setdefault(shaft, []).append((int(m.group(2)), ch))
    for shaft, contacts in shaft_map.items():
        contacts.sort()
        for i in range(len(contacts)-1):
            a = contacts[i][1]; c = contacts[i+1][1]
            anodes.append(a); cathodes.append(c)
            bip_names.append(f"{a}-{c}")

if anodes:
    raw = mne.set_bipolar_reference(
        raw, anode=anodes, cathode=cathodes,
        ch_name=bip_names, verbose=False)
    print(f"    Created {len(bip_names)} bipolar channels")
else:
    print("    WARNING: No bipolar pairs created")

# ─── STEP 7: Verify output ───────────────────────────────────────────────────
print(f"\n[7] VERIFICATION")
print(f"    Final channels:  {len(raw.ch_names)}")
print(f"    Duration:        {raw.times[-1]/60:.2f} min")
print(f"    Sample rate:     {raw.info['sfreq']} Hz")
print(f"    Total samples:   {raw.n_times}")
print(f"    4-sec epochs:    {int(raw.n_times / (SFREQ * 4))}")

# Check thalamic bipolar channels
if GT_CSV.exists():
    gt_sub2 = gt[gt["subject_id"]==SUBJECT]
    thal = gt_sub2[gt_sub2["is_thalamic"]==True]["contact_name"].tolist()
    thal_bip = [ch for ch in raw.ch_names if ch.split("-")[0] in thal]
    print(f"\n    Thalamic bipolar channels: {len(thal_bip)}")
    for ch in thal_bip:
        anode = ch.split("-")[0]
        row   = gt_sub2[gt_sub2["contact_name"]==anode]
        nuc   = row["thalamic_nucleus"].values[0] if len(row) else "?"
        print(f"      {ch}  [{nuc}]")

# Amplitude check on bipolar signal
data = raw.get_data() * 1e6
pp   = np.ptp(data, axis=1)
print(f"\n    Bipolar amplitude (µV peak-to-peak):")
print(f"      Median: {np.median(pp):.1f}")
print(f"      Min:    {pp.min():.1f}")
print(f"      Max:    {pp.max():.1f}")

flat = [raw.ch_names[i] for i,v in enumerate(pp) if v < 0.5]
print(f"      Flat channels (< 0.5 µV): {len(flat)}")
if flat: print(f"        {flat}")

# ─── Save verification file ──────────────────────────────────────────────────
out = Path("outputs/results/step3_verification.npz")
out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    str(out),
    data     = raw.get_data().astype(np.float32),
    ch_names = raw.ch_names,
    sfreq    = raw.info["sfreq"],
    sid      = SUBJECT,
    state    = STATE,
)
print(f"\n✓ Saved verification file: {out}")
print(f"\nReview the output above and confirm:")
print(f"  1. Correct channels dropped (EKG, DC, E)?")
print(f"  2. Thalamic bipolar channels look correct?")
print(f"  3. Amplitude range looks physiologically plausible (50-500 µV)?")
print(f"  4. Number of 4-sec epochs looks reasonable for the clip duration?")
print(f"  5. No unexpected flat channels?")
print(f"\nOnly approve Step 4 (full preprocessing) after confirming all 5.")
