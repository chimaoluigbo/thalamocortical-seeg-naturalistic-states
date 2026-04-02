#!/usr/bin/env python3
"""
STEP 1 — EDF Inventory
======================
Walks the edf/ folder and reports what recordings exist.
Does NOT load signal data. Does NOT process anything. Does NOT save anything.

Run:
    cd /mnt/c/Users/chima/seeg_study
    python3 step1_edf_inventory.py

Output:
    - Prints a table: subject | state | filename | duration_min | sfreq | n_channels
    - Prints a summary: subjects × states matrix with event counts
    - Saves: outputs/results/edf_inventory.csv
"""

import os
from pathlib import Path
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────

DATA_DIR = Path(".")   # run from /mnt/c/Users/chima/seeg_study
EDF_DIR  = DATA_DIR / "edf"

# Folder name → canonical state label
# Add or correct any mappings here before approving Step 2
STATE_FOLDER_MAP = {
    "EATING":    "eating",
    "DRINKING":  "eating",
    "EAT":       "eating",
    "PLAYING":   "playing",
    "PLAY":      "playing",
    "READING":   "reading",
    "READ":      "reading",
    "TALKING":   "talking",
    "TALK":      "talking",
    "SMILING":   "laughing",
    "SMILE":     "laughing",
    "LAUGHING":  "laughing",
    "WATCHING":  "watching_tv",
    "WATCH":     "watching_tv",
    "SCREEN":    "watching_tv",
    "PAIN":      "crying",
    "UPSET":     "crying",
    "CRYING":    "crying",
    "NONEREM":   "nrem_sleep",
    "NON-REM":   "nrem_sleep",
    "NONREM":    "nrem_sleep",
    "NREM":      "nrem_sleep",
    "REM":       "rem_sleep",
}

def folder_to_state(folder_name):
    """Map a folder name to a canonical state label. Returns None if unrecognised."""
    upper = folder_name.upper()
    # Check sleep states first (longer keywords, avoid partial matches)
    for kw in ["NONEREM", "NON-REM", "NONREM", "NREM", "REM"]:
        if kw in upper:
            return STATE_FOLDER_MAP[kw]
    for kw, label in STATE_FOLDER_MAP.items():
        if kw in upper:
            return label
    return None

# ─── EDF header reader (no signal data loaded) ───────────────────────────────

def read_edf_header(path):
    """
    Read EDF header bytes only — returns (duration_sec, sfreq, n_channels).
    Does not load any signal samples.
    """
    import struct

    with open(path, "rb") as f:
        # EDF header: first 256 bytes = global header
        header = f.read(256)
        if len(header) < 256:
            return None, None, None

        # Number of signals (channels): bytes 252-255
        try:
            ns = int(header[252:256].decode("ascii").strip())
        except Exception:
            return None, None, None

        # Number of data records: bytes 236-243
        try:
            n_records = int(header[236:244].decode("ascii").strip())
        except Exception:
            n_records = -1

        # Duration of each data record (seconds): bytes 244-251
        try:
            record_dur = float(header[244:252].decode("ascii").strip())
        except Exception:
            record_dur = -1

        # Total duration
        if n_records > 0 and record_dur > 0:
            duration_sec = n_records * record_dur
        else:
            duration_sec = None

        # Sample frequency: read per-signal header (256 bytes × ns signals)
        # Each signal header: first 256 bytes of ns-signal block
        # Bytes per signal in ns-signal block at offset 256+216*ns: samples per record
        sig_header_offset = 256
        # samples_per_record for signal 0 is at offset 256 + 216*ns + 8*0
        try:
            f.seek(sig_header_offset + 216 * ns)
            spr_bytes = f.read(8 * ns)
            spr_0 = int(spr_bytes[0:8].decode("ascii").strip())
            sfreq = spr_0 / record_dur if record_dur > 0 else None
        except Exception:
            sfreq = None

    return duration_sec, sfreq, ns

# ─── Walk EDF directory ───────────────────────────────────────────────────────

print("Scanning EDF directory...")
print(f"Looking in: {EDF_DIR.resolve()}\n")

if not EDF_DIR.exists():
    print(f"ERROR: {EDF_DIR} does not exist. Run from /mnt/c/Users/chima/seeg_study")
    exit(1)

rows = []
unrecognised_folders = set()

for sid_dir in sorted(EDF_DIR.iterdir()):
    if not sid_dir.is_dir():
        continue
    sid = sid_dir.name

    for edf_path in sorted(sid_dir.rglob("*.edf")):
        # Try parent folder first, then grandparent
        state = folder_to_state(edf_path.parent.name)
        if state is None:
            state = folder_to_state(edf_path.parent.parent.name)
        if state is None:
            unrecognised_folders.add(
                f"{sid} / {edf_path.parent.name} / {edf_path.parent.parent.name}")
            state = "UNKNOWN"

        dur, sfreq, n_ch = read_edf_header(str(edf_path))

        rows.append({
            "subject_id":   sid,
            "state":        state,
            "filename":     edf_path.name,
            "folder":       edf_path.parent.name,
            "duration_min": round(dur / 60, 2) if dur else None,
            "sfreq_hz":     sfreq,
            "n_channels":   n_ch,
            "full_path":    str(edf_path),
        })

df = pd.DataFrame(rows)

# ─── Print full inventory ─────────────────────────────────────────────────────

print("=" * 80)
print("FULL EDF INVENTORY")
print("=" * 80)
print(df[["subject_id","state","filename","duration_min","sfreq_hz","n_channels"]]
      .to_string(index=False))

# ─── Summary: event counts per subject × state ───────────────────────────────

print("\n" + "=" * 80)
print("EVENT COUNTS  (number of EDF recordings per subject × state)")
print("=" * 80)
known = df[df["state"] != "UNKNOWN"]
pivot_n = known.pivot_table(
    index="subject_id", columns="state",
    values="filename", aggfunc="count", fill_value=0)
pivot_n["TOTAL_EVENTS"] = pivot_n.sum(axis=1)
print(pivot_n.to_string())

# ─── Summary: total duration per subject × state ─────────────────────────────

print("\n" + "=" * 80)
print("TOTAL DURATION per subject × state  (minutes)")
print("=" * 80)
pivot_dur = known.pivot_table(
    index="subject_id", columns="state",
    values="duration_min", aggfunc="sum", fill_value=0).round(1)
pivot_dur["TOTAL_MIN"] = pivot_dur.sum(axis=1).round(1)
print(pivot_dur.to_string())

# ─── Sample rate check ────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("SAMPLE RATES")
print("=" * 80)
print(df["sfreq_hz"].value_counts().to_string())
unusual = df[~df["sfreq_hz"].isin([256, 512, 1024, 2048])]
if len(unusual):
    print(f"\nFiles with unusual sample rates ({len(unusual)}):")
    print(unusual[["subject_id","state","filename","sfreq_hz"]].to_string(index=False))
else:
    print("All files have standard sample rates.")

# ─── Unrecognised folders ─────────────────────────────────────────────────────

if unrecognised_folders:
    print("\n" + "=" * 80)
    print("UNRECOGNISED FOLDERS (state label could not be determined)")
    print("=" * 80)
    for f in sorted(unrecognised_folders):
        print(f"  {f}")
    print("\nAdd these to STATE_FOLDER_MAP in step1_edf_inventory.py if needed.")

# ─── Save ─────────────────────────────────────────────────────────────────────

out_path = Path("outputs/results/edf_inventory.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"\n✓ Saved: {out_path}")
print(f"  Total EDF files found: {len(df)}")
print(f"  Total subjects:        {df['subject_id'].nunique()}")
print(f"  Total states found:    {known['state'].nunique()}")
