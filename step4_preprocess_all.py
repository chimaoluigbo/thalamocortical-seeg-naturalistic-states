import re, argparse, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(".")
EDF_DIR  = DATA_DIR / "edf"
PROC_DIR = DATA_DIR / "processed_npz"
GT_CSV   = DATA_DIR / "outputs/results/thalamic_nuclei_ground_truth.csv"
LOG_CSV  = DATA_DIR / "outputs/results/step4_processing_log.csv"
SFREQ = 256
NOTCH = 60.0

STATE_MAP = {
    "EATING":"eating","DRINKING":"eating","EAT":"eating",
    "PLAYING":"playing","PLAY":"playing",
    "READING":"reading","READ":"reading",
    "TALKING":"talking","TALK":"talking",
    "SMILING":"laughing","SMILE":"laughing","LAUGHING":"laughing",
    "WATCHING":"watching_tv","WATCH":"watching_tv",
    "PAIN":"crying","UPSET":"crying","CRYING":"crying",
    "NONEREM":"nrem_sleep","NONREM":"nrem_sleep","NREM":"nrem_sleep","NON-REM":"nrem_sleep",
    "REM":"rem_sleep",
}

def folder_to_state(name):
    u = name.upper()
    for kw in ["NONEREM","NON-REM","NONREM","NREM","REM"]:
        if kw in u: return STATE_MAP[kw]
    for kw, lbl in STATE_MAP.items():
        if kw in u: return lbl
    return None

def is_non_seeg(name):
    s = name.strip().upper()
    if s == "E": return True
    if s.startswith("EKG"): return True
    if s.startswith("DC"): return True
    if s.startswith("MKR"): return True
    for kw in ["OFF","STATUS","TRIGGER","ANNOT","EDF"]:
        if kw in s: return True
    return False

SEEG_PAT = re.compile(r"^[A-Za-z]+\d+$")
def is_valid_seeg(name): return bool(SEEG_PAT.match(name.strip()))
def ts(): return datetime.now().strftime("%H:%M:%S")
def log(msg, level="ok"):
    c = {"ok":"\033[92m","warn":"\033[93m","err":"\033[91m"}.get(level,"")
    print(f"[{ts()}] {c}{msg}\033[0m", flush=True)

def preprocess_clip(edf_path, sid, state, clip_idx, gt_sub, skip_existing):
    import mne
    out_npz = PROC_DIR / f"{sid}_{state}_clip{clip_idx:02d}.npz"
    if skip_existing and out_npz.exists():
        d = np.load(str(out_npz), allow_pickle=True)
        return {"status":"skipped","n_bipolar_ch":d["data"].shape[0],
                "n_epochs":int(d["data"].shape[1]/(SFREQ*4)),
                "duration_min":round(d["data"].shape[1]/SFREQ/60,2),
                "out_file":out_npz.name,"error":""}
    result = {"status":"failed","n_bipolar_ch":0,"n_epochs":0,
              "duration_min":0,"out_file":"","error":""}
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        rename = {}
        for ch in raw.ch_names:
            s = ch.strip()
            rename[ch] = s[4:].strip() if s.upper().startswith("POL ") else s
        raw.rename_channels(rename)
        drop = [ch for ch in raw.ch_names if is_non_seeg(ch) or not is_valid_seeg(ch)]
        if drop: raw.drop_channels(drop)
        if len(raw.ch_names) < 2:
            result["error"] = "Too few channels"; return result
        if raw.info["sfreq"] != SFREQ:
            raw.resample(SFREQ, npad="auto", verbose=False)
        raw.filter(l_freq=0.5, h_freq=100.0, method="fir", fir_design="firwin", verbose=False)
        raw.notch_filter(freqs=[NOTCH, NOTCH*2], verbose=False)
        current = set(raw.ch_names)
        anodes, cathodes, bnames = [], [], []
        if gt_sub is not None and len(gt_sub) > 0:
            for shaft, grp in gt_sub.groupby("electrode_shaft"):
                chs = grp.sort_values("contact_num")["contact_name"].tolist()
                for i in range(len(chs)-1):
                    a,c = chs[i],chs[i+1]
                    if a in current and c in current:
                        anodes.append(a); cathodes.append(c); bnames.append(f"{a}-{c}")
        else:
            sm = {}
            for ch in raw.ch_names:
                m = re.match(r"^([A-Za-z]+)(\d+)$", ch)
                if m: sm.setdefault(m.group(1),[]).append((int(m.group(2)),ch))
            for s2, cs in sm.items():
                cs.sort()
                for i in range(len(cs)-1):
                    a=cs[i][1]; c=cs[i+1][1]
                    anodes.append(a); cathodes.append(c); bnames.append(f"{a}-{c}")
        if not anodes:
            result["error"] = "No bipolar pairs"; return result
        raw = mne.set_bipolar_reference(raw, anode=anodes, cathode=cathodes,
                                         ch_name=bnames, verbose=False)
        data = raw.get_data().astype(np.float32)
        np.savez_compressed(str(out_npz), data=data, ch_names=list(raw.ch_names),
            sfreq=np.float32(SFREQ), sid=sid, state=state,
            clip_idx=clip_idx, edf_file=edf_path.name)
        result.update({"status":"ok","n_bipolar_ch":len(raw.ch_names),
            "n_epochs":int(raw.n_times/(SFREQ*4)),
            "duration_min":round(raw.times[-1]/60,2),"out_file":out_npz.name})
    except Exception as e:
        result["error"] = str(e)[:120]
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    Path("outputs/results").mkdir(parents=True, exist_ok=True)
    gt = None
    if GT_CSV.exists():
        gt = pd.read_csv(GT_CSV)
        gt["contact_num"] = gt["contact_name"].apply(
            lambda x: int(m.group()) if (m:=re.search(r"\d+$",str(x))) else 0)
        log(f"Ground truth: {len(gt)} contacts, {gt['subject_id'].nunique()} subjects")
    else:
        log("No ground truth CSV — using auto shafts","warn")
    all_clips = []
    for sid_dir in sorted(EDF_DIR.iterdir()):
        if not sid_dir.is_dir(): continue
        sid = sid_dir.name
        sc = {}
        for edf in sorted(sid_dir.rglob("*.edf")):
            state = folder_to_state(edf.parent.name) or folder_to_state(edf.parent.parent.name)
            if state: sc.setdefault(state,[]).append(edf)
        for state, edfs in sorted(sc.items()):
            for ci, edf in enumerate(sorted(edfs)):
                all_clips.append((sid,state,ci,edf))
    log(f"Total clips: {len(all_clips)}")
    log_rows = []
    n_ok=n_skip=n_fail=0
    for i,(sid,state,ci,edf) in enumerate(all_clips):
        gt_sub = gt[gt["subject_id"]==sid].copy() if gt is not None else None
        print(f"[{i+1:3d}/{len(all_clips)}] {sid:<6} {state:<12} clip{ci:02d}  {edf.name}",
              end="  ", flush=True)
        r = preprocess_clip(edf, sid, state, ci, gt_sub, args.skip_existing)
        if r["status"]=="ok":
            n_ok+=1
            print(f"OK  {r['duration_min']:.1f}min {r['n_epochs']}ep {r['n_bipolar_ch']}ch")
        elif r["status"]=="skipped":
            n_skip+=1; print("skipped")
        else:
            n_fail+=1; print(f"FAIL: {r['error']}")
        log_rows.append({"subject_id":sid,"state":state,"clip_idx":ci,
            "edf_file":edf.name,**r})
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(LOG_CSV, index=False)
    print(f"\nDone. OK={n_ok} Skip={n_skip} Fail={n_fail}")
    print(f"NPZ files: {len(list(PROC_DIR.glob('*.npz')))}")
    ok_df = log_df[log_df["status"].isin(["ok","skipped"])]
    if len(ok_df):
        pivot = ok_df.groupby(["subject_id","state"]).size().unstack(fill_value=0)
        pivot["TOTAL"] = pivot.sum(axis=1)
        print(pivot.to_string())
    print(f"Log: {LOG_CSV}")

if __name__=="__main__":
    main()
