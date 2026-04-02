import re, argparse, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count

DATA_DIR = Path(".")
PROC_DIR = DATA_DIR / "processed_npz"
GT_CSV   = DATA_DIR / "outputs/results/thalamic_nuclei_ground_truth.csv"
OUT_CSV  = DATA_DIR / "outputs/results/step5_connectivity.csv"
SFREQ=256; EPOCH_SEC=4.0; MIN_EPOCHS=5; NPERSEG=256
N_JOBS=min(4,cpu_count())
FREQ_BANDS={"delta":(0.5,4),"theta":(4,8),"alpha":(8,13),
            "beta":(13,30),"lgamma":(30,70),"broadband":(0.5,70)}

def ts(): return datetime.now().strftime("%H:%M:%S")
def log(msg,level="ok"):
    c={"ok":"\033[92m","warn":"\033[93m","err":"\033[91m"}.get(level,"")
    print(f"[{ts()}] {c}{msg}\033[0m",flush=True)

def band_mean(vals,freqs,fmin,fmax):
    m=(freqs>=fmin)&(freqs<=fmax)
    return float(np.mean(vals[m])) if m.any() else float("nan")

def compute_pair(xt,xc,sfreq):
    from scipy.signal import csd,welch
    nperseg=min(NPERSEG,len(xt)//2)
    if nperseg<16: return None
    f,Pxy=csd(xt,xc,fs=sfreq,nperseg=nperseg)
    _,Pxx=welch(xt,fs=sfreq,nperseg=nperseg)
    _,Pyy=welch(xc,fs=sfreq,nperseg=nperseg)
    d=np.sqrt(Pxx*Pyy)+1e-12
    icoh=np.abs(np.imag(Pxy))/d
    gc_tc=np.abs(np.imag(Pxy))/(Pyy+1e-12)
    gc_ct=np.abs(np.imag(np.conj(Pxy)))/(Pxx+1e-12)
    r={}
    for band,(fmin,fmax) in FREQ_BANDS.items():
        r[f"icoh_{band}"]=band_mean(icoh,f,fmin,fmax)
    r["granger_tc"]=band_mean(gc_tc,f,0.5,70)
    r["granger_ct"]=band_mean(gc_ct,f,0.5,70)
    r["granger_net"]=r["granger_tc"]-r["granger_ct"]
    return r

def get_idx(chs,contacts):
    cs=set(contacts)
    return [i for i,ch in enumerate(chs) if ch.split("-")[0] in cs]

def connectivity_worker(args):
    sid,state,clip_paths,gt_sub=args
    rows=[]
    try:
        if gt_sub is None or len(gt_sub)==0: return rows
        thal_all=gt_sub[gt_sub["is_thalamic"]==True]["contact_name"].tolist()
        cort_all=gt_sub[gt_sub["is_thalamic"]==False]["contact_name"].tolist()
        if not thal_all or not cort_all: return rows
        nuclei={"all_thalamus":thal_all}
        for nuc in gt_sub[gt_sub["is_thalamic"]==True]["thalamic_nucleus"].dropna().unique():
            if nuc:
                nc=gt_sub[gt_sub["thalamic_nucleus"]==nuc]["contact_name"].tolist()
                if nc: nuclei[str(nuc)]=nc
        for nucleus,nuc_contacts in nuclei.items():
            clip_metrics=[]; n_clips_used=0; n_epochs_total=0
            for npz_path in clip_paths:
                try:
                    d=np.load(str(npz_path),allow_pickle=True)
                    data=d["data"].astype(np.float64)
                    sfreq=float(d["sfreq"])
                    chs=list(d["ch_names"])
                    step=int(sfreq*EPOCH_SEC)
                    n_ep=data.shape[1]//step
                    if n_ep<MIN_EPOCHS: continue
                    t_idx=get_idx(chs,nuc_contacts)
                    c_idx=get_idx(chs,cort_all)
                    if not t_idx or not c_idx: continue
                    pair_ep_metrics=[]
                    for ti in t_idx:
                        for ci in c_idx:
                            ep_vals=[]
                            for e in range(n_ep):
                                s=e*step
                                m=compute_pair(data[ti,s:s+step],data[ci,s:s+step],sfreq)
                                if m: ep_vals.append(m)
                            if ep_vals:
                                avg={k:float(np.nanmean([v[k] for v in ep_vals]))
                                     for k in ep_vals[0]}
                                pair_ep_metrics.append(avg)
                    if not pair_ep_metrics: continue
                    clip_avg={k:float(np.nanmean([v[k] for v in pair_ep_metrics]))
                              for k in pair_ep_metrics[0]}
                    clip_metrics.append(clip_avg)
                    n_clips_used+=1; n_epochs_total+=n_ep
                except Exception: continue
            if not clip_metrics: continue
            final={k:float(np.nanmean([v[k] for v in clip_metrics]))
                   for k in clip_metrics[0]}
            final.update({"subject_id":sid,"state":state,"nucleus":nucleus,
                          "n_clips_used":n_clips_used,
                          "n_clips_total":len(clip_paths),
                          "n_epochs_total":n_epochs_total})
            rows.append(final)
    except Exception: pass
    return rows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--skip_existing",action="store_true")
    parser.add_argument("--n_jobs",type=int,default=N_JOBS)
    args=parser.parse_args()
    Path("outputs/results").mkdir(parents=True,exist_ok=True)
    if not GT_CSV.exists(): log(f"GT CSV not found: {GT_CSV}","err"); return
    gt=pd.read_csv(GT_CSV)
    gt["contact_num"]=gt["contact_name"].apply(
        lambda x: int(m.group()) if (m:=re.search(r"\d+$",str(x))) else 0)
    log(f"Ground truth: {len(gt)} contacts, {gt['subject_id'].nunique()} subjects")
    clip_files={}
    for npz in sorted(PROC_DIR.glob("*_clip*.npz")):
        m=re.match(r"^(.+?)_(crying|eating|laughing|nrem_sleep|playing|"
                   r"reading|rem_sleep|talking|watching_tv)_clip(\d+)\.npz$",npz.name)
        if not m: continue
        key=(m.group(1),m.group(2))
        clip_files.setdefault(key,[]).append(str(npz))
    log(f"Subject x state combinations: {len(clip_files)}")
    existing=set()
    if args.skip_existing and OUT_CSV.exists():
        done=pd.read_csv(OUT_CSV)
        for _,r in done.iterrows(): existing.add((r["subject_id"],r["state"]))
        log(f"Skipping {len(existing)} done combinations")
    tasks=[(sid,state,sorted(clips),gt[gt["subject_id"]==sid].copy())
           for (sid,state),clips in sorted(clip_files.items())
           if (sid,state) not in existing]
    log(f"Tasks: {len(tasks)} | Workers: {args.n_jobs}")
    all_rows=[]; completed=0
    if args.n_jobs>1:
        with Pool(args.n_jobs) as pool:
            for rows in pool.imap_unordered(connectivity_worker,tasks):
                all_rows.extend(rows); completed+=1
                if completed%10==0 or completed==len(tasks):
                    log(f"  {completed}/{len(tasks)} done, {len(all_rows)} rows")
    else:
        for i,task in enumerate(tasks):
            print(f"[{i+1:3d}/{len(tasks)}] {task[0]:<6} {task[1]:<12}",end=" ",flush=True)
            rows=connectivity_worker(task)
            all_rows.extend(rows); print(f"-> {len(rows)} rows")
    if not all_rows: log("No rows computed","warn"); return
    conn_df=pd.DataFrame(all_rows)
    if args.skip_existing and OUT_CSV.exists():
        conn_df=pd.concat([pd.read_csv(OUT_CSV),conn_df],ignore_index=True)
    conn_df.to_csv(OUT_CSV,index=False)
    print(f"\nDone. {len(conn_df)} rows saved to {OUT_CSV}")
    print("\nMean broadband iCoh by state:")
    for state,val in conn_df.groupby("state")["icoh_broadband"].mean()\
                             .sort_values(ascending=False).items():
        print(f"  {state:<20} {val:.4f}")
    print("\nMean Granger net (TC-CT) by state:")
    for state,val in conn_df.groupby("state")["granger_net"].mean()\
                             .sort_values(ascending=False).items():
        print(f"  {state:<20} {val:.4f}")
    print(f"\nReview values above and approve Step 6 (statistics + manifold).")

if __name__=="__main__":
    main()
