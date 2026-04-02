import numpy as np, pandas as pd, itertools
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(".")
CONN_CSV = DATA_DIR / "outputs/results/step5_connectivity.csv"
STAT_CSV = DATA_DIR / "outputs/results/step6_statistics.csv"
MANIF_CSV= DATA_DIR / "outputs/results/step6_manifold.csv"
SUMM_TXT = DATA_DIR / "outputs/results/step6_summary.txt"

def ts(): return datetime.now().strftime("%H:%M:%S")
def log(msg,level="ok"):
    c={"ok":"\033[92m","warn":"\033[93m","err":"\033[91m"}.get(level,"")
    print(f"[{ts()}] {c}{msg}\033[0m",flush=True)

def cohen_d(a,b):
    na,nb=len(a),len(b)
    if na<2 or nb<2: return float("nan")
    pooled=np.sqrt(((na-1)*np.var(a,ddof=1)+(nb-1)*np.var(b,ddof=1))/(na+nb-2))
    return (np.mean(a)-np.mean(b))/(pooled+1e-12)

Path("outputs/results").mkdir(parents=True,exist_ok=True)
if not CONN_CSV.exists(): print(f"Not found: {CONN_CSV}"); exit(1)
df=pd.read_csv(CONN_CSV)
log(f"Loaded {len(df)} rows")
states=sorted(df["state"].unique())
metrics=[m for m in ["icoh_broadband","icoh_alpha","icoh_beta","icoh_theta",
                     "icoh_delta","icoh_lgamma","granger_net"] if m in df.columns]

from scipy.stats import mannwhitneyu,kruskal
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

log("Pairwise Mann-Whitney U tests...")
stat_rows=[]
for s1,s2 in itertools.combinations(states,2):
    for metric in metrics:
        v1=df[df["state"]==s1][metric].dropna().values
        v2=df[df["state"]==s2][metric].dropna().values
        if len(v1)<3 or len(v2)<3: continue
        try:
            stat,p=mannwhitneyu(v1,v2,alternative="two-sided")
            stat_rows.append({"state1":s1,"state2":s2,"metric":metric,
                "mean1":round(float(np.mean(v1)),6),"mean2":round(float(np.mean(v2)),6),
                "n1":len(v1),"n2":len(v2),"U_stat":round(float(stat),2),
                "p_raw":float(p),"cohen_d":round(float(cohen_d(v1,v2)),4)})
        except: continue
stat_df=pd.DataFrame(stat_rows)
if len(stat_df):
    _,q,_,_=multipletests(stat_df["p_raw"].values,method="fdr_bh")
    stat_df["q_fdr"]=q; stat_df["significant"]=q<0.05
    stat_df=stat_df.sort_values("q_fdr").reset_index(drop=True)
stat_df.to_csv(STAT_CSV,index=False)
n_sig=int(stat_df["significant"].sum()) if "significant" in stat_df.columns else 0
log(f"{len(stat_df)} comparisons, {n_sig} significant (FDR q<0.05)")

log("Kruskal-Wallis per nucleus...")
kw_rows=[]
for nucleus in ["CM","Pul","AV","all_thalamus"]:
    sub=df[df["nucleus"]==nucleus]
    if not len(sub): continue
    for metric in ["icoh_broadband","icoh_alpha","granger_net"]:
        if metric not in sub.columns: continue
        groups=[sub[sub["state"]==s][metric].dropna().values
                for s in states if len(sub[sub["state"]==s])>=3]
        if len(groups)<2: continue
        try:
            H,p=kruskal(*groups)
            kw_rows.append({"nucleus":nucleus,"metric":metric,
                "H":round(H,3),"p":round(p,4),"n_groups":len(groups),
                "n_obs":sum(len(g) for g in groups)})
        except: continue
kw_df=pd.DataFrame(kw_rows)
if len(kw_df):
    print("\nKruskal-Wallis per nucleus:")
    print(kw_df.to_string(index=False))

log("PCA manifold...")
feat_cols=[c for c in df.columns if c.startswith("icoh_")]
manif_df=df[df["nucleus"]=="all_thalamus"].dropna(subset=feat_cols).copy()
parts=[]
for sid,grp in manif_df.groupby("subject_id"):
    if len(grp)<2: continue
    grp=grp.copy(); grp[feat_cols]=StandardScaler().fit_transform(grp[feat_cols].values)
    parts.append(grp)
var=[0,0]; scaled=None
if parts:
    scaled=pd.concat(parts,ignore_index=True)
    X=scaled[feat_cols].fillna(0).values
    n_comp=min(5,X.shape[0]-1,X.shape[1])
    pca=PCA(n_components=n_comp); coords=pca.fit_transform(X)
    scaled["PC1"]=coords[:,0]; scaled["PC2"]=coords[:,1] if n_comp>1 else 0.0
    var=pca.explained_variance_ratio_
    log(f"PCA: PC1={var[0]:.1%}  PC2={var[1]:.1%}  total={sum(var[:2]):.1%}")
    print("\nState positions on PC1:")
    for state,val in scaled.groupby("state")["PC1"].mean().sort_values().items():
        print(f"  {state:<20} PC1={val:+.3f}")
    scaled.to_csv(MANIF_CSV,index=False)

lines=["="*65,"STEP 6 SUMMARY",f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}","="*65,""]
lines+=["iCoh broadband by state (all_thalamus)","-"*45]
for state,val in df[df["nucleus"]=="all_thalamus"].groupby("state")["icoh_broadband"].mean().sort_values(ascending=False).items():
    lines.append(f"  {state:<20} {val:.4f}")
lines+=["","Nucleus x state broadband iCoh","-"*45]
pivot=df[df["nucleus"].isin(["CM","Pul","AV"])].groupby(["nucleus","state"])["icoh_broadband"].mean().unstack().round(4)
lines.append(pivot.to_string())
lines+=["","Granger net (TC-CT) by state","-"*45]
for state,val in df[df["nucleus"]=="all_thalamus"].groupby("state")["granger_net"].mean().sort_values(ascending=False).items():
    lines.append(f"  {state:<20} {val:+.4f}")
lines+=["",f"Pairwise tests: {len(stat_df)}, significant: {n_sig}","-"*45]
if n_sig>0:
    sig=stat_df[stat_df["significant"]].head(15)
    for _,r in sig.iterrows():
        lines.append(f"  {r['state1']:<13}{r['state2']:<13}{r['metric']:<20}"
                     f"{r['mean1']:8.4f}{r['mean2']:8.4f}{r['cohen_d']:7.3f}{r['q_fdr']:12.2e}")
if len(kw_df): lines+=["","Kruskal-Wallis","-"*45,kw_df.to_string(index=False)]
if scaled is not None:
    lines+=["",f"PCA: PC1={var[0]:.1%} PC2={var[1]:.1%}","-"*45]
    for state,val in scaled.groupby("state")["PC1"].mean().sort_values().items():
        lines.append(f"  {state:<20} PC1={val:+.3f}")
report="\n".join(lines)
print("\n"+report)
open(SUMM_TXT,"w").write(report)
log(f"Done. Summary: {SUMM_TXT} | Stats: {STAT_CSV} | Manifold: {MANIF_CSV}")
print("\nIf results look correct, approve Step 7 (publication figures).")
