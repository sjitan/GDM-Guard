import argparse,os,time,json
import numpy as np,pandas as pd
from joblib import load
from datetime import datetime
from extract_selfie_features import main as extract_main
def run_extract(video,device,duration,tmp_path):
    import sys,runpy
    argv=["extract_selfie_features.py"]
    if video: argv+=["--video",video]
    else: argv+=["--device",str(device)]
    argv+=["--duration",str(duration),"--out",tmp_path]
    sys.argv=argv
    runpy.run_module("extract_selfie_features",run_name="__main__")
def tier_of(p):
    if p>=0.30: return "High"
    if p>=0.20: return "Moderate"
    if p>=0.10: return "Low"
    return "Very Low"
def steps_of(t):
    if t=="High": return "Early OGTT 16–20w; nutrition consult now; postpartum test auto-enrolled"
    if t=="Moderate": return "Discuss early OGTT; nutrition referral; enroll for postpartum test"
    if t=="Low": return "Standard 24–28w OGTT; routine counseling; postpartum test reminder"
    return "Standard care; postpartum test reminder"
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video",type=str,default="")
    ap.add_argument("--device",type=int,default=0)
    ap.add_argument("--duration",type=float,default=10.0)
    ap.add_argument("--age",type=float,default=None)
    ap.add_argument("--bmi",type=float,default=None)
    ap.add_argument("--parity",type=int,default=None)
    ap.add_argument("--prior_gdm",type=int,default=0)
    ap.add_argument("--family_dm",type=int,default=0)
    ap.add_argument("--htn",type=int,default=0)
    ap.add_argument("--macrosomia",type=int,default=0)
    ap.add_argument("--ethnicity",type=str,default="Other")
    ap.add_argument("--prs",type=float,default=np.nan)
    ap.add_argument("--outdir",type=str,default="sessions")
    args=ap.parse_args()
    os.makedirs(args.outdir,exist_ok=True)
    tmp=os.path.join(args.outdir,"last_features.json")
    run_extract(args.video,args.device,args.duration,tmp)
    feats=json.load(open(tmp))
    age=args.age if args.age is not None else 30.0
    bmi=args.bmi if args.bmi is not None else 27.0
    parity=args.parity if args.parity is not None else 1
    df=pd.DataFrame([{
        "age":age,
        "BMI":bmi,
        "parity":parity,
        "prior_GDM":int(args.prior_gdm),
        "family_DM":int(args.family_dm),
        "chronic_HTN":int(args.htn),
        "prior_macrosomia":int(args.macrosomia),
        "ethnicity":args.ethnicity,
        "gwg_dev_from_IOM":0.0,
        "rPPG_HR":feats.get("rPPG_HR",np.nan),
        "rPPG_stability":feats.get("rPPG_stability",0.0),
        "sleep_hours_24h":feats.get("sleep_hours_24h",np.nan),
        "neck_circ_norm":feats.get("neck_circ_norm",np.nan),
        "PRS_0to10":args.prs,
        "PRS_missing":0 if np.isfinite(args.prs) else 1
    }])
    model_path="models/m3.joblib"
    if not os.path.exists(model_path): raise SystemExit("missing m3.joblib")
    model=load(model_path)
    prob=float(model.predict_proba(df)[0,1])
    t=tier_of(prob)
    s=steps_of(t)
    ts=int(time.time())
    row=df.copy()
    row["prob"]=prob
    row["risk_tier"]=t
    row["next_steps"]=s
    out_csv=os.path.join(args.outdir,f"session_{ts}.csv")
    row.to_csv(out_csv,index=False)
    print(f"{prob:.3f} {t} {s}\n{out_csv}")
if __name__=="__main__":
    main()
