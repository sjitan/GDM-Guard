import os,random
import pandas as pd, numpy as np
from datetime import datetime, timedelta
schema={"M0":["age","bmi","parity","prior_gdm","family_dm","ethnicity"],
        "M1":["gwg_slope","gwg_dev_iom"],
        "M2":["rppg_bpm_mean","perclos","blink_rate","neck_norm"],
        "M3":["prs_bin"]}
cols=["patient_id","measurement_date"]+schema["M0"]+schema["M1"]+schema["M2"]+schema["M3"]
os.makedirs("data",exist_ok=True)
n=200
start=datetime(2025,1,1)
ethn=["White","Black","Asian","Hispanic","Mixed","Other"]
rng=np.random.default_rng(42)
rows=[]
for i in range(n):
    pid=f"P{i:04d}"
    days=rng.choice(np.arange(0,210,7),size=rng.integers(1,5),replace=False)
    for d in sorted(days):
        age=int(rng.integers(18,45))
        bmi=float(np.round(rng.normal(27,5),1))
        parity=int(rng.integers(0,4))
        prior=int(rng.random()<0.18)
        fam=int(rng.random()<0.25)
        eth=str(ethn[int(rng.integers(len(ethn)))])
        gwg_slope=float(np.round(rng.normal(0.35,0.12),3))
        gwg_dev=float(np.round(rng.normal(0.0,0.4),3))
        bpm=float(np.round(rng.normal(78,10),1))
        perclos=float(np.round(np.clip(rng.normal(0.18,0.07),0,1),3))
        blink=float(np.round(np.clip(rng.normal(16,5),1,60),1))
        neck=float(np.round(np.clip(rng.normal(1.9,0.25),1.0,3.2),3))
        prs=int(rng.choice([0,1,2],p=[0.6,0.3,0.1]))
        rows.append([pid,(start+timedelta(days=int(d))).strftime("%Y-%m-%d"),
                     age,bmi,parity,prior,fam,eth,gwg_slope,gwg_dev,bpm,perclos,blink,neck,prs])
pd.DataFrame(rows,columns=cols).to_csv("data/fake_patients_M0_M3_timeseries.csv",index=False)
