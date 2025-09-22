import json,math,os
import numpy as np,pandas as pd
from pathlib import Path
rng=np.random.default_rng(12345)
N=10000
def clip(a,lo,hi): return float(np.clip(a,lo,hi))
ETH=["White","Black","Hispanic","Asian","Other"]
def draw_subject(i):
    age=clip(rng.normal(30,5),15,49)
    eth=ETH[rng.integers(0,len(ETH))]
    BMI=clip(rng.normal(27,6),16,55)
    parity=int(np.clip(rng.poisson(1.0),0,6))
    prior_GDM=int(rng.random()<0.12)
    family_DM=int(rng.random()<0.28)
    chronic_HTN=int(rng.random()<0.10)
    prior_macrosomia=int(rng.random()<0.08)
    PRS_missing=int(rng.random()>0.15)
    PRS_0to10=float(np.clip(rng.normal(5,2),0,10)) if PRS_missing==0 else np.nan
    base_fglu=clip(85+0.6*(BMI-25)+rng.normal(0,8),70,140)
    base_a1c=clip(5.1+0.03*(BMI-25)+rng.normal(0,0.25),4.8,6.5)
    return dict(subj=i,age=age,ethnicity=eth,BMI=BMI,parity=parity,prior_GDM=prior_GDM,family_DM=family_DM,chronic_HTN=chronic_HTN,prior_macrosomia=prior_macrosomia,PRS_0to10=PRS_0to10,PRS_missing=PRS_missing,base_fglu=base_fglu,base_a1c=base_a1c)
subs=[draw_subject(i) for i in range(N)]
rows=[]
for s in subs:
    for stage,gw_lo,gw_hi in [("T1",6,13),("T2",18,26),("T3",24,28)]:
        gw=clip(rng.uniform(gw_lo,gw_hi),6,40)
        gwg_slope=clip(rng.normal(0.35,0.20),-0.10,1.20)
        iom_target=0.35
        gwg_dev=clip(gwg_slope-iom_target,-0.50,0.80)
        neck_circ_norm=clip(1.00+0.01*(s["BMI"]-25)+rng.normal(0,0.03),0.85,1.25)
        edema_proxy=clip(0.05+0.10*max(0,gwg_dev)+rng.normal(0,0.05),-0.20,0.60)
        rPPG_HR=clip(75+0.4*(s["BMI"]-25)+rng.normal(0,6),55,110)
        rPPG_stability=clip(1.0/(1.0+np.exp(-(rng.normal(0.0,1.0)-0.2))),0.0,1.0)
        sleep_hours_24h=clip(rng.normal(7.2,1.0),3.0,10.0)
        sleep_quality_score=float(np.clip(70+5*(sleep_hours_24h-7)+rng.normal(0,8),0,100))
        perclos=clip(0.18+0.02*(7.0-sleep_hours_24h)+rng.normal(0,0.03),0.05,0.50)
        blink_rate_min=clip(rng.normal(18+2*(7.0-sleep_hours_24h),5),8,35)
        fasting_glucose=clip(s["base_fglu"]+rng.normal(0,5),70,140)
        A1C=clip(s["base_a1c"]+rng.normal(0,0.15),4.8,6.5)
        if rng.random()<0.30 and stage in ["T1","T2"]: fasting_glucose=np.nan
        if rng.random()<0.20 and stage in ["T1","T2"]: A1C=np.nan
        if rng.random()<0.30: sleep_hours_24h=np.nan
        if rng.random()<0.30: rPPG_HR=np.nan
        if rng.random()<0.30: rPPG_stability=np.nan
        rows.append(dict(subj=s["subj"],stage_code=stage,gestational_week_at_capture=gw,age=s["age"],ethnicity=s["ethnicity"],BMI=s["BMI"],parity=s["parity"],prior_GDM=s["prior_GDM"],family_DM=s["family_DM"],chronic_HTN=s["chronic_HTN"],prior_macrosomia=s["prior_macrosomia"],fasting_glucose=fasting_glucose,A1C=A1C,gwg_slope_kg_per_wk=gwg_slope,gwg_dev_from_IOM=gwg_dev,rPPG_HR=rPPG_HR,rPPG_stability=rPPG_stability,perclos=perclos,blink_rate_min=blink_rate_min,neck_circ_norm=neck_circ_norm,edema_proxy=edema_proxy,sleep_hours_24h=sleep_hours_24h,sleep_quality_score=sleep_quality_score,PRS_0to10=s["PRS_0to10"],PRS_missing=s["PRS_missing"]))
df=pd.DataFrame(rows)
BMI=df["BMI"]
FG=df["fasting_glucose"].fillna(85.0)
A1C=df["A1C"].fillna(5.2)
GWG=df["gwg_dev_from_IOM"]
PRS=df["PRS_0to10"].fillna(5.0)
NECK=df["neck_circ_norm"]
EDE=df["edema_proxy"]
SLP=df["sleep_hours_24h"].fillna(7.0)
STAB=df["rPPG_stability"].fillna(0.5)
AGE=df["age"]
ETH_SHIFT=df["ethnicity"].map({"White":0.0,"Asian":0.05,"Hispanic":0.10,"Black":0.12,"Other":0.02}).fillna(0.0)
logit=-3.2
logit+=0.07*(BMI-25)
logit+=0.04*(FG-85)
logit+=0.35*((A1C-5.2)/0.3)
logit+=0.80*df["prior_GDM"]
logit+=0.30*df["family_DM"]
logit+=0.50*GWG
logit+=0.15*((PRS-5)/2.0)
logit+=0.12*((NECK-1.0)/0.1)
logit+=0.10*(EDE/0.2)
logit+=0.20*((7.0-SLP)/2.0)
logit+=0.12*((0.5-STAB)/0.25)
logit+=0.10*((AGE-30)/5.0)
logit+=ETH_SHIFT
p_raw=1/(1+np.exp(-logit))
mask_early=df.stage_code.isin(["T1","T2","T3"])
avg=np.nanmean(p_raw[mask_early])
target=0.10
calib=np.log(target/(1-target))-np.log(avg/(1-avg)) if np.isfinite(avg) and 0<avg<1 else 0.0
p=1/(1+np.exp(-(logit+calib)))
gdm_by_subj={}
for sid,g in df.groupby("subj"):
    prob=float(np.nanmean(p.loc[g.index]))
    if not np.isfinite(prob): prob=target
    y=int(rng.random()<prob)
    gdm_by_subj[sid]=y
df["GDM_dx"]=df["subj"].map(gdm_by_subj).astype(int)
t4=[]
for s in subs:
    y=gdm_by_subj[s["subj"]]
    base=0.58
    base+=0.06 if y==1 else 0.0
    base-=0.04 if s["ethnicity"] in ["Black","Hispanic"] else 0.0
    base+=0.04 if s["parity"]<=1 else -0.04
    base=np.clip(base,0.20,0.90)
    done=int(rng.random()<base)
    val=np.nan
    if done==1:
        mu=94 if y==0 else 112
        val=clip(rng.normal(mu,12),70,180)
    predm=int(done==1 and val>=100 and val<126)
    t2d12=int((y==1 and done==1 and val>=126) or (y==1 and rng.random()<0.06))
    wr12=clip(rng.normal(4.0+1.5*y,2.5),-2,15)
    t4.append(dict(subj=s["subj"],stage_code="T4",gestational_week_at_capture=44.0,age=s["age"],ethnicity=s["ethnicity"],BMI=s["BMI"],parity=s["parity"],prior_GDM=s["prior_GDM"],family_DM=s["family_DM"],chronic_HTN=s["chronic_HTN"],prior_macrosomia=s["prior_macrosomia"],fasting_glucose=np.nan,A1C=np.nan,gwg_slope_kg_per_wk=np.nan,gwg_dev_from_IOM=np.nan,rPPG_HR=np.nan,rPPG_stability=np.nan,perclos=np.nan,blink_rate_min=np.nan,neck_circ_norm=np.nan,edema_proxy=np.nan,sleep_hours_24h=np.nan,sleep_quality_score=np.nan,PRS_0to10=(s["PRS_0to10"] if not np.isnan(s["PRS_0to10"]) else np.nan),PRS_missing=s["PRS_missing"],GDM_dx=y,pp_glucose_test_done_T4=done,pp_glucose_value_T4=(val if done==1 else np.nan),prediabetes_postpartum=predm,incident_T2D_12m=t2d12,weight_retention_12m=wr12))
df_t4=pd.DataFrame(t4)
for c in ["pp_glucose_test_done_T4","pp_glucose_value_T4","prediabetes_postpartum","incident_T2D_12m","weight_retention_12m"]:
    if c not in df.columns: df[c]=np.nan
df=pd.concat([df,df_t4],ignore_index=True)
df["sleep_hours_24h_missing"]=df["sleep_hours_24h"].isna().astype(int)
df["fasting_glucose_missing"]=df["fasting_glucose"].isna().astype(int)
df["A1C_missing"]=df["A1C"].isna().astype(int)
df["rPPG_missing"]=(df["rPPG_HR"].isna() | df["rPPG_stability"].isna()).astype(int)
Path("data").mkdir(parents=True,exist_ok=True)
out="data/GDMGuard_dataset_v3_T1T2T3T4.csv"
df.to_csv(out,index=False)
qc={ "rows":int(df.shape[0]), "cols":int(df.shape[1]), "subjects":int(df["subj"].nunique()),
     "gdm_prev_est":float(df[df.stage_code.isin(["T1","T2","T3"])].groupby("subj")["GDM_dx"].max().mean()),
     "t4_completion_rate":float(pd.to_numeric(df.loc[df.stage_code=="T4","pp_glucose_test_done_T4"]).mean())}
Path("reports").mkdir(parents=True,exist_ok=True)
with open("reports/leakage_checks.txt","w") as f: f.write(json.dumps(qc,indent=2))
print(out,qc)
