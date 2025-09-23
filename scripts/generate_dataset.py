import os, numpy as np, pandas as pd
os.makedirs("data", exist_ok=True)
rng=np.random.default_rng(42); n=4000
age=rng.normal(30,5,n).clip(18,45)
bmi=rng.normal(27,5,n).clip(18,45)
parity=rng.integers(0,3,n)
prior_gdm=rng.integers(0,2,n)
family_dm=rng.integers(0,2,n)
ethnicity_bin=rng.integers(0,2,n)
gwg1=rng.normal(0.35,0.15,n).clip(-0.2,0.9)
gwg2=rng.normal(0.45,0.18,n).clip(-0.2,1.1)
gwg3=rng.normal(0.30,0.15,n).clip(-0.3,0.9)
gwg_dev=rng.normal(0.8,0.8,n).clip(-2.0,3.0)
rppg_bpm_mean=rng.normal(82,9,n).clip(50,140)
rppg_snr_mean=rng.normal(2.1,0.7,n).clip(0,6)
perclos=rng.normal(0.22,0.08,n).clip(0,1)
blink_per_min=rng.normal(14,5,n).clip(0,45)
neck_norm=rng.normal(2.0,0.25,n).clip(1.2,3.2)
PRS_0to10=rng.integers(0,11,n)
PRS_missing=(rng.random(n)<0.1).astype(int)
def z(x): return (x-x.mean())/(x.std()+1e-6)
lin_pp= -2.0 + 0.05*z(age) + 0.10*z(bmi) + 0.30*prior_gdm + 0.18*family_dm + 0.12*z(gwg2) + 0.10*z(gwg_dev) + 0.08*z(rppg_bpm_mean) + 0.14*z(perclos) + 0.08*z(neck_norm) + 0.06*(PRS_0to10/10) - 0.10*z(rppg_snr_mean)
pp_prob=1/(1+np.exp(-lin_pp))
pp_glucose_test_done_T4=(rng.random(n)<pp_prob).astype(int)
lin_t2d= -3.0 + 0.05*z(bmi) + 0.20*prior_gdm + 0.10*family_dm + 0.10*z(gwg3) + 0.15*z(perclos) + 0.10*z(neck_norm) + 0.12*(PRS_0to10/10) - 0.08*z(rppg_snr_mean)
t2d_prob=1/(1+np.exp(-lin_t2d))
incident_T2D_12m=(rng.random(n)<t2d_prob).astype(int)
df=pd.DataFrame(dict(
  age=age,bmi=bmi,parity=parity,prior_gdm=prior_gdm,family_dm=family_dm,ethnicity_bin=ethnicity_bin,
  gwg_slope_kg_per_wk_T1=gwg1,gwg_slope_kg_per_wk_T2=gwg2,gwg_slope_kg_per_wk_T3=gwg3,gwg_dev_from_IOM=gwg_dev,
  rppg_bpm_mean=rppg_bpm_mean,rppg_snr_mean=rppg_snr_mean,perclos=perclos,blink_per_min=blink_per_min,neck_norm=neck_norm,
  PRS_0to10=PRS_0to10,PRS_missing=PRS_missing,
  pp_glucose_test_done_T4=pp_glucose_test_done_T4,incident_T2D_12m=incident_T2D_12m
))
df.to_csv("data/fake_cohort_T1_T4.csv", index=False)
print("data/fake_cohort_T1_T4.csv")
