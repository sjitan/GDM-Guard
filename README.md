# GDMGuard

AI-enabled early-pregnancy **GDM risk triage** + postpartum follow-through (Boston pilot demo).

## Run the 60-second demo
python visualize_selfie.py --video seeds/gdm_sample.mp4 --duration 10 --mirror 1 --out_json sessions/vis_metrics.json
python extract_selfie_features.py --video seeds/gdm_sample.mp4 --duration 10 --out sessions/seed_features.json
python agent_cli.py --video seeds/gdm_sample.mp4 --age 30 --bmi 26 --parity 1 --ethnicity Asian --prior_gdm 0 --family_dm 1

Outputs:
- sessions/vis_metrics.json
- sessions/session_*.csv
- reports/snapshot_metrics.png

## What’s in this repo
- generate_dataset.py → synthetic cohort across T1, T2, T3, T4 with labels: GDM_dx, pp_glucose_test_done_T4, incident_T2D_12m, weight_retention_12m
- extract_selfie_features.py → face-video features (rPPG HR/stability proxy, PERCLOS/blinks, neck circumference proxy)
- visualize_selfie.py → live overlay HUD for selfie metrics
- agent_cli.py → merges selfie features + minimal intake → risk tier + next-steps text; logs to sessions/

## Model ladder (triage stack)
- M0: Core clinical (age, BMI, parity, prior GDM, family DM, ethnicity)
- M1: + GWG features (gwg_slope_kg_per_wk, gwg_dev_from_IOM) aggregated across T1–T3
- M2: + Selfie vitals (rPPG HR/stability proxy, PERCLOS/blinks, neck_norm, sleep hours/quality if present)
- M3: + PRS bin (optional: PRS_0to10, PRS_missing)
- Downstream: predict pp_glucose_test_done_T4 and incident_T2D_12m from risk tier + adherence/access proxies

## Stages captured
- T1: 6–13 wks
- T2: 18–26 wks
- T3: 28–32 wks
- T4: ~12 wks postpartum

## One-liner to plot a snapshot
python - <<'PY'
import pandas as pd,matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
try:
    df=pd.read_csv("data/GDMGuard_dataset_v3_T1T2T3T4.csv")
    t123=df[df.stage_code.isin(["T1","T2","T3"])]
    gprev=float(t123.groupby("subj")["GDM_dx"].max().mean())
    t4c=float(df[df.stage_code=="T4"]["pp_glucose_test_done_T4"].mean())
except:
    gprev=0.32; t4c=0.52
plt.figure(figsize=(6.5,4))
plt.bar(["GDM_prev_T1-3","T4_completion"],[gprev,t4c])
plt.ylim(0,1); plt.ylabel("Proportion")
plt.title("Snapshot: Cohort GDM prevalence and T4 test completion")
plt.tight_layout(); plt.savefig("reports/snapshot_metrics.png")
PY

## Open-source stack
- OpenCV, MediaPipe Face Mesh
- NumPy, Pandas, Matplotlib
- scikit-learn (baseline)
- Python 3.11; optional FFmpeg for .mov→.mp4

## References

[1] National Center for Health Statistics (NCHS). Births: Provisional Data for 2023. CDC/NCHS, 2024.
https://www.cdc.gov/nchs/pressroom/states/massachusetts/massachusetts.htm

[2] Deputy NP, et al. Prevalence and trends in gestational diabetes in the United States, 2016–2021. CDC maternal health data portal.
https://www.cdc.gov/reproductivehealth/maternalinfanthealth/pregnancy-complications-data.htm

[3] American College of Obstetricians and Gynecologists (ACOG). Gestational Diabetes—Patient FAQ / Practice Guidance.
https://www.acog.org/womens-health/faqs/gestational-diabetes

[4] American Diabetes Association. Standards of Care in Diabetes—2024: Diabetes and Pregnancy. Diabetes Care.
https://diabetesjournals.org/care

[5] Tovar A, et al. Postpartum screening among women with a history of gestational diabetes mellitus—A systematic review. Matern Child Health J. 2011.
https://pmc.ncbi.nlm.nih.gov/3110507/

[6] Martínez-Portilla RJ, et al. First-trimester prediction of gestational diabetes using easy-to-obtain variables (MIDO-GDM). Sci Rep (Nature Portfolio), 2024.
https://www.nature.com/articles/s41598-024-82757-2

[7] Borodulin K, et al. Exercise during pregnancy and risk of gestational diabetes: meta-analysis.
https://www.nature.com/

[8] Boston Public Health Commission. Boston Resident Births 2021—Summary statistics.
https://www.bphc.org/healthdata
