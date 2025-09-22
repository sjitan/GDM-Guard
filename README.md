# GDMGuard

AI-enabled early pregnancy **GDM risk triage** + postpartum follow-through (Boston pilot demo).  
This repo includes a **synthetic dataset generator**, quick plots, and a baseline modeling scaffold.

## Why this exists
- Standard OGTT happens at **24–28 wks**; many high-risk patients aren’t triaged earlier.
- **Postpartum glucose testing** completion is poor; follow-through matters for T2D prevention.
- Goal: low-friction triage + workflow signals that increase timely testing and follow-up.

## What’s here
- `generate_dataset.py` → creates **synthetic** cohort across **T1, T2, T4** with:
  - core clinical (age, BMI, parity, hx), GWG features, selfie-proxied vitals (rPPG HR stability, neck-proxy, sleep cues), optional PRS flag
  - labels: `GDM_dx`, `pp_glucose_test_done_T4`, `incident_T2D_12m`, `weight_retention_12m`
- `reports/prevalence_missingness.png` → quick prevalence/coverage sanity plot
- `data/GDMGuard_dataset_v2_2.csv` → generated CSV (if you’ve run the script)

## Quickstart

python3 -m venv .venv
source .venv/bin/activate
python -m pip install –upgrade pip -r requirements.txt
python generate_dataset.py
python - <<‘PY’
import pandas as pd,matplotlib
matplotlib.use(“Agg”)
from matplotlib import pyplot as plt
df=pd.read_csv(“data/GDMGuard_dataset_v2_2.csv”)
t12=df[df.stage_code.isin([“T1”,“T2”])]
gprev=t12.groupby(“subj”)[“GDM_dx”].max().mean()
t4c=df[df.stage_code==“T4”][“pp_glucose_test_done_T4”].mean()
plt.figure(figsize=(6,4)); plt.bar([“GDM_prev”,“T4_completion”],[gprev,t4c]); plt.ylim(0,1); plt.tight_layout(); plt.savefig(“reports/prevalence_missingness.png”)
print(“done”)
PY

## Modeling scaffold (outline)
- M0: core clinical features (MIDO-style)
- M1: + GWG features
- M2: + selfie-proxied vitals (rPPG HR/stability, sleepiness cues, neck-proxy)
- M3: + PRS flag (optional)
- Postpartum modules: predict `pp_glucose_test_done_T4` and `incident_T2D_12m` from risk tier + access/adherence proxies

## Repo layout

data/         # generated CSVs (gitignored by default)
reports/      # quick plots + QC text
generate_dataset.py
requirements.txt

## Notes
- **Synthetic data only**; no real PHI.
- Use Python **3.11** for prebuilt wheels on macOS/Apple Silicon.

## References (for context)
- CDC/US natality tables (birth counts; GDM prevalence)
- First-trimester GDM risk models (e.g., MIDO-GDM, Nature portfolio)
- ACOG/ADA guidance on screening windows and postpartum testing
