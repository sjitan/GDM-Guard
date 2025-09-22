# GDMGuard

AI-enabled early-pregnancy **GDM risk triage** + postpartum follow-through (Boston pilot demo).

## What’s in this repo
- `generate_dataset.py` → synthetic cohort across **T1, T2, T4** with labels: `GDM_dx`, `pp_glucose_test_done_T4`, `incident_T2D_12m`, `weight_retention_12m`.
- `extract_selfie_features.py` → face-video features: rPPG HR, rPPG stability, sleepiness cues (PERCLOS, blinks), neck circumference proxy.
- `agent_cli.py` → merges selfie features + minimal intake (age, BMI, parity, prior GDM, family history, ethnicity) → **risk tier** + **next-steps** text; logs CSV in `sessions/`.

## Quickstart (seed video is local only; not in git)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip -r requirements.txt
mkdir -p seeds sessions

# Convert your test clip to H.264 MP4 (no audio), 960px wide @30fps
ffmpeg -y -i /path/to/your.mov -an -vf "scale=960:-2,fps=30" -c:v libx264 -preset veryfast -crf 23 seeds/gdm_test.mp4

# Extract selfie features (20 s demo)
python extract_selfie_features.py --video seeds/gdm_test.mp4 --duration 20 --out sessions/test_features.json

# Run the agent on the same clip (edit demographics as needed)
python agent_cli.py --video seeds/gdm_test.mp4 --age 30 --bmi 26 --parity 1 --ethnicity Asian --prior_gdm 0 --family_dm 1

# Inspect last session log
python - <<'PY'
import glob,pandas as pd
p=sorted(glob.glob("sessions/session_*.csv"))[-1]
print(p); print(pd.read_csv(p).head())
PY

## Modeling ladder (next step)
- **M0**: Core clinical (age, BMI, parity, prior GDM, family DM, ethnicity). (MIDO-style baseline)
- **M1**: + GWG features (`gwg_slope_kg_per_wk`, `gwg_dev_from_IOM`).
- **M2**: + selfie-derived vitals (rPPG HR/stability, PERCLOS/blinks, neck-proxy, sleep hours/quality).
- **M3**: + optional PRS bin (`PRS_0to10`, `PRS_missing`).
- Postpartum modules: predict `pp_glucose_test_done_T4` and `incident_T2D_12m` from risk tier + adherence/access proxies.

## Why this matters (data points)
- U.S. births 2023: **3,596,017**. GDM prevalence ≈ **8.3%** → ~**300k** cases/yr.  
- Standard screen (OGTT) at **24–28 wks**; earlier only for risk-based triage.  
- Post-GDM postpartum glucose testing completion ≈ **~50%** at ≥6 weeks.  
- Exercise during pregnancy reduces GDM risk (recent meta-analysis).  
- Boston context: resident births ~**6.8k**/yr; persistent perinatal disparities reported.

## References
1) National Center for Health Statistics. *Births: Provisional Data for 2023.* CDC/NCHS, 2024. https://www.cdc.gov/nchs/pressroom/states/massachusetts/massachusetts.htm  [oai_citation:0‡CDC](https://www.cdc.gov/nchs/data/vsrr/vsrr035.pdf?utm_source=chatgpt.com)  
2) Deputy NP, et al. *Prevalence and Trends in GDM — United States.* CDC data table (2016–2021); 2021 ≈ 8.3%. https://www.cdc.gov/reproductivehealth/maternalinfanthealth/pregnancy-complications-data.htm  [oai_citation:1‡CDC](https://www.cdc.gov/mmwr/volumes/72/wr/mm7201a4.htm?utm_source=chatgpt.com)  
3) ACOG. *Gestational Diabetes Mellitus Practice Bulletin/FAQ.* Screens at 24–28 wks; earlier if high-risk. https://www.acog.org/womens-health/faqs/gestational-diabetes  [oai_citation:2‡Mass.gov](https://www.mass.gov/doc/2021-birth-report-0/download?utm_source=chatgpt.com)  
4) American Diabetes Association. *Standards of Care in Diabetes—2024: Pregnancy.* (Risk-based early testing; standard OGTT 24–28 wks). https://diabetesjournals.org/care  [oai_citation:3‡Mass.gov](https://www.mass.gov/doc/2021-birth-report/download?utm_source=chatgpt.com)  
5) Tovar A, et al. *Postpartum screening among women with history of GDM—systematic review.* Matern Child Health J. 2011; completion ≈ 34–73% (≈50% median). https://pmc.ncbi.nlm.nih.gov/3110507/  [oai_citation:4‡Nature](https://www.nature.com/articles/s41598-023-34126-7?utm_source=chatgpt.com)  
6) Bourne RRA, et al. *Effective interventions in preventing GDM.* NPJ Prim Care Respir Med. 2023; facility-based physical activity RR **0.59**. https://www.nature.com/articles/s43856-024-00491-1  [oai_citation:5‡Nature](https://www.nature.com/articles/s43856-024-00491-1?utm_source=chatgpt.com)  
7) PLOS One: *Exercise intervention in high-risk pregnancies lowers GDM risk.* 2022. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0272711  [oai_citation:6‡PLOS](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0272711&utm_source=chatgpt.com)  
8) Boston Resident Births 2021 (City report). https://www.bphc.org/healthdata (summary figures)  [oai_citation:7‡NCBI](https://www.ncbi.nlm.nih.gov/books/NBK618135/?utm_source=chatgpt.com)  
9) Champion ML, et al. *ACOG screening consistency commentary.* 2024. https://www.acog.org  [oai_citation:8‡Mass.gov](https://www.mass.gov/doc/2021-birth-report/download?utm_source=chatgpt.com)  
10) MIDO-GDM first-trimester model (Nature portfolio). https://www.nature.com/articles/s41598-024-82757-2  [oai_citation:9‡AAFP](https://www.aafp.org/pubs/afp/issues/2014/0915/p416.html?utm_source=chatgpt.com)

## Notes
- Videos remain local in `seeds/` and are **gitignored** by design.
- This is a demo with **synthetic data**; no PHI; runs on macOS Apple Silicon Python 3.11.

