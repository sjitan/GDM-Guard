# GDMGuard

AI-enabled early pregnancy **GDM risk triage** + postpartum follow-through (Boston pilot demo).

## Stages captured
- **T1:** 6–13 wks
- **T2:** 18–26 wks
- **T3:** 28–32 wks
- **T4:** ~12 wks postpartum

## Model stack (triage)
- **M0:** Core clinical (age, BMI, parity, prior GDM, family DM, ethnicity)
- **M1:** + **GWG** features (gwg_slope_kg_per_wk, gwg_dev_from_IOM) aggregated across **T1–T3**
- **M2:** + **Selfie vitals** (face rPPG HR/stability, fatigue/sleepiness cues: PERCLOS/blinks, **neck circumference proxy**) ± sleep hours/quality
- **M3:** + **PRS** bin (PRS_0to10, PRS_missing)
- Output: **risk_tier** (low/med/high + confidence) → **Next-steps** (e.g., early OGTT 16–20w; nutrition consult; auto-enroll T4 test)

## What’s in this repo
- `generate_dataset.py` → synthetic cohort across **T1/T2/T3/T4** with labels: `GDM_dx`, `pp_glucose_test_done_T4`, `incident_T2D_12m`, `weight_retention_12m`
- `extract_selfie_features.py` → face-video features (rPPG HR, rPPG stability, PERCLOS, blinks, neck proxy)
- `agent_cli.py` → merges selfie features + minimal intake → **risk_tier** + **next_steps**; logs CSV in `sessions/`
- `visualize_selfie.py` → overlay demo (face mesh + on-screen PERCLOS/blinks/**neck_norm**), writes `sessions/vis_metrics.json`
- `data/` (generated), `reports/` (plots), `seeds/` (local demo clips), `sessions/` (logs)

## Quickstart
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip -r requirements.txt

## Seed demo (local file, gitignored)
python extract_selfie_features.py --video seeds/gdm_test.mp4 --duration 20 --out sessions/test_features.json
python agent_cli.py --video seeds/gdm_test.mp4 --age 30 --bmi 26 --parity 1 --ethnicity Asian --prior_gdm 0 --family_dm 1

## Visualizer (overlay + JSON)
python visualize_selfie.py --video seeds/gdm_test.mp4 --duration 20 --out_json sessions/vis_metrics.json

## Outputs
- **Sessions CSV**: `sessions/session_*.csv` (includes `prob`,`risk_tier`,`next_steps`, and `neck_norm` if present)
- **Features JSON**: `sessions/test_features.json`
- **Visualizer JSON**: `sessions/vis_metrics.json` (keys: `perclos`,`blink_rate`,`neck_norm`)
- **Synthetic dataset**: `data/GDMGuard_dataset_v3_T1T2T3T4.csv`
- **QC plot**: `reports/prevalence_missingness_v3.png`

## References
- U.S. births 2023 (3.596M): CDC/NCHS Provisional 2024. https://www.cdc.gov/nchs/pressroom/states/massachusetts/massachusetts.htm
- GDM prevalence ≈ 8.3% (2021): CDC table. https://www.cdc.gov/reproductivehealth/maternalinfanthealth/pregnancy-complications-data.htm
- ACOG: screen 24–28 wks; risk-based earlier. https://www.acog.org/womens-health/faqs/gestational-diabetes
- ADA Standards of Care—Pregnancy. https://diabetesjournals.org/care
- Postpartum glucose testing completion ≈ ~50%: Tovar A. 2011. https://pmc.ncbi.nlm.nih.gov/3110507/
- Exercise lowers GDM risk (meta-analysis). https://www.nature.com/articles/s43856-024-00491-1
- Boston births (residents). https://www.bphc.org/healthdata
- First-trimester model (MIDO-GDM). https://www.nature.com/articles/s41598-024-82757-2

## Notes
- Synthetic demo only; no PHI. macOS Apple Silicon, Python 3.11.
