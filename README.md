# GDMGuard
AI-enabled early-pregnancy **GDM risk triage** + postpartum follow-through (Boston pilot demo).
[https://youtu.be/P_uuaGIJ_EQ](https://youtu.be/FO4qJDFnGwM)

## Quick run (local)
**Demo (CLI, writes features + recommendation)**
- macOS/Linux/WSL: `./run_demo.sh`
- Windows (PowerShell): `bash run_demo.sh` (from Git Bash/MSYS2)

**Qt HUD (overlay + live telemetry)**
- macOS/Linux/WSL: `./run_qt.sh`
- Windows (PowerShell): `bash run_qt.sh`

Outputs:
- `sessions/vis_metrics.json` (PERCLOS, blinks, neck_norm)
- `sessions/session_*.csv` (agent log)
- `reports/` (snapshot figure if generated)

## What’s here
- `apps/qt_app.py` – Qt HUD visualizer for the selfie metrics
- `apps/cli_visualize.py` – headless selfie pass (JSON metrics)
- `apps/extract_selfie_features.py` – rPPG HR/stability, PERCLOS/blinks, neck proxy
- `apps/agent_cli.py` – merges selfie + intake → risk tier + next steps
- `lib/rppg_utils.py` – rPPG/ROI and face-mesh helpers
- `scripts/generate_dataset.py` – synthetic cohort across **T1, T2, T3, T4** with labels:
  `GDM_dx`, `pp_glucose_test_done_T4`, `incident_T2D_12m`, `weight_retention_12m`

## Model ladder (triage stack)
- **M0:** Core clinical (age, BMI, parity, prior GDM, family DM, ethnicity)
- **M1:** + **GWG** features (gwg_slope_kg_per_wk, gwg_dev_from_IOM) aggregated **T1–T3**
- **M2:** + **Selfie vitals** (rPPG HR/stability proxy, PERCLOS/blinks, neck_norm, optional sleep)
- **M3:** + **PRS** bin (optional: `PRS_0to10`, `PRS_missing`)
- Downstream: predict `pp_glucose_test_done_T4`, `incident_T2D_12m`

## Stages captured
- **T1:** 6–13 wks • **T2:** 18–26 wks • **T3:** 28–32 wks • **T4:** ~12 wks postpartum

## Open-source stack
OpenCV • MediaPipe Face Mesh • NumPy • Pandas • Matplotlib • PySide6 (Qt) • Python 3.11

## Notes
- Seed video is included at `seeds/gdm_sample.mp4` for instant runs.
- This is a **synthetic/demo** pipeline; no PHI; not a medical device.

## References (selected)
1. NCHS. Births: Provisional Data for 2023. CDC/NCHS, 2024.
2. CDC. Gestational Diabetes trends (2016–2021).
3. ACOG. Gestational Diabetes—screening 24–28 wks; risk-based earlier.
4. ADA. Standards of Care in Diabetes—2024: Pregnancy.
5. Tovar A, et al. Postpartum screening after GDM—systematic review. Matern Child Health J. 2011.
6. Martínez-Portilla RJ, et al. First-trimester GDM prediction with easy-to-obtain variables (MIDO-GDM). Sci Rep. 2024.

## Selfie vitals (what you’ll see)
- **rPPG** (face-based HR & stability proxy) in the Qt HUD and selfie extractor
- **Fatigue cues**: PERCLOS, blinks/min
- **Neck proxy**: neck_norm metric (edema/neck circumference proxy)

