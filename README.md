# GDMGuard

AI-enabled early-pregnancy **GDM risk triage** + postpartum follow-through (Boston pilot demo).

## Quick Qt visual demo (copy–paste)
python qt_visual_demo.py --video seeds/gdm_sample.mp4 --mirror 1 --hud_scale 1.3

## 60s CLI demo
python visualize_selfie.py --video seeds/gdm_sample.mp4 --duration 12 --mirror 1 --out_json sessions/vis_metrics.json
python extract_selfie_features.py --video seeds/gdm_sample.mp4 --duration 12 --out sessions/seed_features.json
python agent_cli.py --video seeds/gdm_sample.mp4 --age 30 --bmi 26 --parity 1 --ethnicity Asian --prior_gdm 1 --family_dm 1

# GDMGuard

AI-enabled early-pregnancy **GDM risk triage** + postpartum follow-through (Boston pilot demo).

## Quick demo (copy & paste)
**Python 3.11 required.** Works on macOS, Linux, Windows.

### macOS / Linux
    git clone https://github.com/sjitan/GDM-Guard.git
    cd GDM-Guard
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip -r requirements.txt
    mkdir -p sessions reports
    python visualize_selfie.py --video seeds/gdm_sample.mp4 --duration 10 --mirror 1 --out_json sessions/vis_metrics.json
    python extract_selfie_features.py --video seeds/gdm_sample.mp4 --duration 10 --out sessions/seed_features.json
    python agent_cli.py --video seeds/gdm_sample.mp4 --age 30 --bmi 26 --parity 1 --ethnicity Asian --prior_gdm 1 --family_dm 1

### Windows (PowerShell)
    git clone https://github.com/sjitan/GDM-Guard.git
    cd GDM-Guard
    python -m venv .venv
    .venv\Scripts\Activate
    python -m pip install --upgrade pip -r requirements.txt
    mkdir sessions, reports
    python visualize_selfie.py --video seeds\gdm_sample.mp4 --duration 10 --mirror 1 --out_json sessions\vis_metrics.json
    python extract_selfie_features.py --video seeds\gdm_sample.mp4 --duration 10 --out sessions\seed_features.json
    python agent_cli.py --video seeds\gdm_sample.mp4 --age 30 --bmi 26 --parity 1 --ethnicity Asian --prior_gdm 1 --family_dm 1

**Outputs**
- sessions/vis_metrics.json — PERCLOS, blink rate, neck_norm
- sessions/seed_features.json — selfie features
- sessions/session_*.csv — agent log
- reports/snapshot_metrics.png — prevalence + T4 completion

## What’s in this repo
- generate_dataset.py — synthetic cohort across **T1, T2, T3, T4**
- extract_selfie_features.py — rPPG HR/stability, PERCLOS/blinks, neck proxy
- visualize_selfie.py — overlay HUD for selfie metrics
- agent_cli.py — merges features + intake → risk tier + next steps
- seeds/gdm_sample.mp4 — lightweight demo clip

## Model ladder (triage stack)
- M0: Core clinical (age, BMI, parity, prior GDM, family DM, ethnicity)
- M1: + gestational weight gain (T1-3 slope, deviation from IOM)
- M2: + selfie vitals (rPPG HR/stability proxy, PERCLOS/blinks, neck_norm, sleep cues)
- M3: + PRS bin (optional)
- Downstream: predict postpartum screen completion (T4) + incident T2D (12m)

## Stages captured
- T1: 6–13 wks
- T2: 18–26 wks
- T3: 28–32 wks
- T4: ~12 wks postpartum

## One-liner snapshot
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
- OpenCV, MediaPipe Face Mesh; NumPy, Pandas, Matplotlib
- scikit-learn (baseline); Python 3.11
- FFmpeg (optional, for .mov → .mp4)

## References
1. CDC/NCHS. Births: Provisional Data for 2023 — https://www.cdc.gov/nchs/pressroom/states/massachusetts/massachusetts.htm
2. CDC. GDM prevalence data (2016–2021) — https://www.cdc.gov/reproductivehealth/maternalinfanthealth/pregnancy-complications-data.htm
3. ACOG. Gestational Diabetes Mellitus FAQ — https://www.acog.org/womens-health/faqs/gestational-diabetes
4. ADA. Standards of Care in Diabetes—2024: Pregnancy — https://diabetesjournals.org/care
5. Tovar A, et al. Postpartum screening after GDM — https://pmc.ncbi.nlm.nih.gov/3110507/
6. Martínez-Portilla RJ, et al. MIDO-GDM (Sci Rep, 2024) — https://www.nature.com/articles/s41598-024-82757-2
7. Boston Public Health Commission. Resident Births 2021 — https://www.bphc.org/healthdata
