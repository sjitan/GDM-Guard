#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate || (python3 -m venv .venv && source .venv/bin/activate)
python -m pip install --upgrade pip >/dev/null
pip install -r requirements.txt >/dev/null
python apps/cli_visualize.py --video seeds/gdm_sample.mp4 --duration 10 --mirror 1 --out_json sessions/vis_metrics.json
python apps/extract_selfie_features.py --video seeds/gdm_sample.mp4 --duration 10 --out sessions/seed_features.json
python apps/agent_cli.py --video seeds/gdm_sample.mp4 --age 30 --bmi 26 --parity 1 --ethnicity Asian --prior_gdm 1 --family_dm 1
