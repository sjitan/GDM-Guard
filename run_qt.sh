#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate || (python3 -m venv .venv && source .venv/bin/activate)
python -m pip install --upgrade pip >/dev/null
pip install -r requirements.txt >/dev/null
python apps/qt_app.py --video seeds/gdm_sample.mp4 --mirror 1
