#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ -d ".venv" ]; then source .venv/bin/activate; fi
export QT_QPA_PLATFORM=cocoa
export PYTHONPATH=.
python apps/qt_smoke.py --video "seeds/gdm_sample.mp4" --mirror 1
