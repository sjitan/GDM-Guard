#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
. .venv/bin/activate 2>/dev/null || true
export QT_QPA_PLATFORM=cocoa QT_MAC_WANTS_LAYER=1
PYTHONPATH=. python -u apps/results_viewer.py
