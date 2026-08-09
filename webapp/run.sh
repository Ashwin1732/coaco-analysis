#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
exec ../thermal_classifier/.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
