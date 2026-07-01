#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
CONFIG="${1:-configs/phase2.yaml}"

python scripts/01_prepare_data.py --config "$CONFIG"
python scripts/02_run_fluid.py --config "$CONFIG" --graph lubm1
python scripts/02_run_fluid.py --config "$CONFIG" --graph lubm2
python scripts/03_build_rbt.py --config "$CONFIG" --graph lubm1
python scripts/03_build_rbt.py --config "$CONFIG" --graph lubm2
python scripts/04_train_student.py --config "$CONFIG"
python scripts/05_eval_transfer.py --config "$CONFIG"
python scripts/06_run_ablations.py --config "$CONFIG"
pytest tests/ -q
