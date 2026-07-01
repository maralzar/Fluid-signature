#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

python scripts/01_prepare_data.py
python scripts/02_run_fluid.py --graph lubm1
python scripts/02_run_fluid.py --graph lubm2
python scripts/03_build_rbt.py --graph lubm1
python scripts/03_build_rbt.py --graph lubm2
python scripts/04_train_student.py
python scripts/05_eval_transfer.py
pytest tests/ -q
