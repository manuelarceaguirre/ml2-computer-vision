#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile experiment.py autoresearch_eval.py
python3 autoresearch_eval.py
