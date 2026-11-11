#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -c 'import sys; assert sys.version_info[:2] == (3,10), "Use Python 3.10"'
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
python run_b.py check
