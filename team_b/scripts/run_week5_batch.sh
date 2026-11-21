#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: bash scripts/run_week5_batch.sh MANIFEST.jsonl OUTPUT.jsonl" >&2
  exit 64
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEAM_B_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST="$(realpath "$1")"
OUTPUT="$(realpath -m "$2")"
CONFIG="${TEAM_B_DIR}/config.week5_stable_v2.json"

cd "${TEAM_B_DIR}"
python run_b_fast.py check --config "${CONFIG}" --manifest "${MANIFEST}"
python run_b_fast.py batch --config "${CONFIG}" --manifest "${MANIFEST}" --output "${OUTPUT}"
python scripts/validate_week5_batch.py --manifest "${MANIFEST}" --output "${OUTPUT}" --config "${CONFIG}"
