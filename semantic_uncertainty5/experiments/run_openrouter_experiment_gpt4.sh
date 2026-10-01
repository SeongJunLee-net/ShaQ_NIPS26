#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY before running ShaQ.}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-utils/config.yaml}"
DATASET_PATH="${DATASET_PATH:-data/ambig_qa/dev_light.json}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
MAX_WORKERS="${MAX_WORKERS:-5}"

if [[ ! -f "${DATASET_PATH}" ]]; then
    echo "Dataset not found: ${DATASET_PATH}" >&2
    echo "Prepare it with: python data/prepare_ambigqa.py" >&2
    exit 1
fi

"${PYTHON_BIN}" run.py \
    --config "${CONFIG_PATH}" \
    --dataset "${DATASET_PATH}" \
    --model openai/gpt-4 \
    --max_samples "${MAX_SAMPLES}" \
    --sampling_seed 42 \
    --max_workers "${MAX_WORKERS}" \
    --m 3 \
    --n 5 \
    --answer_temperature 0.7 \
    --replace_temperature 0.9 \
    --evaluate \
    --dump_answer_traces
