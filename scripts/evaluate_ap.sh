#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

: "${CHECKPOINT:?Set CHECKPOINT to a trained .pth file}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RSVG_PATH="${RSVG_PATH:-data/DIOR_RSVG}"
TOKENIZER_PATH="${TOKENIZER_PATH:-roberta-base}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" evaluate_ap.py \
  --dataset_file rsvg \
  --rsvg_path "${RSVG_PATH}" \
  --resume "${CHECKPOINT}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --max_size 640 \
  --num_queries 10 \
  --with_box_refine \
  --binary \
  --backbone resnet50 \
  "$@"
