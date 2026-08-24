#!/usr/bin/env bash
set -euo pipefail

ENCODER="${1:-}"
if [[ -z "${ENCODER}" ]]; then
  echo "Usage: scripts/run_ablation.sh {bilstm|bert_tiny|roberta|roberta_frozen} [extra train.py args...]" >&2
  exit 2
fi
shift

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RSVG_PATH="${RSVG_PATH:-data/DIOR_RSVG}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ablation/${ENCODER}_gsbi}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29516}"

EXTRA_ENCODER_ARGS=()
case "${ENCODER}" in
  bilstm)
    TOKENIZER_PATH="${TOKENIZER_PATH:-roberta-base}"
    TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-roberta-base}"
    ;;
  bert_tiny)
    TOKENIZER_PATH="${TOKENIZER_PATH:-google/bert_uncased_L-2_H-128_A-2}"
    TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-google/bert_uncased_L-2_H-128_A-2}"
    ;;
  roberta)
    TOKENIZER_PATH="${TOKENIZER_PATH:-roberta-base}"
    TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-roberta-base}"
    ;;
  roberta_frozen)
    TOKENIZER_PATH="${TOKENIZER_PATH:-roberta-base}"
    TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-roberta-base}"
    EXTRA_ENCODER_ARGS+=(--freeze_text_encoder)
    ;;
  *)
    echo "Unknown encoder: ${ENCODER}" >&2
    exit 2
    ;;
esac

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
"${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  --use_env \
  train.py \
  --dataset_file rsvg \
  --rsvg_path "${RSVG_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --text_encoder_type "${ENCODER}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --text_encoder_path "${TEXT_ENCODER_PATH}" \
  --lr 1e-4 \
  --lr_backbone 5e-5 \
  --lr_text_encoder 1e-5 \
  --weight_decay 5e-4 \
  --gsbi_d_geo 64 \
  --binary \
  --with_box_refine \
  --num_frames 1 \
  --epochs 70 \
  --lr_drop 60 \
  --max_size 640 \
  --hidden_dim 256 \
  --nheads 8 \
  --dec_layers 4 \
  --num_feature_levels 4 \
  --num_queries 10 \
  --backbone resnet50 \
  "${EXTRA_ENCODER_ARGS[@]}" \
  "$@"
