#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RSVG_PATH="${RSVG_PATH:-data/DIOR_RSVG}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gadan_bilstm_gsbi}"
TOKENIZER_PATH="${TOKENIZER_PATH:-roberta-base}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29515}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
"${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  --use_env \
  train.py \
  --dataset_file rsvg \
  --rsvg_path "${RSVG_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --text_encoder_type bilstm \
  --bilstm_embed_dim 300 \
  --bilstm_hidden_dim 128 \
  --bilstm_num_layers 2 \
  --bilstm_dropout 0.1 \
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
  "$@"
