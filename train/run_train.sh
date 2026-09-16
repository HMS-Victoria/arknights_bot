#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [ -d /root/miniconda3/bin ]; then
  export PATH="/root/miniconda3/bin:$PATH"
fi

python train/train_lora.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --data_path data/train.jsonl \
  --eval_data data/val.jsonl \
  --output_dir output/凯尔希-lora \
  --max_length 2048 \
  --epochs 3 \
  --batch_size 2 \
  --grad_accum 8 \
  --eval_batch_size 1 \
  --lr 2e-4 \
  --lora_r 32 \
  --lora_alpha 64 \
  --save_steps 100 \
  --eval_steps 200
