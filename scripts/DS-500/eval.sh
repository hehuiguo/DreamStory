#!/usr/bin/env bash
# Example: evaluate one generated subset on its own (every run_*.sh already evaluates after generation).
# Metrics go to $OUTPUT_ROOT/eval/metric.txt
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/DS-500/eval.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

DATA_ROOT=./results/DS-500/data/2-subj/
OUTPUT_ROOT=./results/DS-500/IR-Diffusion/2-subj/

python ./src/DreamStory/eval/eval_metric.py evaluate \
    --gen_root $OUTPUT_ROOT \
    --data_root $DATA_ROOT \
    --aesthetics_model_path ./checkpoints/sac+logos+ava1-l14-linearMSE.pth \
    --clip_root ./checkpoints/clip \
    --dreamsim_root ./checkpoints/dreamsim \
    --clip_t_model openai/clip-vit-base-patch16
