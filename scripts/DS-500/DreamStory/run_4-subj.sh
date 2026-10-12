#!/usr/bin/env bash
# DS-500 4-subject stress test (100 cases) with DreamStory: generate every case (finished cases are skipped), then evaluate.
# Run scripts/DS-500/prepare_data.sh first. Metrics go to $OUTPUT_ROOT/eval/metric.txt
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/DS-500/DreamStory/run_4-subj.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."

DATA_ROOT=./results/DS-500/data/4-subj/
OUTPUT_ROOT=./results/DS-500/DreamStory/4-subj/

python ./src/DreamStory/pipe/pipe_test.py test_benchmark \
    --case_root $DATA_ROOT \
    --output_root $OUTPUT_ROOT \
    --seed 666 \
    --attn_processor_name "DreamStoryAttnProcessor" \
    --is_spatial_self_attn False --is_isolate_sa False --sam_step 40 \
    --width 1280 --height 768 --num_inference_steps 50 --guidance_scale 7.0 \
    --start_layer 36 --dropout 0.5 --retry_num 20

python ./src/DreamStory/eval/eval_metric.py evaluate \
    --gen_root $OUTPUT_ROOT \
    --data_root $DATA_ROOT \
    --aesthetics_model_path ./checkpoints/sac+logos+ava1-l14-linearMSE.pth \
    --clip_root ./checkpoints/clip \
    --dreamsim_root ./checkpoints/dreamsim \
    --clip_t_model openai/clip-vit-base-patch16
