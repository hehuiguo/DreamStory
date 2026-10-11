#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python ./src/DreamStory/pipe/pipe_test.py test \
    --prompts_path="./results/examples/example.json" \
    --output_root="./results/example_IR-Diffusion/" \
    --attn_processor_name "DreamStoryScaledAttnProcessor"
