#!/usr/bin/env bash
# Convert the DS-500 meta files into per-case inference json (results/DS-500/data/<subset>/<case>/final_meta/).
# Shared by IR-Diffusion and DreamStory, only needs to run once. CPU only.
# Usage: bash scripts/DS-500/prepare_data.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

# Synthetic-Cases-400, 2 subjects (100 cases)
python ./src/DreamStory/data/data.py prepare_data \
    --input_meta_file_path ./DS-500/Synthetic-Cases-400/2-subject_meta.json \
    --output_meta_root ./results/DS-500/data/2-subj/ \
    --token_match fuzzy

# Synthetic-Cases-400, 3 subjects (100 cases)
python ./src/DreamStory/data/data.py prepare_data \
    --input_meta_file_path ./DS-500/Synthetic-Cases-400/3-subject_meta.json \
    --output_meta_root ./results/DS-500/data/3-subj/ \
    --token_match fuzzy

# 4-subject stress test (100 cases)
python ./src/DreamStory/data/data.py prepare_data \
    --input_meta_file_path ./DS-500/4-Subject-Stress-Test/4-subject_meta_eval100.json \
    --output_meta_root ./results/DS-500/data/4-subj/ \
    --token_match type

# 2-subject interaction (100 cases)
python ./src/DreamStory/data/data.py prepare_data \
    --input_meta_file_path ./DS-500/2-Subject-Interaction/2-subject_meta_eval100.json \
    --output_meta_root ./results/DS-500/data/2-subj-interaction/ \
    --token_match type
