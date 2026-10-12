# DreamStory Project

This repository is the **official implementation** of [DreamStory (TPAMI 2025)](https://ieeexplore.ieee.org/document/11130395) and [IR-Diffusion](https://arxiv.org/abs/2411.19261).  


# Installation

### Create Conda Environment.
- `conda create -n ds python=3.10`
- `conda activate ds`

### Install PyTorch (select the appropriate CUDA version).
- `pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 xformers --index-url https://download.pytorch.org/whl/cu121`

### Install DreamStory
- `git clone https://github.com/hehuiguo/DreamStory`
- `cd DreamStory`
- `pip install -e .`

### Install GroundedSAM
- `git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git`
- `cd Grounded-Segment-Anything`
- `pip install -e segment_anything`
- `pip install --no-build-isolation -e GroundingDINO`

### Install Evaluation Dependencies (only for the DS-500 evaluation)
- `pip install git+https://github.com/openai/CLIP.git`
- Download the aesthetics predictor `sac+logos+ava1-l14-linearMSE.pth` from [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) into `./checkpoints/`. The CLIP and DreamSim weights are downloaded automatically into `./checkpoints/clip` and `./checkpoints/dreamsim`.

# Usage Examples

- `python ./src/DreamStory/pipe/pipe_test.py test --prompts_path="./results/examples/example.json" --output_root="./results/example_debug/" `

The generated image will be saved as ./results/example_debug/output_image_00.png.

### DreamStory Mode (without Isolation/Reposition)
- `bash scripts/run_DreamStory.sh`

### IR-Diffusion Mode (with Isolation and Reposition Attention)
- `bash scripts/run_IR_Diffusion.sh`

### DreamStory Mode with KV-Cache (memory-efficient)
- `bash scripts/run_DreamStory_kv_cache.sh`

### IR-Diffusion Mode with KV-Cache (memory-efficient)
- `bash scripts/run_IR_Diffusion_kv_cache.sh`

> **Note:** The KV-Cache mode splits the UNet forward pass into a reference pass and a scene pass, significantly reducing peak GPU memory usage while producing equivalent results.

# DS-500 Benchmark

Scripts for generating and evaluating the multi-subject subsets of [DS-500](./DS-500/):

| Subset | Meta file | Cases |
|--------|-----------|-------|
| `2-subj` | `DS-500/Synthetic-Cases-400/2-subject_meta.json` | 100 |
| `3-subj` | `DS-500/Synthetic-Cases-400/3-subject_meta.json` | 100 |
| `4-subj` | `DS-500/4-Subject-Stress-Test/4-subject_meta_eval100.json` | 100 |
| `2-subj-interaction` | `DS-500/2-Subject-Interaction/2-subject_meta_eval100.json` | 100 |

```
scripts/DS-500/
├── prepare_data.sh          # convert the DS-500 meta files (run once, shared by both methods)
├── eval.sh                  # example: evaluate one generated subset on its own
├── IR-Diffusion/            # run_2-subj.sh, run_3-subj.sh, run_4-subj.sh, run_2-subj-interaction.sh
└── DreamStory/              # same four scripts with the DreamStory settings
```

### 1. Prepare Data
- `bash scripts/DS-500/prepare_data.sh`

Each case is written to `./results/DS-500/data/<subset>/<case>/final_meta/<case>.json`.

### 2. Generate and Evaluate
- `CUDA_VISIBLE_DEVICES=0 bash scripts/DS-500/IR-Diffusion/run_2-subj.sh`
- `CUDA_VISIBLE_DEVICES=1 bash scripts/DS-500/DreamStory/run_2-subj.sh`

Each script generates every case of the subset with seed 666, then evaluates them. All parameters are written in the script, edit them there for small adjustments. Images are saved to `./results/DS-500/<method>/<subset>/<case>/output_image_00.png`, and the metrics (AES, CLIP-T, DS, D&C-DS, the 4 metrics reported in the paper) to `./results/DS-500/<method>/<subset>/eval/metric.txt`.

> **Note:** Finished cases are skipped, so an interrupted run can simply be relaunched. To re-evaluate an existing subset only, edit and run `bash scripts/DS-500/eval.sh`.

##  Citation
🌟 Support Us! If you find this project useful, please consider giving it a ⭐ to help others discover it!

📖 Cite Us! If this project contributes to your research, we would appreciate it if you could cite our paper:
```bibtex
@article{IR_Diffusion,
  title={Improving Multi-Subject Consistency in Open-Domain Image Generation with Isolation and Reposition Attention},
  author={He, Huiguo and Wang, Qiuyue and Zhou, Yuan and Cai, Yuxuan and Chao, Hongyang and Yin, Jian and Yang, Huan},
  journal={arXiv preprint arXiv:2411.19261},
  year={2024}
}

@article{DreamStory,
  title={{DreamStory}: Open-Domain Story Visualization by LLM-Guided Multi-Subject Consistent Diffusion}, 
  author={He, Huiguo and Yang, Huan and Tuo, Zixi and Zhou, Yuan and Wang, Qiuyue and Zhang, Yuhang and Liu, Zeyu and Huang, Wenhao and Chao, Hongyang and Yin, Jian},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  year={2025},
  volume={47},
  number={12},
  pages={11874-11891},
  doi={10.1109/TPAMI.2025.3600149}
}
```