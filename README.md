# Installation

### Create Conda Environment.
- `conda create -n ds python=3.10`
- `conda activate ds`

### Install PyTorch (select the appropriate CUDA version).
- `pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 xformers --index-url https://download.pytorch.org/whl/cu121`

### Install DreamStory
- `pip install -e .`

### Install GroundedSAM
- `git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git`
- `cd Grounded-Segment-Anything`
- `pip install -e segment_anything`
- `pip install -e GroundingDINO`

# Usage Examples

- `python ./src/DreamStory/pipe/pipe_test.py test --prompts_path="./results/examples/example.json" --output_root="./results/example_debug/" `

The generated image will be saved as ./results/example_debug/output_image_00.png.

<!-- # Issues

### libGL.so.1
- `apt install libgl1`

### libgthread-2.0.so.0
- `apt install libglib2.0-0` -->