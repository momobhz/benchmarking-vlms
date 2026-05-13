#!/bin/bash
#SBATCH --account=3dv
#SBATCH --gpus=5060ti:1
#SBATCH --output=/work/courses/3dv/team43/logs/%x-%j.out
#SBATCH --job-name=finetune
#SBATCH --mem=64G
#SBATCH --time=12:00:00

source /work/courses/3dv/team43/benchmarking-vlms/.venv/bin/activate
cd /work/courses/3dv/team43/benchmarking-vlms/finetune

python main.py --config configs/llama_vision.yam