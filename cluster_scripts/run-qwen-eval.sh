#!/bin/bash

# ETH student cluster submission script for Robo2VLM benchmark evaluation.
#
# Submit with:
#   sbatch run_qwen_eval.sh

#SBATCH --job-name=robo2vlm-qwen-eval
#SBATCH --time=05:00:00
#SBATCH --account=3dv
#SBATCH --gpus=5060ti:1
#SBATCH --output=/work/courses/3dv/team43/logs/%x-%j.out

CUDA_MODULE="${CUDA_MODULE:-cuda/13.0}"
. /etc/profile.d/modules.sh
module add "$CUDA_MODULE"

set -euo pipefail

COURSE_TAG="${COURSE_TAG:-3dv}"
ACCOUNT_TAG="${SLURM_JOB_ACCOUNT:-3dv}"
TEAM_ROOT="${TEAM_ROOT:-/work/courses/${COURSE_TAG}/team43}"
REPO_DIR="${REPO_DIR:-$TEAM_ROOT/benchmarking-vlms}"
VENV_DIR="${VENV_DIR:-$TEAM_ROOT/3dv-env-cu130}"
SCRATCH_BASE="${SCRATCH_BASE:-$TEAM_ROOT/robo2vlm_qwen_eval_cache}"
BENCHMARK_DIR="${BENCHMARK_DIR:-$REPO_DIR/benchmark}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$BENCHMARK_DIR/evaluation.py}"
DATASET_NAME="${DATASET_NAME:-keplerccc/Robo2VLM-1}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
TP_SIZE="${TP_SIZE:-1}"

echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Course tag: ${COURSE_TAG}"
echo "Account: ${ACCOUNT_TAG}"
echo "Team root: ${TEAM_ROOT}"
echo "Repo: ${REPO_DIR}"
echo "Benchmark dir: ${BENCHMARK_DIR}"
echo "Eval script: ${EVAL_SCRIPT}"
echo "Venv: ${VENV_DIR}"
echo "Scratch: ${SCRATCH_BASE}"
echo "CUDA module: ${CUDA_MODULE}"
echo "Model: ${MODEL_ID}"
echo "Dataset: ${DATASET_NAME} (${DATASET_SPLIT})"
echo "Max samples: ${MAX_SAMPLES}"
echo "Tensor parallel size: ${TP_SIZE}"

mkdir -p "$SCRATCH_BASE"/{hf,transformers,datasets,vllm,tmp}

export HF_HOME="$SCRATCH_BASE/hf"
export TRANSFORMERS_CACHE="$SCRATCH_BASE/transformers"
export HF_DATASETS_CACHE="$SCRATCH_BASE/datasets"
export VLLM_CACHE_ROOT="$SCRATCH_BASE/vllm"
export TMPDIR="$SCRATCH_BASE/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [ ! -d "$REPO_DIR" ]; then
  echo "Repository directory does not exist: $REPO_DIR" >&2
  exit 1
fi

if [ ! -d "$BENCHMARK_DIR" ]; then
  echo "Benchmark directory does not exist: $BENCHMARK_DIR" >&2
  exit 1
fi

if [ ! -f "$EVAL_SCRIPT" ]; then
  echo "Evaluation script does not exist: $EVAL_SCRIPT" >&2
  exit 1
fi

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "Virtual environment not found: $VENV_DIR" >&2
  echo "Create it before submitting, for example:" >&2
  echo "  python3 -m venv $VENV_DIR" >&2
  echo "  source $VENV_DIR/bin/activate" >&2
  echo "  pip install --upgrade pip" >&2
  echo "  pip install torch torchvision transformers datasets pillow tqdm numpy huggingface_hub vllm" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"

cd "$BENCHMARK_DIR"

echo "Starting benchmark evaluation..."
module list
nvidia-smi

python3 - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count visible to torch: {torch.cuda.device_count()}")
PY

python3 "$EVAL_SCRIPT" \
  --models "$MODEL_ID" \
  --dataset "$DATASET_NAME" \
  --split "$DATASET_SPLIT" \
  --max_samples "$MAX_SAMPLES" \
  --tensor_parallel_size "$TP_SIZE" \
  --batch_size 1

echo "Finished. Results are under:"
echo "  $REPO_DIR/benchmark/results"

