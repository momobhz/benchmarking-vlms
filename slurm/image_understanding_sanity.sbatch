#!/bin/bash

# Run image-understanding sanity diagnostics for an existing evaluation result.
#
# Submit with:
#   sbatch slurm/image_understanding_sanity.sbatch \
#     runs/eval/qwen25_3b_spatial_cot_200_t0/results/qwen25_3b_spatial_cot_200_t0_Qwen2.5-VL-3B-Instruct_spatial_cot_20260513_170901.json \
#     Qwen/Qwen2.5-VL-3B-Instruct \
#     20
#
# Extra arguments after MAX_EXAMPLES are forwarded to the Python script, e.g.:
#   sbatch slurm/image_understanding_sanity.sbatch <result.json> <model-id> 20 \
#     --probe overlay_describe --save-probe-images

#SBATCH --job-name=vlm-img-sanity
#SBATCH --time=10:00:00
#SBATCH --account=3dv
#SBATCH --gpus=5060ti:1
#SBATCH --ntasks=1
#SBATCH --output=/work/courses/3dv/team43/logs/%x-%j.out

. /etc/profile.d/modules.sh
module add "${CUDA_MODULE:-cuda/13.0}"
export VLM_BENCH_MODULES_INITIALIZED=1

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
COURSE_TAG="${COURSE_TAG:-3dv}"
TEAM_ROOT="${TEAM_ROOT:-/work/courses/${COURSE_TAG}/team43}"
REPO_DIR="${REPO_DIR:-$SUBMIT_DIR}"
VENV_DIR="${VENV_DIR:-$TEAM_ROOT/3dv-env-cu130}"
SCRATCH_BASE="${SCRATCH_BASE:-$TEAM_ROOT/vlm_bench_cache}"
JOB_TMPDIR="${TMPDIR:-$SCRATCH_BASE/tmp}"

DEFAULT_RESULT_JSON="runs/eval/qwen25_3b_spatial_cot_200_t0/results/qwen25_3b_spatial_cot_200_t0_Qwen2.5-VL-3B-Instruct_spatial_cot_20260513_170901.json"
RESULT_JSON="${RESULT_JSON:-${1:-$DEFAULT_RESULT_JSON}}"
MODEL_ID="${MODEL_ID:-${2:-Qwen/Qwen2.5-VL-3B-Instruct}}"
MAX_EXAMPLES="${MAX_EXAMPLES:-${3:-20}}"

if [ "$#" -ge 1 ]; then
  shift
fi
if [ "$#" -ge 1 ]; then
  shift
fi
if [ "$#" -ge 1 ]; then
  shift
fi
FORWARDED_ARGS=("$@")

if [ "${RESULT_JSON#/}" = "$RESULT_JSON" ]; then
  RESULT_JSON="$REPO_DIR/$RESULT_JSON"
fi

echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Repo: $REPO_DIR"
echo "Venv: $VENV_DIR"
echo "Scratch: $SCRATCH_BASE"
echo "Job tmp: $JOB_TMPDIR"
echo "Result JSON: $RESULT_JSON"
echo "Model: $MODEL_ID"
echo "Max examples: $MAX_EXAMPLES"

if [ ! -d "$REPO_DIR" ]; then
  echo "Repository directory does not exist: $REPO_DIR" >&2
  exit 1
fi

if [ ! -f "$RESULT_JSON" ]; then
  echo "Result JSON does not exist: $RESULT_JSON" >&2
  exit 1
fi

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "Virtual environment not found: $VENV_DIR" >&2
  exit 1
fi

mkdir -p "$SCRATCH_BASE"/{hf,transformers,datasets,vllm,matplotlib}
mkdir -p "$JOB_TMPDIR"

export HF_HOME="$SCRATCH_BASE/hf"
export TRANSFORMERS_CACHE="$SCRATCH_BASE/transformers"
export HF_DATASETS_CACHE="$SCRATCH_BASE/datasets"
export VLLM_CACHE_ROOT="$SCRATCH_BASE/vllm"
export TMPDIR="$JOB_TMPDIR"
export MPLCONFIGDIR="$SCRATCH_BASE/matplotlib"
export COURSE_TAG
export TEAM_ROOT
export REPO_DIR
export VENV_DIR
export SCRATCH_BASE
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/benchmark:${PYTHONPATH:-}"

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

module list
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

python3 - <<'PY'
try:
    import torch
except Exception as exc:
    print(f"PyTorch import failed: {exc}")
else:
    print(f"PyTorch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count visible to torch: {torch.cuda.device_count()}")
PY

COMMAND=(
  python3 scripts/run_image_understanding_sanity.py
  --result-json "$RESULT_JSON"
  --model "$MODEL_ID"
  --max-examples "$MAX_EXAMPLES"
  --max-tokens "${MAX_TOKENS:-512}"
  --batch-size "${BATCH_SIZE:-1}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
  --output-dir "${OUTPUT_DIR:-runs/diagnostics/image_understanding}"
  --sanity-seed "${SANITY_SEED:-11}"
)

if [ -n "${RUN_NAME:-}" ]; then
  COMMAND+=(--run-name "$RUN_NAME")
fi

if [ "${SAVE_PROBE_IMAGES:-0}" = "1" ]; then
  COMMAND+=(--save-probe-images)
fi

COMMAND+=("${FORWARDED_ARGS[@]}")

printf 'Running diagnostic command:\n  '
printf '%q ' "${COMMAND[@]}"
printf '\n'

"${COMMAND[@]}"
