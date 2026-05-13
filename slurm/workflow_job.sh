#!/bin/bash

set -euo pipefail

WORKFLOW_COMMAND="${WORKFLOW_COMMAND:-eval}"
CONFIG_PATH="${CONFIG_PATH:-${1:-}}"
COURSE_TAG="${COURSE_TAG:-3dv}"
TEAM_ROOT="${TEAM_ROOT:-/work/courses/${COURSE_TAG}/team43}"
REPO_DIR="${REPO_DIR:-$TEAM_ROOT/benchmarking-vlms}"
VENV_DIR="${VENV_DIR:-$TEAM_ROOT/3dv-env-cu130}"
SCRATCH_BASE="${SCRATCH_BASE:-$TEAM_ROOT/vlm_bench_cache}"
CUDA_MODULE="${CUDA_MODULE:-cuda/13.0}"
DRY_RUN="${DRY_RUN:-0}"
JOB_TMPDIR="${TMPDIR:-$SCRATCH_BASE/tmp}"

if [ -z "$CONFIG_PATH" ]; then
  echo "CONFIG_PATH is required. Pass it as the first argument or set CONFIG_PATH." >&2
  exit 1
fi

if [ "${CONFIG_PATH#/}" = "$CONFIG_PATH" ]; then
  CONFIG_PATH="$REPO_DIR/$CONFIG_PATH"
fi

if [ "${VLM_BENCH_MODULES_INITIALIZED:-0}" != "1" ]; then
  if ! command -v module >/dev/null 2>&1; then
    if [ ! -f /etc/profile.d/modules.sh ]; then
      echo "The modules initialization script was not found at /etc/profile.d/modules.sh." >&2
      echo "Run this workflow through Slurm on the ETH student cluster, or load CUDA manually." >&2
      exit 1
    fi
    . /etc/profile.d/modules.sh
  fi
  module add "$CUDA_MODULE"
fi

echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Workflow: $WORKFLOW_COMMAND"
echo "Config: $CONFIG_PATH"
echo "Team root: $TEAM_ROOT"
echo "Repo: $REPO_DIR"
echo "Venv: $VENV_DIR"
echo "Scratch: $SCRATCH_BASE"
echo "Job tmp: $JOB_TMPDIR"
echo "CUDA module: $CUDA_MODULE"

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
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

if [ ! -d "$REPO_DIR" ]; then
  echo "Repository directory does not exist: $REPO_DIR" >&2
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Config file does not exist: $CONFIG_PATH" >&2
  exit 1
fi

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "Virtual environment not found: $VENV_DIR" >&2
  echo "Create it before submitting and install the workflow dependencies." >&2
  exit 1
fi

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

COMMAND=(python3 -m vlm_bench "$WORKFLOW_COMMAND" --config "$CONFIG_PATH")
if [ "$DRY_RUN" = "1" ]; then
  COMMAND+=(--dry-run)
fi

printf 'Running workflow command:\n  %q' "${COMMAND[@]}"
printf '\n'

"${COMMAND[@]}"
