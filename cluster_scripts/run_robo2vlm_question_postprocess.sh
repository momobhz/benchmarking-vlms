#!/bin/bash

# ETH student cluster submission script for deterministic Robo2VLM post-processing.
#
# This script operates on an existing curation output directory. It does not
# call the LLM curator again. Instead it:
# 1. Moves known goal-state questions from spatial_reasoning to neither.
# 2. Assigns deterministic subcategories from question-text signal phrases.
#
# Submit with:
#   sbatch run_robo2vlm_question_postprocess.sh

#SBATCH --job-name=robo2vlm-question-postprocess
#SBATCH --time=00:20:00
#SBATCH --account=3dv
#SBATCH --output=/work/courses/3dv/team43/logs/%x-%j.out

set -euo pipefail

COURSE_TAG="${COURSE_TAG:-3dv}"
ACCOUNT_TAG="${SLURM_JOB_ACCOUNT:-3dv}"
TEAM_ROOT="${TEAM_ROOT:-/work/courses/${COURSE_TAG}/team43}"
REPO_DIR="${REPO_DIR:-$TEAM_ROOT/benchmarking-vlms}"
VENV_DIR="${VENV_DIR:-$TEAM_ROOT/3dv-env-cu130}"
OUTPUT_DIR="${OUTPUT_DIR:-$TEAM_ROOT/outputs/robo2vlm_spatial_affordance}"
RELABEL_SCRIPT_PATH="${RELABEL_SCRIPT_PATH:-$REPO_DIR/scripts/relabel_robo2vlm_goal_state_questions.py}"
SUBCATEGORIZE_SCRIPT_PATH="${SUBCATEGORIZE_SCRIPT_PATH:-$REPO_DIR/scripts/subcategorize_robo2vlm_questions.py}"
MATCH_PHRASE="${MATCH_PHRASE:-Which configuration shows the goal state that the robot should achieve?}"
FROM_LABEL="${FROM_LABEL:-spatial_reasoning}"
TO_LABEL="${TO_LABEL:-neither}"

echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Course tag: ${COURSE_TAG}"
echo "Account: ${ACCOUNT_TAG}"
echo "Team root: ${TEAM_ROOT}"
echo "Repo: ${REPO_DIR}"
echo "Venv: ${VENV_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Relabel script: ${RELABEL_SCRIPT_PATH}"
echo "Subcategorize script: ${SUBCATEGORIZE_SCRIPT_PATH}"

if [ ! -d "$REPO_DIR" ]; then
  echo "Repository directory does not exist: $REPO_DIR" >&2
  exit 1
fi

if [ ! -f "$RELABEL_SCRIPT_PATH" ]; then
  echo "Relabel script does not exist: $RELABEL_SCRIPT_PATH" >&2
  exit 1
fi

if [ ! -f "$SUBCATEGORIZE_SCRIPT_PATH" ]; then
  echo "Subcategorize script does not exist: $SUBCATEGORIZE_SCRIPT_PATH" >&2
  exit 1
fi

if [ ! -f "$OUTPUT_DIR/curation_results.jsonl" ]; then
  echo "Expected curation results were not found: $OUTPUT_DIR/curation_results.jsonl" >&2
  exit 1
fi

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "Virtual environment not found: $VENV_DIR" >&2
  echo "Create it before submitting, for example:" >&2
  echo "  python3 -m venv $VENV_DIR" >&2
  echo "  source $VENV_DIR/bin/activate" >&2
  echo "  pip install --upgrade pip" >&2
  echo "  pip install datasets sentence-transformers umap-learn matplotlib scikit-learn tqdm plotly" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"

cd "$REPO_DIR"
export PYTHONUNBUFFERED=1

RELABEL_CMD=(
  python3 "$RELABEL_SCRIPT_PATH"
  --output-dir "$OUTPUT_DIR"
  --match-phrase "$MATCH_PHRASE"
  --from-label "$FROM_LABEL"
  --to-label "$TO_LABEL"
)

SUBCATEGORIZE_CMD=(
  python3 "$SUBCATEGORIZE_SCRIPT_PATH"
  --output-dir "$OUTPUT_DIR"
)

echo "Running relabel pass..."
printf '  %q' "${RELABEL_CMD[@]}"
printf '\n'
"${RELABEL_CMD[@]}"

echo "Running subcategory pass..."
printf '  %q' "${SUBCATEGORIZE_CMD[@]}"
printf '\n'
"${SUBCATEGORIZE_CMD[@]}"

echo "Finished."
echo "Updated curation results: $OUTPUT_DIR/curation_results.jsonl"
echo "Updated summary: $OUTPUT_DIR/summary.json"
