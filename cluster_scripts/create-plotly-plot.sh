#!/bin/bash

# ETH student cluster submission script for generating an interactive Plotly
# HTML plot from Robo2VLM UMAP CSV output.
#
# Submit with:
#   sbatch run_robo2vlm_question_plotly.sh

#SBATCH --job-name=robo2vlm-question-plotly
#SBATCH --time=00:30:00
#SBATCH --account=3dv
#SBATCH --output=/work/courses/3dv/team43/logs/%x-%j.out

set -euo pipefail

COURSE_TAG="${COURSE_TAG:-3dv}"
ACCOUNT_TAG="${SLURM_JOB_ACCOUNT:-3dv}"
TEAM_ROOT="${TEAM_ROOT:-/work/courses/${COURSE_TAG}/team43}"
REPO_DIR="${REPO_DIR:-$TEAM_ROOT/benchmarking-vlms}"
VENV_DIR="${VENV_DIR:-$TEAM_ROOT/3dv-env-cu130}"
SCRIPT_PATH="${SCRIPT_PATH:-$REPO_DIR/scripts/plot_robo2vlm_question_clusters.py}"
OUTPUT_DIR="${OUTPUT_DIR:-$TEAM_ROOT/outputs/robo2vlm_umap}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
INPUT_CSV="${INPUT_CSV:-}"
OUTPUT_HTML="${OUTPUT_HTML:-}"
TITLE="${TITLE:-Robo2VLM Question Embeddings Projected with UMAP}"
MARKER_SIZE="${MARKER_SIZE:-10.0}"
MARKER_OPACITY="${MARKER_OPACITY:-0.6}"
INCLUDE_PLOTLYJS="${INCLUDE_PLOTLYJS:-cdn}"

sanitize_name() {
  printf '%s' "$1" | tr -c '[:alnum:]_-' '_'
}

if [ -z "$INPUT_CSV" ]; then
  EMBEDDING_MODEL_TAG="$(sanitize_name "$EMBEDDING_MODEL")"
  INPUT_CSV="$OUTPUT_DIR/${DATASET_SPLIT}_${EMBEDDING_MODEL_TAG}_umap.csv"
fi

echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Course tag: ${COURSE_TAG}"
echo "Account: ${ACCOUNT_TAG}"
echo "Team root: ${TEAM_ROOT}"
echo "Repo: ${REPO_DIR}"
echo "Script: ${SCRIPT_PATH}"
echo "Venv: ${VENV_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Input CSV: ${INPUT_CSV}"
echo "Output HTML: ${OUTPUT_HTML:-default next to CSV}"
echo "Title: ${TITLE}"
echo "Marker size: ${MARKER_SIZE}"
echo "Marker opacity: ${MARKER_OPACITY}"
echo "Plotly JS mode: ${INCLUDE_PLOTLYJS}"

if [ ! -d "$REPO_DIR" ]; then
  echo "Repository directory does not exist: $REPO_DIR" >&2
  exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
  echo "Plotly script does not exist: $SCRIPT_PATH" >&2
  exit 1
fi

if [ ! -f "$INPUT_CSV" ]; then
  echo "Input CSV does not exist: $INPUT_CSV" >&2
  exit 1
fi

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "Virtual environment not found: $VENV_DIR" >&2
  echo "Create it before submitting, for example:" >&2
  echo "  python3 -m venv $VENV_DIR" >&2
  echo "  source $VENV_DIR/bin/activate" >&2
  echo "  pip install --upgrade pip" >&2
  echo "  pip install plotly" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"

cd "$REPO_DIR"

echo "Starting Robo2VLM Plotly export..."

CMD=(
  python3 "$SCRIPT_PATH"
  --input-csv "$INPUT_CSV"
  --title "$TITLE"
  --marker-size "$MARKER_SIZE"
  --marker-opacity "$MARKER_OPACITY"
  --include-plotlyjs "$INCLUDE_PLOTLYJS"
)

if [ -n "$OUTPUT_HTML" ]; then
  CMD+=(--output-html "$OUTPUT_HTML")
fi

printf 'Running command:\n  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "Finished."
if [ -n "$OUTPUT_HTML" ]; then
  echo "Interactive plot: $OUTPUT_HTML"
else
  echo "Interactive plot written next to the CSV."
fi
