#!/bin/bash

# ETH student cluster submission script for Robo2VLM question embedding +
# UMAP visualization.
#
# Submit with:
#   sbatch run_robo2vlm_question_umap.sh

#SBATCH --job-name=robo2vlm-question-umap
#SBATCH --time=04:00:00
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
SCRATCH_BASE="${SCRATCH_BASE:-$TEAM_ROOT/robo2vlm_question_pipeline_cache}"

CURATION_SCRIPT_PATH="${CURATION_SCRIPT_PATH:-$REPO_DIR/scripts/recategorize_robo2vlm.py}"
UMAP_SCRIPT_PATH="${UMAP_SCRIPT_PATH:-$REPO_DIR/scripts/visualize_robo2vlm_question_clusters.py}"
PLOTLY_SCRIPT_PATH="${PLOTLY_SCRIPT_PATH:-$REPO_DIR/scripts/plot_robo2vlm_question_clusters.py}"

DATASET_NAME="${DATASET_NAME:-keplerccc/Robo2VLM-1}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

CURATION_OUTPUT_DIR="${CURATION_OUTPUT_DIR:-$TEAM_ROOT/outputs/robo2vlm_spatial_affordance}"
CURATION_RESULTS_PATH="${CURATION_RESULTS_PATH:-$CURATION_OUTPUT_DIR/curation_results.jsonl}"
CURATION_MODEL="${CURATION_MODEL:-gpt-4.1-mini}"
PROMPT_VERSION="${PROMPT_VERSION:-spatial_affordance_v1}"
API_KEY_ENV="${API_KEY_ENV:-OPENAI_API_KEY}"
BASE_URL="${BASE_URL:-https://api.openai.com/v1}"
CURATION_TIMEOUT_SECONDS="${CURATION_TIMEOUT_SECONDS:-90}"
CURATION_MAX_RETRIES="${CURATION_MAX_RETRIES:-4}"
CURATION_RETRY_BACKOFF_SECONDS="${CURATION_RETRY_BACKOFF_SECONDS:-2.0}"
CURATION_BATCH_SIZE="${CURATION_BATCH_SIZE:-25}"
CURATION_START_INDEX="${CURATION_START_INDEX:-0}"
CURATION_END_INDEX="${CURATION_END_INDEX:-}"
CURATION_RESUME="${CURATION_RESUME:-1}"
CURATION_OVERWRITE="${CURATION_OVERWRITE:-1}"
CURATION_STREAMING="${CURATION_STREAMING:-1}"

UMAP_OUTPUT_DIR="${UMAP_OUTPUT_DIR:-$TEAM_ROOT/outputs/robo2vlm_umap}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
UMAP_BATCH_SIZE="${UMAP_BATCH_SIZE:-512}"
DEVICE="${DEVICE:-cuda}"
UMAP_STREAMING="${UMAP_STREAMING:-1}"
PCA_COMPONENTS="${PCA_COMPONENTS:-50}"
UMAP_NEIGHBORS="${UMAP_NEIGHBORS:-30}"
UMAP_MIN_DIST="${UMAP_MIN_DIST:-0.05}"
UMAP_METRIC="${UMAP_METRIC:-cosine}"
RANDOM_STATE="${RANDOM_STATE:-42}"
POINT_SIZE="${POINT_SIZE:-4.0}"
POINT_ALPHA="${POINT_ALPHA:-0.35}"
FIGURE_WIDTH="${FIGURE_WIDTH:-12}"
FIGURE_HEIGHT="${FIGURE_HEIGHT:-9}"
DPI="${DPI:-300}"
SAVE_EMBEDDINGS="${SAVE_EMBEDDINGS:-0}"

PLOT_TITLE="${PLOT_TITLE:-Robo2VLM Question Embeddings Projected with UMAP}"
MARKER_SIZE="${MARKER_SIZE:-10.0}"
MARKER_OPACITY="${MARKER_OPACITY:-0.5}"
INCLUDE_PLOTLYJS="${INCLUDE_PLOTLYJS:-cdn}"
PLOT_OUTPUT_HTML="${PLOT_OUTPUT_HTML:-}"

sanitize_name() {
  printf '%s' "$1" | tr -c '[:alnum:]_-' '_'
}

EMBEDDING_MODEL_TAG="$(sanitize_name "$EMBEDDING_MODEL")"
UMAP_CSV_PATH="${UMAP_CSV_PATH:-$UMAP_OUTPUT_DIR/${DATASET_SPLIT}_${EMBEDDING_MODEL_TAG}_umap.csv}"
if [ -z "$PLOT_OUTPUT_HTML" ]; then
  PLOT_OUTPUT_HTML="${UMAP_CSV_PATH%.csv}_interactive.html"
fi

echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Course tag: ${COURSE_TAG}"
echo "Account: ${ACCOUNT_TAG}"
echo "Team root: ${TEAM_ROOT}"
echo "Repo: ${REPO_DIR}"
echo "Venv: ${VENV_DIR}"
echo "Scratch: ${SCRATCH_BASE}"
echo "CUDA module: ${CUDA_MODULE}"
echo "Dataset: ${DATASET_NAME} (${DATASET_SPLIT})"
echo "Max samples: ${MAX_SAMPLES:-all}"
echo "Curator script: ${CURATION_SCRIPT_PATH}"
echo "Curator output dir: ${CURATION_OUTPUT_DIR}"
echo "Curator results: ${CURATION_RESULTS_PATH}"
echo "Curator model: ${CURATION_MODEL}"
echo "Prompt version: ${PROMPT_VERSION}"
echo "API key env: ${API_KEY_ENV}"
echo "Base URL: ${BASE_URL}"
echo "Curator batch size: ${CURATION_BATCH_SIZE}"
echo "Curator resume: ${CURATION_RESUME}"
echo "Curator overwrite: ${CURATION_OVERWRITE}"
echo "Curator streaming: ${CURATION_STREAMING}"
echo "UMAP script: ${UMAP_SCRIPT_PATH}"
echo "UMAP output dir: ${UMAP_OUTPUT_DIR}"
echo "Embedding model: ${EMBEDDING_MODEL}"
echo "Embedding batch size: ${UMAP_BATCH_SIZE}"
echo "Device: ${DEVICE}"
echo "UMAP streaming: ${UMAP_STREAMING}"
echo "PCA components: ${PCA_COMPONENTS}"
echo "UMAP neighbors: ${UMAP_NEIGHBORS}"
echo "UMAP min dist: ${UMAP_MIN_DIST}"
echo "UMAP metric: ${UMAP_METRIC}"
echo "Random state: ${RANDOM_STATE}"
echo "Save embeddings: ${SAVE_EMBEDDINGS}"
echo "Plotly script: ${PLOTLY_SCRIPT_PATH}"
echo "UMAP CSV: ${UMAP_CSV_PATH}"
echo "Plot HTML: ${PLOT_OUTPUT_HTML}"
echo "Plot title: ${PLOT_TITLE}"
echo "Marker size: ${MARKER_SIZE}"
echo "Marker opacity: ${MARKER_OPACITY}"
echo "Plotly JS mode: ${INCLUDE_PLOTLYJS}"

mkdir -p "$SCRATCH_BASE"/{hf,transformers,datasets,tmp,matplotlib}
mkdir -p "$CURATION_OUTPUT_DIR"
mkdir -p "$UMAP_OUTPUT_DIR"

export HF_HOME="$SCRATCH_BASE/hf"
export TRANSFORMERS_CACHE="$SCRATCH_BASE/transformers"
export HF_DATASETS_CACHE="$SCRATCH_BASE/datasets"
export TMPDIR="$SCRATCH_BASE/tmp"
export MPLCONFIGDIR="$SCRATCH_BASE/matplotlib"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

source "$VENV_DIR/bin/activate"

cd "$REPO_DIR"

UMAP_CMD=(
  python3 "$UMAP_SCRIPT_PATH"
  --dataset-name "$DATASET_NAME"
  --split "$DATASET_SPLIT"
  --curation-results "$CURATION_RESULTS_PATH"
  --output-dir "$UMAP_OUTPUT_DIR"
  --embedding-model "$EMBEDDING_MODEL"
  --batch-size "$UMAP_BATCH_SIZE"
  --device "$DEVICE"
  --pca-components "$PCA_COMPONENTS"
  --umap-neighbors "$UMAP_NEIGHBORS"
  --umap-min-dist "$UMAP_MIN_DIST"
  --umap-metric "$UMAP_METRIC"
  --random-state "$RANDOM_STATE"
  --point-size "$POINT_SIZE"
  --point-alpha "$POINT_ALPHA"
  --figure-width "$FIGURE_WIDTH"
  --figure-height "$FIGURE_HEIGHT"
  --dpi "$DPI"
  --group-by subcategory
)

if [ -n "$MAX_SAMPLES" ]; then
  UMAP_CMD+=(--max-samples "$MAX_SAMPLES")
fi

if [ "$UMAP_STREAMING" = "1" ]; then
  UMAP_CMD+=(--streaming)
else
  UMAP_CMD+=(--no-streaming)
fi

if [ "$SAVE_EMBEDDINGS" = "1" ]; then
  UMAP_CMD+=(--save-embeddings)
fi

printf 'Running UMAP command:\n  %q' "${UMAP_CMD[@]}"
printf '\n'

"${UMAP_CMD[@]}"

if [ ! -f "$UMAP_CSV_PATH" ]; then
  echo "Expected UMAP CSV was not created: $UMAP_CSV_PATH" >&2
  exit 1
fi

PLOTLY_CMD=(
  python3 "$PLOTLY_SCRIPT_PATH"
  --input-csv "$UMAP_CSV_PATH"
  --output-html "$PLOT_OUTPUT_HTML"
  --title "$PLOT_TITLE"
  --marker-size "$MARKER_SIZE"
  --marker-opacity "$MARKER_OPACITY"
  --include-plotlyjs "$INCLUDE_PLOTLYJS"
  --color-by subcategory
)

printf 'Running Plotly command:\n  %q' "${PLOTLY_CMD[@]}"
printf '\n'

"${PLOTLY_CMD[@]}"

echo "Finished."
echo "UMAP outputs: $UMAP_OUTPUT_DIR"
echo "Interactive plot: $PLOT_OUTPUT_HTML"
