# VQA Dataset

This dataset contains Visual Question Answering (VQA) examples collected from multimodal trajectories. The dataset includes questions, multiple-choice answers, and associated images. All images are stored directly in the dataset.

## Dataset Structure

The dataset contains the following fields:

- `id`: Unique identifier for each VQA item
- `question`: The question text
- `choices`: List of possible answer choices
- `correct_answer_idx`: Index of the correct answer in the choices list
- `images`: List of question images as actual image data (not just paths)
- `choice_images`: List of images associated with each answer choice as actual image data
- `metadata`: Additional metadata about the VQA item (stored as JSON string)

## Usage

You can load the dataset using the HuggingFace datasets library:

```python
from datasets import load_dataset, load_from_disk
import matplotlib.pyplot as plt

# Load from HuggingFace Hub
dataset = load_dataset("your-username/dataset-name")
dataset = load_from_disk("hf_dataset")
# Example: Get a sample item
sample = dataset[0]
print(f"Question: {sample['question']}")
print(f"Choices: {sample['choices']}")
print(f"Correct answer: {sample['choices'][sample['correct_answer_idx']]}")

# Display an image associated with the question
if sample['images']:
    plt.figure(figsize=(10, 8))
    plt.imshow(sample['images'][0])
    plt.title(f"Question: {sample['question']}")
    plt.axis('off')
    plt.show()
```

## Dataset Creation

This dataset was created by:
1. Extracting VQA items from trajectory data
2. Randomly sampling a subset of the items
3. Loading and embedding actual image data directly in the dataset
4. Converting to the HuggingFace datasets format

## Question Recategorization

Use `scripts/recategorize_robo2vlm.py` to relabel Robo2VLM questions for the course project taxonomy:

- `spatial_reasoning`
- `affordance_understanding`
- `neither`

The script reads the published dataset from Hugging Face, sends each question to an OpenAI model using the Responses API with structured JSON output, and writes:

- `curation_results.jsonl`: canonical labeled records with rationale, confidence, and prompt version
- `by_label/*.jsonl`: filtered subsets for each new category
- `summary.json`: aggregate counts and original-tag breakdowns when tags are available

Example:

```bash
export OPENAI_API_KEY=...
python3 scripts/recategorize_robo2vlm.py \
  --dataset-name keplerccc/Robo2VLM-1 \
  --split test \
  --output-dir outputs/robo2vlm_spatial_affordance \
  --max-samples 200 \
  --resume
```

Notes:

- Install the Hugging Face `datasets` package before running the curator.
- `curation_results.jsonl` is the source of truth for downstream scripts.
- The output keeps source IDs instead of duplicating image payloads; join on `id` with the original dataset when you want to benchmark VLMs on the filtered subsets.

If you need a deterministic cleanup pass for known misclassifications, use
`scripts/relabel_robo2vlm_goal_state_questions.py`. It rewrites
`curation_results.jsonl`, regenerates `by_label/*.jsonl`, and updates
`summary.json` for any question whose text contains the goal-state phrase:

```bash
python3 scripts/relabel_robo2vlm_goal_state_questions.py \
  --output-dir outputs/robo2vlm_spatial_affordance
```

To assign deterministic subcategories on top of the existing top-level labels,
run:

```bash
python3 scripts/subcategorize_robo2vlm_questions.py \
  --output-dir outputs/robo2vlm_spatial_affordance
```

This adds `curation_subcategory` to each record and creates nested subsets:

- `by_label/spatial_reasoning/{distance,direction,none}.jsonl`
- `by_label/affordance_understanding/{grasp_stability,object_blockage,none}.jsonl`

For the ETH student cluster, run the deterministic post-processing pipeline
without re-calling the LLM:

```bash
sbatch run_robo2vlm_question_postprocess.sh
```

## Question Embedding Visualization

Use `scripts/visualize_robo2vlm_question_clusters.py` to verify the category split by
embedding every question and projecting the question embeddings into 2D with UMAP.

Example:

```bash
python3 scripts/visualize_robo2vlm_question_clusters.py \
  --curation-results outputs/robo2vlm_spatial_affordance/curation_results.jsonl \
  --output-dir outputs/robo2vlm_umap \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --group-by subcategory
```

The script reads labels from `curation_results.jsonl` and writes:

- `*_umap.csv`: one row per question with `source_index`, `id`, `label`, `subcategory`, and 2D coordinates
- `*_umap.png`: scatter plot colored by the requested grouping
- `*_umap_summary.json`: dataset, label counts, and projection settings

Notes:

- Streaming mode is enabled by default to avoid materializing the full dataset locally.
- Use `--no-streaming` only if you explicitly want the Hugging Face dataset cached on disk.

## Interactive Cluster Plot

Use `scripts/plot_robo2vlm_question_clusters.py` to turn the generated `*_umap.csv`
into an interactive Plotly HTML plot where hovering a point shows the original
question text.

Example:

```bash
python3 scripts/plot_robo2vlm_question_clusters.py \
  --input-csv outputs/robo2vlm_umap/test_sentence-transformers_all-MiniLM-L6-v2_umap.csv \
  --color-by subcategory
```

This writes `*_interactive.html` next to the input CSV by default.

Dependency:

- Install Plotly with `python3 -m pip install plotly`
