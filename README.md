# Benchmarking VLMs for Robotic Spatial and Affordance Understanding

This repository contains a university research project evaluating
vision-language models on Robo2VLM-1 robotics VQA questions. The project focuses
on two capabilities that matter for manipulation tasks:

- Spatial reasoning over camera views, depth, and directional overlays.
- Affordance understanding, especially object blockage and grasp stability.

The maintained workflow is config-driven:

1. Curate Robo2VLM questions into spatial, affordance, and excluded subsets.
2. Evaluate VLM checkpoints on fixed subsets with zero-shot and CoT prompts.
3. Analyze accuracy, subcategory behavior, and visual-grounding sanity checks.

Legacy Robo2VLM generation and fine-tuning code is retained for provenance, but
the benchmark entry points live under `src/vlm_bench`, `configs`, `scripts`,
and `slurm`.

## Quick Start

Install the local package:

```bash
pip install -e ".[eval,analysis,test]"
```

Dry-run an evaluation locally. This validates the config and command plumbing
without loading vLLM or running inference.

```bash
vlm-bench eval --config configs/eval/qwen25_3b_spatial_cot_200.yaml --dry-run
```

Submit the same evaluation to the cluster:

```bash
sbatch slurm/eval.sbatch configs/eval/qwen25_3b_spatial_cot_200.yaml
```

Override experiment parameters without editing the config:

```bash
vlm-bench eval \
  --config configs/eval/qwen25_3b_spatial_cot_200.yaml \
  --set data.max_samples=20 \
  --set prompt.mode=zero_shot \
  --set inference.temperature=0.2 \
  --dry-run
```

Run the test suite:

```bash
pytest
```

## Repository Structure

```text
configs/          Experiment, curation, analysis, and cluster defaults
src/vlm_bench/    Config loader, CLI, prompt templates, taxonomy, workflow runners
benchmark/        Adapted legacy Robo2VLM/vLLM evaluator
scripts/          Dataset curation, post-processing, and visualization utilities
slurm/            ETH cluster entry points
cluster_scripts/  Compatibility wrappers around the maintained Slurm workflow
generation/       Original Robo2VLM data-generation code
finetune/         Original fine-tuning code and notes
docs/             Focused workflow and result documentation
runs/             Tracked evaluation and diagnostic result artifacts
output/pdf/       Tracked figures and inserts used in the report/poster
poster/           Poster source and final PDF
tests/            Smoke and unit tests
```

## Evaluation Configs

Use `configs/eval/*.yaml` to control:

- `model.id`: Hugging Face checkpoint.
- `data.subset`: `full`, `spatial`, `affordance`, `neither`, or a supported subcategory.
- `data.max_samples`: sample count for capped runs.
- `prompt.mode`: `cot` or `zero_shot`.
- `inference.temperature`: sampling temperature.
- `image.mode`: `rgb` or `rgb_depth`.

Results are written to:

```text
runs/eval/<run.name>/
  config.resolved.yaml
  results/*.json
```

## Workflows

Evaluation:

```bash
vlm-bench eval --config configs/eval/qwen25_3b_spatial_cot_200.yaml --dry-run
sbatch slurm/eval.sbatch configs/eval/qwen25_3b_spatial_cot_200.yaml
```

The ready-to-run 200-sample Qwen2.5-VL and DeepSeek-VL2 matrix lives in
[configs/eval/README.md](configs/eval/README.md).

Curation:

```bash
vlm-bench curate --config configs/curation/robo2vlm_spatial_affordance.yaml --dry-run
sbatch slurm/curate.sbatch configs/curation/robo2vlm_spatial_affordance.yaml
```

Question embedding analysis:

```bash
vlm-bench analyze --config configs/analysis/robo2vlm_question_umap.yaml --dry-run
sbatch slurm/analyze.sbatch configs/analysis/robo2vlm_question_umap.yaml
```

Fine-tuning:

```bash
sbatch slurm/finetune.sbatch
```

Primary documentation:

- [docs/evaluation.md](docs/evaluation.md)
- [docs/evaluation-results-and-sanity.md](docs/evaluation-results-and-sanity.md)
- [docs/curation.md](docs/curation.md)
- [docs/cluster.md](docs/cluster.md)
- [docs/benchmark-patches.md](docs/benchmark-patches.md)

## Results and Artifacts

The tracked result artifacts are intentionally small enough for repository
submission:

- `runs/eval/*/results/*.json`: 200-example evaluation outputs.
- `runs/diagnostics/image_understanding/*/image_understanding_sanity.json`:
  image-understanding sanity probes.
- `output/pdf/`: generated figures used by the report and poster.
- `poster/final.pdf`: final A1 project poster.

Large reproducible caches, model weights, virtual environments, and local build
outputs are ignored.

## Cluster Notes

The evaluator in `benchmark/` uses vLLM and should be run on a compatible GPU
node. The Slurm scripts default to ETH 3DV student cluster paths and cache
locations, but all important paths can be overridden with environment variables:

```bash
TEAM_ROOT=/work/courses/3dv/team43 \
VENV_DIR=/work/courses/3dv/team43/3dv-env-cu130 \
SCRATCH_BASE=/work/courses/3dv/team43/vlm_bench_cache \
sbatch slurm/eval.sbatch configs/eval/qwen25_3b_spatial_cot_200.yaml
```

Authenticate with Hugging Face on the login node using the same `HF_HOME` that
the job will use. Batch jobs should not call `hf auth login`.

See [docs/cluster.md](docs/cluster.md) for setup commands, CUDA/PyTorch matching
notes, storage guidance, and ETH cluster documentation links.

## Citation

This project builds on Robo2VLM:

```bibtex
@misc{chen2025robo2vlmvisualquestionanswering,
    title={Robo2VLM: Visual Question Answering from Large-Scale In-the-Wild Robot Manipulation Datasets},
    author={Kaiyuan Chen and Shuangyu Xie and Zehan Ma and Pannag Sanketi and Ken Goldberg},
    year={2025},
    eprint={2505.15517},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```
