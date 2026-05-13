# Benchmarking VLMs on Spatial and Affordance Understanding

This repository evaluates vision-language models on Robo2VLM-style robotics VQA
questions, with emphasis on spatial reasoning and affordance understanding.

The original Robo2VLM generation and fine-tuning code is still present, but the
main benchmark workflow is now config-driven:

1. Curate or post-process question labels.
2. Evaluate a VLM on a curated subset.
3. Analyze question clusters and result outputs.

## Quick Start

Install the local package in your environment:

```bash
pip install -e .
```

Dry-run an evaluation command locally. This only validates config plumbing and
prints the command; it does not load vLLM or run inference.

```bash
vlm-bench eval --config configs/eval/qwen25_3b_spatial_cot.yaml --dry-run
```

Submit the same evaluation to the cluster:

```bash
sbatch slurm/eval.sbatch configs/eval/qwen25_3b_spatial_cot.yaml
```

Override experiment parameters without editing the config:

```bash
vlm-bench eval \
  --config configs/eval/qwen25_3b_spatial_cot.yaml \
  --set data.max_samples=20 \
  --set prompt.mode=zero_shot \
  --set inference.temperature=0.2 \
  --dry-run
```

## Main Evaluation Knobs

Use `configs/eval/*.yaml` to control:

- `model.id`: VLM checkpoint to evaluate.
- `data.subset`: `full`, `spatial`, `affordance`, `neither`, or a supported subcategory.
- `data.max_samples`: sample count for the run.
- `prompt.mode`: `cot` or `zero_shot`.
- `inference.temperature`: model sampling temperature.
- `image.mode`: `rgb` or `rgb_depth`.

Results are written to:

```text
runs/eval/<run.name>/
  config.resolved.yaml
  results/*.json
```

## Repository Structure

```text
configs/          Experiment, curation, analysis, and cluster defaults
src/vlm_bench/    Config loader, CLI, prompt templates, taxonomy, workflow runners
benchmark/        Legacy vLLM evaluator backend
scripts/          Data curation, post-processing, visualization scripts
slurm/            Generic cluster entrypoints
cluster_scripts/  Compatibility wrappers for older submission commands
generation/       Original Robo2VLM data-generation code
finetune/         Original fine-tuning code
docs/             Workflow documentation
tests/            Smoke and unit tests
```

## Workflows

Evaluation:

```bash
vlm-bench eval --config configs/eval/qwen25_3b_spatial_cot.yaml --dry-run
sbatch slurm/eval.sbatch configs/eval/qwen25_3b_spatial_cot.yaml
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

See:

- [docs/evaluation.md](docs/evaluation.md)
- [docs/curation.md](docs/curation.md)
- [docs/cluster.md](docs/cluster.md)
- [docs/benchmark-patches.md](docs/benchmark-patches.md)

## ETH Cluster Notes

The evaluator in `benchmark/` uses vLLM and should be run on a compatible GPU
node. The Slurm scripts default to the ETH 3DV student cluster paths and cache
locations, but all important paths can be overridden with environment variables.
The scripts follow the student cluster guidance: they set `--account`, `--time`,
request `--gpus=5060ti:1` and `--ntasks=1`, load
`/etc/profile.d/modules.sh` immediately after the `#SBATCH` block, activate
`cuda/13.0`, and keep the cluster-provided `$TMPDIR` for local temporary files.

```bash
TEAM_ROOT=/work/courses/3dv/team43 \
VENV_DIR=/work/courses/3dv/team43/3dv-env-cu130 \
SCRATCH_BASE=/work/courses/3dv/team43/vlm_bench_cache \
sbatch slurm/eval.sbatch configs/eval/qwen25_3b_spatial_cot.yaml
```

Authenticate with Hugging Face on the login node using the same `HF_HOME` that
the job will use. Batch jobs should not call `hf auth login`.

See [docs/cluster.md](docs/cluster.md) for the setup command, CUDA/PyTorch
matching notes, storage guidance, and links to the ETH cluster documentation.

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
