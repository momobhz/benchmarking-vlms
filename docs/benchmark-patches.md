# Benchmark Compatibility Notes

This page preserves the useful information from the old root-level
`code-report.md`. The current codebase already includes these fixes; this page
documents why they exist.

## ETH Student Cluster Constraints

- Prefer `5060ti` GPU jobs for this benchmark path. The current vLLM-based
  evaluator is not a good fit for `1080ti` nodes.
- Batch jobs are non-interactive. Run Hugging Face authentication on the login
  node before submitting jobs; do not run `hf auth login` inside Slurm scripts.
- PyTorch must match the CUDA module loaded in the job. For `cuda/13.0`, install
  PyTorch from the `cu130` wheel index.
- Slurm jobs must set the course/project account and an explicit runtime. The
  scripts use `#SBATCH --account=3dv`, `#SBATCH --time=...`, and
  `#SBATCH --gpus=5060ti:1`.
- The maintained Slurm entrypoints request `#SBATCH --ntasks=1`; this avoids the
  ETH Slurm rejection triggered by combining `--cpus-per-task` with GPU TRES.
- The Slurm entrypoints source modules with `. /etc/profile.d/modules.sh` and
  load `cuda/13.0` immediately after the `#SBATCH` block, matching the ETH
  student cluster documentation.

See [cluster.md](cluster.md) for current setup and submission commands.

## Evaluator Fixes Kept In The Codebase

The evaluator has been adapted from the upstream Robo2VLM code for smaller GPU
jobs and current dependency versions:

- `benchmark/vision_language.py` falls back to `argparse.ArgumentParser` when
  `vllm.utils.FlexibleArgumentParser` is unavailable.
- `benchmark/evaluation.py` streams and truncates datasets when `max_samples` is
  set, avoiding full split materialization for quick runs.
- Model loader calls handle both `(questions, modality, model_id)` and
  `(questions, modality)` signatures.
- Tensor parallelism is controlled by `--tensor_parallel_size`, so single-GPU
  student cluster jobs can use `1`.
- CLI `--batch_size` is passed through to the evaluator.
- Multiple-choice answer extraction uses direct option-letter parsing instead of
  loading a second LLM just to parse responses.
- Integer `correct_answer` values are converted to option letters before
  comparison.
- Prompt mode, temperature, max tokens, run name, and output directory are now
  explicit evaluation arguments and are normally supplied through
  `configs/eval/*.yaml`.
- DeepSeek-VL2 requires `timm`; it is part of the repository's `[eval]`
  dependency extra.

## Practical Baseline

For the 3DV student cluster, the expected baseline is:

```bash
. /etc/profile.d/modules.sh
module add cuda/13.0
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install --no-cache-dir -e "/work/courses/3dv/team43/benchmarking-vlms[eval]"
```

Submit config-driven evaluations with:

```bash
sbatch slurm/eval.sbatch configs/eval/qwen25_3b_spatial_cot_200.yaml
```
