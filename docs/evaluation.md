# Evaluation Workflow

Evaluations are config-driven. A config captures the model, curated subset,
sample count, prompt mode, temperature, image mode, and Slurm-independent
runtime settings.

Dry-run locally without loading any model:

```bash
PYTHONPATH=src python3 -m vlm_bench eval \
  --config configs/eval/qwen25_3b_spatial_cot.yaml \
  --dry-run
```

Run on the cluster:

```bash
sbatch slurm/eval.sbatch configs/eval/qwen25_3b_spatial_cot.yaml
```

Override a value without creating a new file:

```bash
PYTHONPATH=src python3 -m vlm_bench eval \
  --config configs/eval/qwen25_3b_spatial_cot.yaml \
  --set data.max_samples=20 \
  --set prompt.mode=zero_shot \
  --dry-run
```

Important config fields:

- `model.id`: Hugging Face model ID.
- `data.subset`: `full`, `spatial`, `affordance`, `neither`, or supported subcategories.
- `data.curation_results`: required for any curated subset.
- `data.max_samples`: optional cap for quick runs.
- `prompt.mode`: `cot` or `zero_shot`.
- `inference.temperature`: vLLM sampling temperature.
- `inference.batch_size`: evaluator batch size.
- `image.mode`: `rgb` or `rgb_depth`.

Results are written under `run.output_dir/run.name/results/`, with a resolved
copy of the config at `run.output_dir/run.name/config.resolved.yaml`.

## 200-Sample Evaluation Matrix

The repository includes ready-to-submit configs for the requested Qwen2.5-VL and
DeepSeek-VL2 runs:

```bash
sbatch slurm/eval.sbatch configs/eval/qwen25_3b_spatial_cot_200.yaml
sbatch slurm/eval.sbatch configs/eval/qwen25_3b_spatial_zeroshot_200.yaml
sbatch slurm/eval.sbatch configs/eval/qwen25_3b_affordance_cot_200.yaml
sbatch slurm/eval.sbatch configs/eval/qwen25_3b_affordance_zeroshot_200.yaml

sbatch slurm/eval.sbatch configs/eval/deepseek_vl2_tiny_spatial_cot_200.yaml
sbatch slurm/eval.sbatch configs/eval/deepseek_vl2_tiny_spatial_zeroshot_200.yaml
sbatch slurm/eval.sbatch configs/eval/deepseek_vl2_tiny_affordance_cot_200.yaml
sbatch slurm/eval.sbatch configs/eval/deepseek_vl2_tiny_affordance_zeroshot_200.yaml

sbatch slurm/eval.sbatch configs/eval/gemma3_4b_spatial_cot_200.yaml
sbatch slurm/eval.sbatch configs/eval/gemma3_4b_spatial_zeroshot_200.yaml
sbatch slurm/eval.sbatch configs/eval/gemma3_4b_affordance_cot_200.yaml
sbatch slurm/eval.sbatch configs/eval/gemma3_4b_affordance_zeroshot_200.yaml
```

See [configs/eval/README.md](../configs/eval/README.md) for the same matrix in
table form.

Muse Spark API-template configs are also present:

```text
configs/eval/muse_spark_spatial_cot_200.yaml
configs/eval/muse_spark_spatial_zeroshot_200.yaml
configs/eval/muse_spark_affordance_cot_200.yaml
configs/eval/muse_spark_affordance_zeroshot_200.yaml
```

They use `model.backend: meta_api_private_preview`. The local evaluator currently
supports vLLM checkpoints only, so these files document the intended benchmark
runs until an API backend is added.
