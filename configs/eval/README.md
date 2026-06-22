# Evaluation Configs

The 200-sample model/category/prompt matrix requested for cluster evaluation is:

| Model | Category | Prompt | Config |
| --- | --- | --- | --- |
| Qwen2.5-VL 3B | spatial | CoT | `qwen25_3b_spatial_cot_200.yaml` |
| Qwen2.5-VL 3B | spatial | zero-shot | `qwen25_3b_spatial_zeroshot_200.yaml` |
| Qwen2.5-VL 3B | affordance | CoT | `qwen25_3b_affordance_cot_200.yaml` |
| Qwen2.5-VL 3B | affordance | zero-shot | `qwen25_3b_affordance_zeroshot_200.yaml` |
| DeepSeek-VL2 tiny | spatial | CoT | `deepseek_vl2_tiny_spatial_cot_200.yaml` |
| DeepSeek-VL2 tiny | spatial | zero-shot | `deepseek_vl2_tiny_spatial_zeroshot_200.yaml` |
| DeepSeek-VL2 tiny | affordance | CoT | `deepseek_vl2_tiny_affordance_cot_200.yaml` |
| DeepSeek-VL2 tiny | affordance | zero-shot | `deepseek_vl2_tiny_affordance_zeroshot_200.yaml` |
| Gemma 3 4B IT | spatial | CoT | `gemma3_4b_spatial_cot_200.yaml` |
| Gemma 3 4B IT | spatial | zero-shot | `gemma3_4b_spatial_zeroshot_200.yaml` |
| Gemma 3 4B IT | affordance | CoT | `gemma3_4b_affordance_cot_200.yaml` |
| Gemma 3 4B IT | affordance | zero-shot | `gemma3_4b_affordance_zeroshot_200.yaml` |

Muse Spark configs are included as API templates, because Muse Spark is not a
local vLLM checkpoint:

| Model | Category | Prompt | Config |
| --- | --- | --- | --- |
| Muse Spark | spatial | CoT | `muse_spark_spatial_cot_200.yaml` |
| Muse Spark | spatial | zero-shot | `muse_spark_spatial_zeroshot_200.yaml` |
| Muse Spark | affordance | CoT | `muse_spark_affordance_cot_200.yaml` |
| Muse Spark | affordance | zero-shot | `muse_spark_affordance_zeroshot_200.yaml` |

These use `model.backend: meta_api_private_preview`. The current `vlm-bench eval`
runner intentionally rejects non-`vllm` backends so these templates are not
accidentally submitted as local GPU jobs.

Submit one config with:

```bash
sbatch slurm/eval.sbatch configs/eval/qwen25_3b_spatial_cot_200.yaml
```

Dry-run locally without loading a model:

```bash
PYTHONPATH=src python3 -m vlm_bench eval \
  --config configs/eval/qwen25_3b_spatial_cot_200.yaml \
  --dry-run
```
