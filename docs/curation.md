# Curation Workflow

The curation workflow labels Robo2VLM questions into:

- `spatial_reasoning`
- `affordance_understanding`
- `neither`

The canonical taxonomy lives in `src/vlm_bench/curation/taxonomy.py`.

Dry-run the curation command:

```bash
PYTHONPATH=src python3 -m vlm_bench curate \
  --config configs/curation/robo2vlm_spatial_affordance.yaml \
  --dry-run
```

Run on the cluster:

```bash
sbatch slurm/curate.sbatch configs/curation/robo2vlm_spatial_affordance.yaml
```

The config can also run deterministic post-processing after LLM curation:

- `postprocess.relabel_goal_state`: moves known goal-state matching questions to `neither`.
- `postprocess.subcategorize`: assigns deterministic subcategories such as `distance`,
  `direction`, `grasp_stability`, and `object_blockage`.

The output directory contains:

- `curation_results.jsonl`: source of truth for evaluation subsets.
- `by_label/*.jsonl`: materialized top-level subsets.
- nested subcategory JSONL files.
- `summary.json`: counts and metadata.
