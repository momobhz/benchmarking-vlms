# Robo2VLM On ETH Student Cluster: Limitations And Required Patches

## Cluster limitations

- `1080ti` nodes are not viable for this benchmark path.
  - They are `sm_61`.
  - Current `vLLM` requires newer GPUs, and modern PyTorch wheels also dropped support for `sm_61`.
- Use a newer GPU node such as `5060ti`.
  - On the ETH student cluster, `5060ti` has 16 GB VRAM and is compatible with current `vLLM`.
- Batch jobs are non-interactive.
  - `hf auth login` must not run inside the Slurm script.
  - Authenticate on the login node first, using the same `HF_HOME` as the job.
- PyTorch should match the CUDA module loaded in the job.
  - For `cuda/13.0`, install a `cu130` PyTorch wheel as recommended by the cluster docs.

## Environment setup

- In the job script:
  - source modules with `. /etc/profile.d/modules.sh`
  - load CUDA via `module add cuda/13.0`
- Install PyTorch in a fresh env with:

```bash
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

- Keep HF auth outside the batch script.
  - set `HF_HOME=/work/courses/3dv/team43/robo2vlm_qwen_eval_cache/hf`
  - run `hf auth login` once interactively
  - remove `hf auth login` from the Slurm script

## Slurm script fixes

- `#SBATCH --gpus=1080ti:4` was wrong for this setup.
- Use a newer GPU request, for example:

```bash
#SBATCH --account=3dv
#SBATCH --gpus=5060ti:1
#SBATCH --output=/work/courses/3dv/team43/logs/%x-%j.out
```

- Keep paths explicit in the script:
  - repo under `/work/courses/3dv/team43/robo2VLM`
  - venv under `/work/courses/3dv/team43/3dv-env-cu130`
  - logs under `/work/courses/3dv/team43/logs`

## Required code patches

### 1. `FlexibleArgumentParser` compatibility

In [vision_language.py](/work/courses/3dv/team43/robo2VLM/benchmark/vision_language.py), replace:

```python
from vllm.utils import FlexibleArgumentParser
```

with:

```python
try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser
```

### 2. Load only the requested evaluation subset

In [evaluation.py](/work/courses/3dv/team43/robo2VLM/benchmark/evaluation.py), switch dataset loading to streaming and `take(max_samples)` when `max_samples` is set, so the benchmark does not materialize the full split before truncating (the full dataset is too large).

### 3. Fix Qwen loader signature mismatch

`_get_model_request_data()` assumed every model loader takes `(questions, modality, model_id)`, but Qwen's loader only takes `(questions, modality)`.

Use:

```python
try:
    return self.model_loader(instructed_questions, modality, self.model_id)
except TypeError:
    return self.model_loader(instructed_questions, modality)
```

### 4. Stop hardcoding tensor parallel size 4 for Qwen

Replace the Qwen-specific special case with:

```python
engine_args_dict["tensor_parallel_size"] = tensor_parallel_size
```

This allows `--tensor_parallel_size 1` on the cluster.

### 5. Make CLI batch size actually apply

In `main()`, replace the fixed batch size call with:

```python
model_results = evaluate_model(
    model_id,
    dataset,
    max_batch_size=args.batch_size,
    tensor_parallel_size=args.tensor_parallel_size
)
```

### 6. Replace the LLM answer extractor with direct parsing

The original benchmark loads an additional `meta-llama/Llama-3.2-3B-Instruct` vLLM engine just to extract option letters. On a single 16 GB GPU this caused memory pressure and prevented Qwen from loading reliably.

Replace `extract_letter_answer` with:

```python
def extract_letter_answer(queries, predicted_answers):
    """Extract letter answers (A, B, C, D, E) directly from model outputs."""
    if not isinstance(predicted_answers, list):
        return choice_answer_clean(predicted_answers)

    return [choice_answer_clean(answer) for answer in predicted_answers]
```

Also remove the extractor initialization call from `evaluate_model`:

```python
initialize_answer_extractor(...)
```

### 7. Fix `correct_answer` type mismatch

The dataset stores `correct_answer` as an integer index, but the benchmark compared it as if it were already a letter string and called `.upper()` on it.

The failing line was:

```python
"correct": letter_answers[i] == batch_data[i]["correct_answer"].upper() if letter_answers[i] else False,
```

Replace it with an index-to-letter conversion:

```python
expected_answer = batch_data[i]["correct_answer"]
if isinstance(expected_answer, int):
    expected_letter = chr(65 + expected_answer)
else:
    expected_letter = str(expected_answer).upper()
```

and then compare against `expected_letter`.

## Practical conclusion

The benchmark can run on the ETH student cluster only if:

- a modern GPU node such as `5060ti` is used
- CUDA-matched PyTorch is installed
- Hugging Face authentication is done outside the batch job
- the benchmark code is patched for:
  - dataset subsetting
  - Qwen loader compatibility
  - 1-GPU tensor parallelism
  - direct answer parsing instead of a second LLM extractor
  - `correct_answer` integer-to-letter conversion

Without those patches, the stock benchmark code was not robust enough for the cluster setup.
