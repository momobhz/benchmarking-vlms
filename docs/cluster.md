# Cluster Workflow

These scripts are written for the D-INFK ETH student cluster documented at:

- Student cluster overview: https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentCluster
- Running jobs: https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterRunningJobs
- CUDA and PyTorch: https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterCuda
- Copying data: https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterCopyingData

Cluster jobs use thin Slurm entrypoints and a shared body:

- `slurm/eval.sbatch`
- `slurm/curate.sbatch`
- `slurm/analyze.sbatch`
- `slurm/workflow_job.sh`

The shared body sets Hugging Face, dataset, vLLM, and Matplotlib cache paths
under `SCRATCH_BASE`, preserves `$TMPDIR` for local temporary files, activates
the configured virtual environment, and runs the selected `vlm_bench` workflow.

The Slurm entrypoints follow the cluster requirements:

- `#SBATCH --account=3dv` selects the course/project tag.
- `#SBATCH --time=...` sets the maximum runtime instead of relying on the 60 minute default.
- `#SBATCH --gpus=5060ti:1` requests a specific 16 GB RTX 5060 Ti GPU.
- `#SBATCH --ntasks=1` keeps each submission to one Slurm task. The ETH student
  cluster rejects `--cpus-per-task` together with the GPU request.
- `. /etc/profile.d/modules.sh` is the first command after the `#SBATCH` block.
- `module add cuda/13.0` activates the CUDA module used by the environment.
- The cluster-provided `$TMPDIR` is preserved for local temporary files.

Typical evaluation:

```bash
sbatch slurm/eval.sbatch configs/eval/qwen25_3b_spatial_cot.yaml
```

Fine-tuning jobs use the same cluster conventions:

```bash
sbatch slurm/finetune.sbatch
FINETUNE_CONFIG=llama_vision.yaml sbatch slurm/finetune.sbatch
```

Common environment overrides:

```bash
TEAM_ROOT=/work/courses/3dv/team43 \
VENV_DIR=/work/courses/3dv/team43/3dv-env-cu130 \
SCRATCH_BASE=/work/courses/3dv/team43/vlm_bench_cache \
sbatch slurm/eval.sbatch configs/eval/qwen25_3b_spatial_cot.yaml
```

## Environment Setup

Create the virtual environment on the login node. The login nodes are for setup,
building, and compiling, not for benchmark computation.

```bash
ssh student-cluster.inf.ethz.ch
. /etc/profile.d/modules.sh
module avail
module add cuda/13.0

python3 -m venv /work/courses/3dv/team43/3dv-env-cu130
source /work/courses/3dv/team43/3dv-env-cu130/bin/activate
pip install --upgrade pip
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install --no-cache-dir -e "/work/courses/3dv/team43/benchmarking-vlms[eval,analysis]"
```

The PyTorch wheel suffix must match the loaded CUDA module. For `cuda/13.0`,
use the `cu130` PyTorch index URL.

DeepSeek-VL2 needs `timm` for its vision tower. It is included in the
`[eval]` extra. If your environment predates this change, update it with:

```bash
source /work/courses/3dv/team43/3dv-env-cu130/bin/activate
pip install --no-cache-dir timm
```

## Storage

Use `space` on the cluster to inspect writable network filesystems. The default
repo paths assume the course work area exists under `/work/courses/3dv/team43`.

The job uses:

- `SCRATCH_BASE` for Hugging Face, dataset, vLLM, and Matplotlib caches.
- `$TMPDIR` for local temporary files on the compute node. The cluster deletes
  this directory when the job ends.

If the team work area becomes tight, point `SCRATCH_BASE` at your personal
network scratch path, for example:

```bash
SCRATCH_BASE=/work/scratch/$USER/vlm_bench_cache \
sbatch slurm/eval.sbatch configs/eval/qwen25_3b_spatial_cot.yaml
```

Network scratch has retention and file-count limits, so do not use it as the
only copy of important results.

## Useful Cluster Commands

```bash
courses       # show available course/project tags and time limits
space         # show writable storage locations
squeue        # inspect queued/running jobs
scancel <job> # cancel a bad queued or running job
```

Legacy scripts in `cluster_scripts/` delegate to the same shared workflow body
and select default configs. Root-level job scripts have been removed; use
`slurm/*.sbatch` instead.
