from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from vlm_bench.config import append_flag, get_nested, write_config
from vlm_bench.paths import default_eval_script, repo_root


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def validate_eval_config(config: dict[str, Any]) -> None:
    models = _as_list(get_nested(config, "model.id")) or _as_list(config.get("models"))
    if not models:
        raise ValueError("Evaluation config requires model.id or models.")

    backend = get_nested(config, "model.backend", "vllm")
    if backend != "vllm":
        raise ValueError(
            f"The local evaluation runner only supports model.backend='vllm', got {backend!r}. "
            "Use this config as an API template until an API evaluator backend is added."
        )

    subset = get_nested(config, "data.subset", "full")
    curation_results = get_nested(config, "data.curation_results")
    if subset != "full" and not curation_results:
        raise ValueError("Evaluation config with data.subset != full requires data.curation_results.")

    prompt_mode = get_nested(config, "prompt.mode", "cot")
    if prompt_mode not in ("cot", "zero_shot"):
        raise ValueError("prompt.mode must be 'cot' or 'zero_shot'.")


def resolve_run_output_dir(config: dict[str, Any], root: Path | None = None) -> Path:
    root = root or repo_root()
    output_root = Path(get_nested(config, "run.output_dir", "runs/eval"))
    if not output_root.is_absolute():
        output_root = root / output_root

    run_name = get_nested(config, "run.name")
    if run_name:
        return output_root / str(run_name)
    return output_root


def resolve_input_path(value: Any, root: Path) -> str | None:
    """Resolve input paths from configs against repo and cluster roots.

    Evaluation configs are intentionally portable and use paths such as
    ``outputs/...``. On the ETH cluster, historical curation outputs often live
    one level above the repo under ``$TEAM_ROOT/outputs``. Prefer an existing
    path when possible, otherwise return the repo-relative path for a clear
    downstream error.
    """
    if value is None:
        return None

    path = Path(str(value))
    if path.is_absolute():
        return str(path)

    candidates = [root / path]
    team_root = os.environ.get("TEAM_ROOT")
    if team_root:
        candidates.append(Path(team_root) / path)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(candidates[-1] if team_root else candidates[0])


def build_evaluation_command(config: dict[str, Any], root: Path | None = None) -> list[str]:
    validate_eval_config(config)
    root = root or repo_root()

    eval_script = Path(get_nested(config, "paths.eval_script", default_eval_script()))
    if not eval_script.is_absolute():
        eval_script = root / eval_script

    models = _as_list(get_nested(config, "model.id")) or _as_list(config.get("models"))

    command = [sys.executable, str(eval_script), "--models", *models]
    append_flag(command, "--dataset", get_nested(config, "data.dataset"))
    append_flag(command, "--split", get_nested(config, "data.split"))
    append_flag(command, "--subset", get_nested(config, "data.subset"))
    append_flag(command, "--curation-results", resolve_input_path(get_nested(config, "data.curation_results"), root))
    append_flag(command, "--max_samples", get_nested(config, "data.max_samples"))
    append_flag(command, "--tensor_parallel_size", get_nested(config, "model.tensor_parallel_size"))
    append_flag(command, "--batch_size", get_nested(config, "inference.batch_size"))
    append_flag(command, "--temperature", get_nested(config, "inference.temperature"))
    append_flag(command, "--max_tokens", get_nested(config, "inference.max_tokens"))
    append_flag(command, "--prompt-mode", get_nested(config, "prompt.mode"))
    append_flag(command, "--image_mode", get_nested(config, "image.mode"))
    append_flag(command, "--depth_model", get_nested(config, "image.depth_model"))
    append_flag(command, "--depth_device", get_nested(config, "image.depth_device"))
    append_flag(command, "--sanity_check", get_nested(config, "sanity_check.mode"))
    append_flag(command, "--sanity_seed", get_nested(config, "sanity_check.seed"))
    append_flag(command, "--run-name", get_nested(config, "run.name"))
    append_flag(command, "--output-dir", resolve_run_output_dir(config, root) / "results")
    append_flag(command, "--config-path", config.get("_config_path"))

    if get_nested(config, "model.trust_remote_code", False):
        command.append("--trust_remote_code")

    return command


def run_eval_config(config: dict[str, Any], *, dry_run: bool = False, print_command: bool = False) -> int:
    root = repo_root()
    run_dir = resolve_run_output_dir(config, root)
    resolved_config_path = run_dir / "config.resolved.yaml"
    command = build_evaluation_command(config, root)

    if dry_run or print_command:
        print(shlex.join(command), flush=True)

    if dry_run:
        return 0

    write_config(resolved_config_path, config)
    env = os.environ.copy()
    src_path = str(root / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(command, cwd=root, env=env, check=False)
    return completed.returncode
