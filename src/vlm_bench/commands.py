from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from vlm_bench.config import append_flag, get_nested
from vlm_bench.paths import default_script, repo_root


def _run_commands(commands: list[list[str]], *, dry_run: bool) -> int:
    for command in commands:
        print(shlex.join(command), flush=True)
        if dry_run:
            continue
        env = os.environ.copy()
        root = repo_root()
        env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        completed = subprocess.run(command, cwd=root, env=env, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


def build_curate_commands(config: dict[str, Any]) -> list[list[str]]:
    curation_script = Path(get_nested(config, "paths.recategorize_script", default_script("recategorize_robo2vlm.py")))
    if not curation_script.is_absolute():
        curation_script = repo_root() / curation_script

    command = [sys.executable, str(curation_script)]
    append_flag(command, "--dataset-name", get_nested(config, "dataset.name"))
    append_flag(command, "--split", get_nested(config, "dataset.split"))
    append_flag(command, "--output-dir", get_nested(config, "output.dir"))
    append_flag(command, "--model", get_nested(config, "curator.model"))
    append_flag(command, "--prompt-version", get_nested(config, "curator.prompt_version"))
    append_flag(command, "--api-key-env", get_nested(config, "curator.api_key_env"))
    append_flag(command, "--base-url", get_nested(config, "curator.base_url"))
    append_flag(command, "--timeout-seconds", get_nested(config, "curator.timeout_seconds"))
    append_flag(command, "--max-retries", get_nested(config, "curator.max_retries"))
    append_flag(command, "--retry-backoff-seconds", get_nested(config, "curator.retry_backoff_seconds"))
    append_flag(command, "--batch-size", get_nested(config, "runtime.batch_size"))
    append_flag(command, "--max-samples", get_nested(config, "runtime.max_samples"))
    append_flag(command, "--start-index", get_nested(config, "runtime.start_index"))
    append_flag(command, "--end-index", get_nested(config, "runtime.end_index"))

    if get_nested(config, "runtime.resume", False):
        command.append("--resume")
    if get_nested(config, "runtime.overwrite", False):
        command.append("--overwrite")
    if get_nested(config, "runtime.streaming", True) is False:
        command.append("--no-streaming")

    commands = [command]
    output_dir = get_nested(config, "output.dir")
    if get_nested(config, "postprocess.relabel_goal_state", False):
        commands.append(
            [
                sys.executable,
                str(default_script("relabel_robo2vlm_goal_state_questions.py")),
                "--output-dir",
                str(output_dir),
            ]
        )
    if get_nested(config, "postprocess.subcategorize", False):
        commands.append(
            [
                sys.executable,
                str(default_script("subcategorize_robo2vlm_questions.py")),
                "--output-dir",
                str(output_dir),
            ]
        )
    return commands


def run_curate_config(config: dict[str, Any], *, dry_run: bool = False) -> int:
    return _run_commands(build_curate_commands(config), dry_run=dry_run)


def build_analyze_commands(config: dict[str, Any]) -> list[list[str]]:
    visualize_script = Path(get_nested(config, "paths.visualize_script", default_script("visualize_robo2vlm_question_clusters.py")))
    plot_script = Path(get_nested(config, "paths.plot_script", default_script("plot_robo2vlm_question_clusters.py")))
    if not visualize_script.is_absolute():
        visualize_script = repo_root() / visualize_script
    if not plot_script.is_absolute():
        plot_script = repo_root() / plot_script

    command = [sys.executable, str(visualize_script)]
    append_flag(command, "--dataset-name", get_nested(config, "dataset.name"))
    append_flag(command, "--split", get_nested(config, "dataset.split"))
    append_flag(command, "--curation-results", get_nested(config, "curation.results"))
    append_flag(command, "--output-dir", get_nested(config, "output.dir"))
    append_flag(command, "--embedding-model", get_nested(config, "embedding.model"))
    append_flag(command, "--batch-size", get_nested(config, "embedding.batch_size"))
    append_flag(command, "--device", get_nested(config, "embedding.device"))
    append_flag(command, "--max-samples", get_nested(config, "dataset.max_samples"))
    append_flag(command, "--pca-components", get_nested(config, "umap.pca_components"))
    append_flag(command, "--umap-neighbors", get_nested(config, "umap.neighbors"))
    append_flag(command, "--umap-min-dist", get_nested(config, "umap.min_dist"))
    append_flag(command, "--umap-metric", get_nested(config, "umap.metric"))
    append_flag(command, "--random-state", get_nested(config, "umap.random_state"))
    append_flag(command, "--point-size", get_nested(config, "plot.point_size"))
    append_flag(command, "--point-alpha", get_nested(config, "plot.point_alpha"))
    append_flag(command, "--figure-width", get_nested(config, "plot.figure_width"))
    append_flag(command, "--figure-height", get_nested(config, "plot.figure_height"))
    append_flag(command, "--dpi", get_nested(config, "plot.dpi"))

    if get_nested(config, "dataset.streaming", True):
        command.append("--streaming")
    else:
        command.append("--no-streaming")
    if get_nested(config, "embedding.save_embeddings", False):
        command.append("--save-embeddings")

    commands = [command]
    if get_nested(config, "plot.interactive", True):
        plot_command = [sys.executable, str(plot_script)]
        append_flag(plot_command, "--input-csv", get_nested(config, "plot.input_csv"))
        append_flag(plot_command, "--output-html", get_nested(config, "plot.output_html"))
        append_flag(plot_command, "--title", get_nested(config, "plot.title"))
        append_flag(plot_command, "--marker-size", get_nested(config, "plot.marker_size"))
        append_flag(plot_command, "--marker-opacity", get_nested(config, "plot.marker_opacity"))
        append_flag(plot_command, "--include-plotlyjs", get_nested(config, "plot.include_plotlyjs"))
        commands.append(plot_command)
    return commands


def run_analyze_config(config: dict[str, Any], *, dry_run: bool = False) -> int:
    return _run_commands(build_analyze_commands(config), dry_run=dry_run)
