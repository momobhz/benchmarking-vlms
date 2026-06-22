from pathlib import Path

import pytest

from vlm_bench.config import apply_overrides, load_config
from vlm_bench.eval.prompts import apply_prompt_mode
from vlm_bench.eval.runner import build_evaluation_command, resolve_input_path, resolve_run_output_dir


EVAL_MATRIX_200 = [
    (
        "configs/eval/qwen25_3b_spatial_cot_200.yaml",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "spatial",
        "cot",
    ),
    (
        "configs/eval/qwen25_3b_spatial_zeroshot_200.yaml",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "spatial",
        "zero_shot",
    ),
    (
        "configs/eval/qwen25_3b_affordance_cot_200.yaml",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "affordance",
        "cot",
    ),
    (
        "configs/eval/qwen25_3b_affordance_zeroshot_200.yaml",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "affordance",
        "zero_shot",
    ),
    (
        "configs/eval/deepseek_vl2_tiny_spatial_cot_200.yaml",
        "deepseek-ai/deepseek-vl2-tiny",
        "spatial",
        "cot",
    ),
    (
        "configs/eval/deepseek_vl2_tiny_spatial_zeroshot_200.yaml",
        "deepseek-ai/deepseek-vl2-tiny",
        "spatial",
        "zero_shot",
    ),
    (
        "configs/eval/deepseek_vl2_tiny_affordance_cot_200.yaml",
        "deepseek-ai/deepseek-vl2-tiny",
        "affordance",
        "cot",
    ),
    (
        "configs/eval/deepseek_vl2_tiny_affordance_zeroshot_200.yaml",
        "deepseek-ai/deepseek-vl2-tiny",
        "affordance",
        "zero_shot",
    ),
    (
        "configs/eval/gemma3_4b_spatial_cot_200.yaml",
        "google/gemma-3-4b-it",
        "spatial",
        "cot",
    ),
    (
        "configs/eval/gemma3_4b_spatial_zeroshot_200.yaml",
        "google/gemma-3-4b-it",
        "spatial",
        "zero_shot",
    ),
    (
        "configs/eval/gemma3_4b_affordance_cot_200.yaml",
        "google/gemma-3-4b-it",
        "affordance",
        "cot",
    ),
    (
        "configs/eval/gemma3_4b_affordance_zeroshot_200.yaml",
        "google/gemma-3-4b-it",
        "affordance",
        "zero_shot",
    ),
]


def test_build_evaluation_command_uses_core_experiment_knobs():
    config = load_config("configs/eval/qwen25_3b_affordance_zeroshot.yaml")
    command = build_evaluation_command(config, Path.cwd())

    assert "--models" in command
    assert "Qwen/Qwen2.5-VL-3B-Instruct" in command
    assert command[command.index("--subset") + 1] == "affordance"
    assert command[command.index("--prompt-mode") + 1] == "zero_shot"
    assert command[command.index("--temperature") + 1] == "0.0"
    assert command[command.index("--max_tokens") + 1] == "128"
    assert command[command.index("--batch_size") + 1] == "1"


def test_overrides_update_nested_values():
    config = load_config("configs/eval/qwen25_3b_spatial_cot.yaml")
    updated = apply_overrides(
        config,
        [
            "data.max_samples=7",
            "prompt.mode=zero_shot",
            "inference.temperature=0.2",
        ],
    )
    command = build_evaluation_command(updated, Path.cwd())

    assert command[command.index("--max_samples") + 1] == "7"
    assert command[command.index("--prompt-mode") + 1] == "zero_shot"
    assert command[command.index("--temperature") + 1] == "0.2"


def test_subset_config_requires_curation_results():
    config = load_config("configs/eval/qwen25_3b_spatial_cot.yaml")
    config["data"].pop("curation_results")

    with pytest.raises(ValueError, match="curation_results"):
        build_evaluation_command(config, Path.cwd())


def test_non_vllm_backend_is_rejected_by_local_runner():
    config = load_config("configs/eval/muse_spark_spatial_cot_200.yaml")

    with pytest.raises(ValueError, match="only supports model.backend='vllm'"):
        build_evaluation_command(config, Path.cwd())


def test_resolve_run_output_dir_uses_run_name():
    config = load_config("configs/eval/qwen25_3b_spatial_cot.yaml")
    assert resolve_run_output_dir(config, Path("/repo")) == (
        Path("/repo") / "runs" / "eval" / "qwen25_3b_spatial_cot_t0"
    )


def test_resolve_input_path_prefers_existing_team_root_file(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    team_root = tmp_path / "team"
    curation_path = team_root / "outputs" / "robo2vlm_spatial_affordance" / "curation_results.jsonl"
    curation_path.parent.mkdir(parents=True)
    curation_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("TEAM_ROOT", str(team_root))

    assert resolve_input_path(
        "outputs/robo2vlm_spatial_affordance/curation_results.jsonl",
        repo_root,
    ) == str(curation_path)


def test_prompt_modes_are_distinct():
    cot_prompt = apply_prompt_mode(["Question?"], "cot")[0]
    zero_shot_prompt = apply_prompt_mode(["Question?"], "zero_shot")[0]

    assert "Reason step by step" in cot_prompt
    assert "Only output the correct option letter" in zero_shot_prompt


@pytest.mark.parametrize("config_path,model_id,subset,prompt_mode", EVAL_MATRIX_200)
def test_requested_200_sample_eval_matrix(config_path, model_id, subset, prompt_mode):
    config = load_config(config_path)
    command = build_evaluation_command(config, Path.cwd())

    assert model_id in command
    assert command[command.index("--subset") + 1] == subset
    assert command[command.index("--prompt-mode") + 1] == prompt_mode
    assert command[command.index("--max_samples") + 1] == "200"
    assert command[command.index("--temperature") + 1] == "0.0"
    assert command[command.index("--batch_size") + 1] == "1"
