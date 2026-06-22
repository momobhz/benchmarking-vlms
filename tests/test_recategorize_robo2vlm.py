import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.recategorize_robo2vlm import (
    CurationError,
    ALLOWED_LABELS,
    build_responses_payload,
    build_result_record,
    build_user_prompt,
    extract_original_tag,
    load_existing_results,
    make_cache_key,
    prepare_output_directory,
    summarize_results,
    validate_curator_payload,
)


class Args:
    dataset_name = "keplerccc/Robo2VLM-1"
    split = "test"
    model = "gpt-5-mini"
    prompt_version = "spatial_affordance_v1"
    streaming = True


def sample_record():
    return {
        "id": "example-1",
        "question": "Is there any obstacle blocking the robot from reaching the mug?",
        "choices": ["Yes", "No", "Cannot be determined"],
        "correct_answer": 0,
        "source_dataset": "keplerccc/Robo2VLM-1",
        "source_split": "test",
        "source_index": 12,
        "original_tag": "vqa_object_reachable",
    }


def test_extract_original_tag_from_json_metadata():
    row = {"metadata": json.dumps({"tag": "vqa_relative_direction"})}
    assert extract_original_tag(row) == "vqa_relative_direction"


def test_build_user_prompt_includes_choices_and_optional_tag():
    prompt = build_user_prompt(sample_record())
    assert "Question ID: example-1" in prompt
    assert "Original template tag: vqa_object_reachable" in prompt
    assert "1. Yes" in prompt
    assert "2. No" in prompt


def test_make_cache_key_changes_with_prompt_version():
    record = sample_record()
    key_a = make_cache_key(record, "gpt-5-mini", "v1")
    key_b = make_cache_key(record, "gpt-5-mini", "v2")
    assert key_a != key_b


def test_build_responses_payload_omits_temperature_for_gpt5_models():
    payload = build_responses_payload(
        model="gpt-5-mini",
        prompt_version="spatial_affordance_v1",
        record=sample_record(),
    )
    assert payload["model"] == "gpt-5-mini"
    assert "temperature" not in payload
    assert payload["metadata"]["question_id"] == "example-1"


def test_validate_curator_payload_accepts_expected_schema():
    payload = {
        "label": "affordance_understanding",
        "confidence": 0.83,
        "rationale": "The question is about whether the robot can physically reach the object.",
        "signals": ["reachability", "obstacle_blocking"],
    }
    validated = validate_curator_payload(payload)
    assert validated["label"] == "affordance_understanding"
    assert validated["confidence"] == 0.83


def test_validate_curator_payload_rejects_invalid_label():
    with pytest.raises(CurationError):
        validate_curator_payload(
            {
                "label": "both",
                "confidence": 0.5,
                "rationale": "Invalid label",
                "signals": ["mixed"],
            }
        )


def test_load_existing_results_uses_cache_key(tmp_path):
    payload = {"cache_key": "abc123", "curation_label": "neither"}
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    loaded = load_existing_results(path)
    assert loaded["abc123"]["curation_label"] == "neither"


def test_build_result_record_appends_curation_fields():
    record = sample_record()
    curated = {
        "label": "affordance_understanding",
        "confidence": 0.91,
        "rationale": "The benchmark target is whether the object can be reached.",
        "signals": ["reachability"],
    }
    result = build_result_record(
        record=record,
        curated=curated,
        cache_key="cache-key",
        model="gpt-5-mini",
        prompt_version="v1",
        attempts=2,
        latency_seconds=1.2345,
    )
    assert result["curation_label"] == "affordance_understanding"
    assert result["curation_subcategory"] is None
    assert result["curation_attempts"] == 2
    assert result["curation_latency_seconds"] == 1.2345


def test_summarize_results_aggregates_labels_and_tags():
    result_a = {
        "curation_label": "spatial_reasoning",
        "curation_confidence": 0.7,
        "original_tag": "vqa_relative_direction",
    }
    result_b = {
        "curation_label": "affordance_understanding",
        "curation_confidence": 0.9,
        "original_tag": "vqa_object_reachable",
    }
    summary = summarize_results([result_a, result_b], failed_count=1, args=Args())
    assert summary["label_counts"]["spatial_reasoning"] == 1
    assert summary["label_counts"]["affordance_understanding"] == 1
    assert summary["failed_count"] == 1
    assert summary["tag_breakdown"]["vqa_relative_direction"]["spatial_reasoning"] == 1
    assert summary["subcategory_counts"]["spatial_reasoning"] == {
        "distance": 0,
        "direction": 0,
        "none": 0,
    }
    assert summary["subcategory_rules"]["affordance_understanding"]["object_blockage"] == (
        "obstacle blocking"
    )


def test_prepare_output_directory_clears_stale_non_resume_outputs(tmp_path):
    output_dir = tmp_path / "curation"
    by_label_dir = output_dir / "by_label"
    by_label_dir.mkdir(parents=True)
    (output_dir / "failed_records.jsonl").write_text("old\n", encoding="utf-8")
    (output_dir / "summary.json").write_text("old\n", encoding="utf-8")
    (output_dir / "manifest.json").write_text("old\n", encoding="utf-8")
    for label in ALLOWED_LABELS:
        (by_label_dir / f"{label}.jsonl").write_text("old\n", encoding="utf-8")

    prepare_output_directory(output_dir, overwrite=False, resume=False)

    assert not (output_dir / "failed_records.jsonl").exists()
    assert not (output_dir / "summary.json").exists()
    assert not (output_dir / "manifest.json").exists()
    for label in ALLOWED_LABELS:
        assert not (by_label_dir / f"{label}.jsonl").exists()
