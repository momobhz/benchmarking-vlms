import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.relabel_robo2vlm_goal_state_questions import (
    DEFAULT_MATCH_PHRASE,
    relabel_output_dir,
)
from scripts.subcategorize_robo2vlm_questions import subcategorize_output_dir


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_subcategorize_output_dir_rewrites_results_subsets_and_summary(tmp_path):
    output_dir = tmp_path / "curation"
    by_label_dir = output_dir / "by_label"
    by_label_dir.mkdir(parents=True)

    results = [
        {
            "id": "spatial-distance",
            "question": "In the image, which colored point is CLOSEST to the camera?",
            "curation_label": "spatial_reasoning",
            "curation_confidence": 0.9,
            "source_dataset": "keplerccc/Robo2VLM-1",
            "source_split": "test",
            "source_index": 0,
        },
        {
            "id": "spatial-none",
            "question": "Where is the mug relative to the plate?",
            "curation_label": "spatial_reasoning",
            "curation_confidence": 0.8,
            "source_dataset": "keplerccc/Robo2VLM-1",
            "source_split": "test",
            "source_index": 1,
        },
        {
            "id": "affordance-stability",
            "question": "Is the robot's grasp of the peg stable?",
            "curation_label": "affordance_understanding",
            "curation_confidence": 0.95,
            "source_dataset": "keplerccc/Robo2VLM-1",
            "source_split": "test",
            "source_index": 2,
        },
        {
            "id": "affordance-blockage",
            "question": "Is there any obstacle blocking the robot from reaching peg?",
            "curation_label": "affordance_understanding",
            "curation_confidence": 0.93,
            "source_dataset": "keplerccc/Robo2VLM-1",
            "source_split": "test",
            "source_index": 3,
        },
        {
            "id": "neither",
            "question": "Is the robot's gripper open?",
            "curation_label": "neither",
            "curation_confidence": 0.99,
            "source_dataset": "keplerccc/Robo2VLM-1",
            "source_split": "test",
            "source_index": 4,
        },
    ]
    write_jsonl(output_dir / "curation_results.jsonl", results)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "created_at": "2026-04-12T10:00:00+00:00",
                "dataset_name": "keplerccc/Robo2VLM-1",
                "split": "test",
                "model": "gpt-5-mini",
                "prompt_version": "spatial_affordance_v1",
                "streaming": True,
                "processed_count": 5,
                "failed_count": 0,
                "new_count": 5,
                "resume_skipped_count": 0,
                "runtime_seconds": 1.23,
            }
        ),
        encoding="utf-8",
    )

    updated_counts = subcategorize_output_dir(output_dir)

    assert updated_counts == {
        "spatial_reasoning": 2,
        "affordance_understanding": 2,
        "neither": 0,
    }

    updated_results = read_jsonl(output_dir / "curation_results.jsonl")
    by_id = {record["id"]: record for record in updated_results}
    assert by_id["spatial-distance"]["curation_subcategory"] == "distance"
    assert by_id["spatial-none"]["curation_subcategory"] == "none"
    assert by_id["affordance-stability"]["curation_subcategory"] == "grasp_stability"
    assert by_id["affordance-blockage"]["curation_subcategory"] == "object_blockage"
    assert by_id["neither"]["curation_subcategory"] is None

    assert [record["id"] for record in read_jsonl(by_label_dir / "spatial_reasoning.jsonl")] == [
        "spatial-distance",
        "spatial-none",
    ]
    assert [record["id"] for record in read_jsonl(by_label_dir / "spatial_reasoning" / "distance.jsonl")] == [
        "spatial-distance"
    ]
    assert [record["id"] for record in read_jsonl(by_label_dir / "spatial_reasoning" / "none.jsonl")] == [
        "spatial-none"
    ]
    assert [record["id"] for record in read_jsonl(by_label_dir / "affordance_understanding" / "grasp_stability.jsonl")] == [
        "affordance-stability"
    ]
    assert [record["id"] for record in read_jsonl(by_label_dir / "affordance_understanding" / "object_blockage.jsonl")] == [
        "affordance-blockage"
    ]

    updated_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert updated_summary["label_counts"] == {
        "spatial_reasoning": 2,
        "affordance_understanding": 2,
        "neither": 1,
    }
    assert updated_summary["subcategory_counts"]["spatial_reasoning"] == {
        "distance": 1,
        "direction": 0,
        "none": 1,
    }
    assert updated_summary["subcategory_counts"]["affordance_understanding"] == {
        "grasp_stability": 1,
        "object_blockage": 1,
        "none": 0,
    }


def test_relabel_output_dir_moves_goal_state_and_resets_subcategory(tmp_path):
    output_dir = tmp_path / "curation"
    by_label_dir = output_dir / "by_label"
    by_label_dir.mkdir(parents=True)
    moved_question = (
        "The robot's task is to place the mug to the left of the plate. "
        "Which configuration shows the goal state that the robot should achieve?"
    )
    results = [
        {
            "id": "goal-state",
            "question": moved_question,
            "curation_label": "spatial_reasoning",
            "curation_subcategory": "distance",
            "curation_confidence": 0.9,
            "source_dataset": "keplerccc/Robo2VLM-1",
            "source_split": "test",
            "source_index": 0,
        },
        {
            "id": "spatial-direction",
            "question": "Which colored arrow correctly shows the direction the robot will move next?",
            "curation_label": "spatial_reasoning",
            "curation_subcategory": "direction",
            "curation_confidence": 0.92,
            "source_dataset": "keplerccc/Robo2VLM-1",
            "source_split": "test",
            "source_index": 1,
        },
    ]
    write_jsonl(output_dir / "curation_results.jsonl", results)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "created_at": "2026-04-12T10:00:00+00:00",
                "dataset_name": "keplerccc/Robo2VLM-1",
                "split": "test",
                "model": "gpt-5-mini",
                "prompt_version": "spatial_affordance_v1",
                "streaming": True,
                "processed_count": 2,
                "failed_count": 0,
            }
        ),
        encoding="utf-8",
    )

    moved_count = relabel_output_dir(output_dir)

    assert moved_count == 1
    updated_results = {record["id"]: record for record in read_jsonl(output_dir / "curation_results.jsonl")}
    assert updated_results["goal-state"]["curation_label"] == "neither"
    assert updated_results["goal-state"]["curation_subcategory"] is None
    assert updated_results["goal-state"]["curation_override_reason"] == (
        f"Matched goal-state phrase override: {DEFAULT_MATCH_PHRASE}"
    )
    assert updated_results["spatial-direction"]["curation_subcategory"] == "direction"

    updated_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert updated_summary["relabel_moved_count"] == 1
    assert updated_summary["label_counts"] == {
        "spatial_reasoning": 1,
        "affordance_understanding": 0,
        "neither": 1,
    }
    assert updated_summary["subcategory_counts"]["spatial_reasoning"] == {
        "distance": 0,
        "direction": 1,
        "none": 0,
    }
