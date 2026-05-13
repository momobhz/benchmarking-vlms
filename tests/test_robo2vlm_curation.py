import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_bench.curation.taxonomy import (
    AFFORDANCE_LABEL,
    DISTANCE_SUBCATEGORY,
    DIRECTION_SUBCATEGORY,
    GRASP_STABILITY_SUBCATEGORY,
    NEITHER_LABEL,
    NONE_SUBCATEGORY,
    OBJECT_BLOCKAGE_SUBCATEGORY,
    SPATIAL_LABEL,
    infer_curation_subcategory,
    label_for_source_index,
    load_label_by_source_index,
    load_subset_source_indices,
    load_subcategory_by_source_index,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def test_load_label_by_source_index_reads_curated_labels(tmp_path):
    path = tmp_path / "curation_results.jsonl"
    write_jsonl(
        path,
        [
            {"source_index": 2, "curation_label": SPATIAL_LABEL, "source_split": "test"},
            {"source_index": 1, "curation_label": AFFORDANCE_LABEL, "source_split": "test"},
            {"source_index": 0, "curation_label": NEITHER_LABEL, "source_split": "train"},
        ],
    )

    labels = load_label_by_source_index(path, split="test")
    assert labels == {1: AFFORDANCE_LABEL, 2: SPATIAL_LABEL}


def test_label_for_source_index_raises_for_missing_label():
    with pytest.raises(ValueError, match="Missing curation label"):
        label_for_source_index(7, {1: SPATIAL_LABEL})


def test_load_subset_source_indices_filters_by_subset(tmp_path):
    path = tmp_path / "curation_results.jsonl"
    write_jsonl(
        path,
        [
            {"source_index": 3, "curation_label": AFFORDANCE_LABEL, "source_split": "test"},
            {"source_index": 1, "curation_label": SPATIAL_LABEL, "source_split": "test"},
            {"source_index": 2, "curation_label": SPATIAL_LABEL, "source_split": "test"},
        ],
    )

    assert load_subset_source_indices(path, "spatial", split="test") == [1, 2]
    assert load_subset_source_indices(path, "affordance", split="test") == [3]


def test_infer_curation_subcategory_uses_signal_words():
    assert (
        infer_curation_subcategory(
            SPATIAL_LABEL,
            "In the image, which colored point is closest to the camera?",
        )
        == DISTANCE_SUBCATEGORY
    )
    assert (
        infer_curation_subcategory(
            SPATIAL_LABEL,
            "Which colored arrow correctly shows the direction the robot will move next?",
        )
        == DIRECTION_SUBCATEGORY
    )
    assert (
        infer_curation_subcategory(
            AFFORDANCE_LABEL,
            "Is the robot's grasp of the peg stable?",
        )
        == GRASP_STABILITY_SUBCATEGORY
    )
    assert (
        infer_curation_subcategory(
            AFFORDANCE_LABEL,
            "Is there any obstacle blocking the robot from reaching peg?",
        )
        == OBJECT_BLOCKAGE_SUBCATEGORY
    )
    assert infer_curation_subcategory(SPATIAL_LABEL, "Where is the mug?") == NONE_SUBCATEGORY
    assert infer_curation_subcategory(NEITHER_LABEL, "Is the gripper open?") is None


def test_load_subcategory_by_source_index_and_prefixed_subsets(tmp_path):
    path = tmp_path / "curation_results.jsonl"
    write_jsonl(
        path,
        [
            {
                "source_index": 1,
                "source_split": "test",
                "question": "Which colored point is closest to the camera?",
                "curation_label": SPATIAL_LABEL,
                "curation_subcategory": DISTANCE_SUBCATEGORY,
            },
            {
                "source_index": 2,
                "source_split": "test",
                "question": "Is there any obstacle blocking the robot from reaching peg?",
                "curation_label": AFFORDANCE_LABEL,
                "curation_subcategory": OBJECT_BLOCKAGE_SUBCATEGORY,
            },
            {
                "source_index": 3,
                "source_split": "test",
                "question": "Has the robot completed the task successfully?",
                "curation_label": NEITHER_LABEL,
                "curation_subcategory": None,
            },
        ],
    )

    assert load_subcategory_by_source_index(path, split="test") == {
        1: DISTANCE_SUBCATEGORY,
        2: OBJECT_BLOCKAGE_SUBCATEGORY,
        3: None,
    }
    assert load_subset_source_indices(path, "spatial_distance", split="test") == [1]
    assert load_subset_source_indices(path, "affordance_object_blockage", split="test") == [2]


def test_load_label_by_source_index_rejects_conflicting_duplicates(tmp_path):
    path = tmp_path / "curation_results.jsonl"
    write_jsonl(
        path,
        [
            {"source_index": 5, "curation_label": SPATIAL_LABEL, "source_split": "test"},
            {"source_index": 5, "curation_label": AFFORDANCE_LABEL, "source_split": "test"},
        ],
    )

    with pytest.raises(ValueError, match="conflicts with an existing label"):
        load_label_by_source_index(path, split="test")
