import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robo2vlm_curation import (
    AFFORDANCE_LABEL,
    NEITHER_LABEL,
    SPATIAL_LABEL,
    label_for_source_index,
    load_label_by_source_index,
    load_subset_source_indices,
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
