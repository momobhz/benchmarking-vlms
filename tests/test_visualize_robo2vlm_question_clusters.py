import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.visualize_robo2vlm_question_clusters import (
    AFFORDANCE_LABEL,
    NEITHER_LABEL,
    SPATIAL_LABEL,
    label_for_index,
    load_question_records,
    sanitize_name,
    validate_label_sets,
)


def test_label_for_index_uses_existing_cluster_assignments():
    spatial_indices = {2, 5}
    affordance_indices = {1, 4}

    assert label_for_index(2, spatial_indices, affordance_indices) == SPATIAL_LABEL
    assert label_for_index(4, spatial_indices, affordance_indices) == AFFORDANCE_LABEL
    assert label_for_index(9, spatial_indices, affordance_indices) == NEITHER_LABEL


def test_validate_label_sets_rejects_overlap():
    with pytest.raises(ValueError):
        validate_label_sets({1, 2}, {2, 3})


def test_sanitize_name_replaces_model_separators():
    assert sanitize_name("sentence-transformers/all-MiniLM-L6-v2") == (
        "sentence-transformers_all-MiniLM-L6-v2"
    )


def test_load_question_records_streaming_respects_max_samples(monkeypatch):
    rows = [
        {"id": "a", "question": "q0"},
        {"id": "b", "question": "q1"},
        {"id": "c", "question": "q2"},
    ]

    def fake_load_dataset(dataset_name, split, streaming):
        assert dataset_name == "dummy"
        assert split == "test"
        assert streaming is True
        return iter(rows)

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=fake_load_dataset),
    )

    records = load_question_records(
        dataset_name="dummy",
        split="test",
        spatial_indices={1},
        affordance_indices={2},
        streaming=True,
        max_samples=2,
    )

    assert [record["id"] for record in records] == ["a", "b"]
    assert [record["label"] for record in records] == [NEITHER_LABEL, SPATIAL_LABEL]
