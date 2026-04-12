import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robo2vlm_curation import AFFORDANCE_LABEL, NEITHER_LABEL, SPATIAL_LABEL
from scripts.visualize_robo2vlm_question_clusters import load_question_records, sanitize_name


def test_load_question_records_requires_curation_for_each_source_index(monkeypatch):
    def fake_load_dataset(dataset_name, split, streaming):
        return iter([{"id": "a", "question": "q0"}])

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=fake_load_dataset),
    )

    with pytest.raises(ValueError, match="Missing curation label"):
        load_question_records(
            dataset_name="dummy",
            split="test",
            label_by_source_index={},
            streaming=True,
            max_samples=1,
        )


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
        label_by_source_index={
            0: NEITHER_LABEL,
            1: SPATIAL_LABEL,
            2: AFFORDANCE_LABEL,
        },
        streaming=True,
        max_samples=2,
    )

    assert [record["id"] for record in records] == ["a", "b"]
    assert [record["label"] for record in records] == [NEITHER_LABEL, SPATIAL_LABEL]
