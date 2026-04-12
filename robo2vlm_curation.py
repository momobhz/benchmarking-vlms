from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SPATIAL_LABEL = "spatial_reasoning"
AFFORDANCE_LABEL = "affordance_understanding"
NEITHER_LABEL = "neither"

ALLOWED_LABELS = (SPATIAL_LABEL, AFFORDANCE_LABEL, NEITHER_LABEL)
LABEL_ORDER = [SPATIAL_LABEL, AFFORDANCE_LABEL, NEITHER_LABEL]
SUBSET_TO_LABEL = {
    "spatial": SPATIAL_LABEL,
    "affordance": AFFORDANCE_LABEL,
    "neither": NEITHER_LABEL,
}


def _coerce_source_index(value: Any, *, path: Path, line_number: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{path}:{line_number} has an invalid boolean source_index.")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"{path}:{line_number} is missing a usable integer source_index.")


def extract_curation_label(record: dict[str, Any], *, path: Path, line_number: int) -> str:
    label = record.get("curation_label", record.get("label"))
    if label not in ALLOWED_LABELS:
        raise ValueError(f"{path}:{line_number} has an invalid curation label: {label!r}")
    return str(label)


def _iter_curation_records(path: str | Path, split: str | None = None):
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Curation results file was not found: {jsonl_path}")

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{line_number} is not valid JSON.") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{jsonl_path}:{line_number} must contain a JSON object.")
            if split is not None and record.get("source_split") not in (None, split):
                continue
            yield line_number, record


def load_curation_records(path: str | Path, split: str | None = None) -> list[dict[str, Any]]:
    return [record for _, record in _iter_curation_records(path, split=split)]


def load_label_by_source_index(path: str | Path, split: str | None = None) -> dict[int, str]:
    jsonl_path = Path(path)
    label_by_source_index: dict[int, str] = {}

    for line_number, record in _iter_curation_records(jsonl_path, split=split):
        source_index = _coerce_source_index(
            record.get("source_index"),
            path=jsonl_path,
            line_number=line_number,
        )
        label = extract_curation_label(record, path=jsonl_path, line_number=line_number)

        previous = label_by_source_index.get(source_index)
        if previous is not None and previous != label:
            raise ValueError(
                f"{jsonl_path}:{line_number} conflicts with an existing label for source_index "
                f"{source_index}: {previous!r} vs {label!r}"
            )
        label_by_source_index[source_index] = label

    return dict(sorted(label_by_source_index.items()))


def label_for_source_index(source_index: int, label_by_source_index: dict[int, str]) -> str:
    label = label_by_source_index.get(source_index)
    if label is None:
        raise ValueError(
            f"Missing curation label for source_index {source_index}. "
            "Regenerate or complete the curation_results.jsonl file for this split."
        )
    return label


def load_subset_source_indices(
    path: str | Path,
    subset: str,
    split: str | None = None,
) -> list[int]:
    if subset not in SUBSET_TO_LABEL:
        raise ValueError(f"Unsupported subset: {subset!r}")

    target_label = SUBSET_TO_LABEL[subset]
    label_by_source_index = load_label_by_source_index(path, split=split)
    return [
        source_index
        for source_index, label in label_by_source_index.items()
        if label == target_label
    ]
