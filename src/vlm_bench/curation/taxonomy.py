from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SPATIAL_LABEL = "spatial_reasoning"
AFFORDANCE_LABEL = "affordance_understanding"
NEITHER_LABEL = "neither"
DISTANCE_SUBCATEGORY = "distance"
DIRECTION_SUBCATEGORY = "direction"
GRASP_STABILITY_SUBCATEGORY = "grasp_stability"
OBJECT_BLOCKAGE_SUBCATEGORY = "object_blockage"
NONE_SUBCATEGORY = "none"
NO_SUBCATEGORY_GROUP = "no_subcategory"

ALLOWED_LABELS = (SPATIAL_LABEL, AFFORDANCE_LABEL, NEITHER_LABEL)
LABEL_ORDER = [SPATIAL_LABEL, AFFORDANCE_LABEL, NEITHER_LABEL]
GROUP_BY_CHOICES = ("label", "subcategory")
LABEL_TO_SUBCATEGORY_ORDER = {
    SPATIAL_LABEL: [DISTANCE_SUBCATEGORY, DIRECTION_SUBCATEGORY, NONE_SUBCATEGORY],
    AFFORDANCE_LABEL: [
        GRASP_STABILITY_SUBCATEGORY,
        OBJECT_BLOCKAGE_SUBCATEGORY,
        NONE_SUBCATEGORY,
    ],
    NEITHER_LABEL: [],
}
SUBCATEGORY_ORDER = [
    DISTANCE_SUBCATEGORY,
    DIRECTION_SUBCATEGORY,
    GRASP_STABILITY_SUBCATEGORY,
    OBJECT_BLOCKAGE_SUBCATEGORY,
    NONE_SUBCATEGORY,
    NO_SUBCATEGORY_GROUP,
]
SUBCATEGORY_RULES = {
    SPATIAL_LABEL: {
        DISTANCE_SUBCATEGORY: "point",
        DIRECTION_SUBCATEGORY: "which colored arrow",
    },
    AFFORDANCE_LABEL: {
        GRASP_STABILITY_SUBCATEGORY: "stable",
        OBJECT_BLOCKAGE_SUBCATEGORY: "obstacle blocking",
    },
    NEITHER_LABEL: {},
}
SUBSET_TO_LABEL = {
    "spatial": SPATIAL_LABEL,
    "affordance": AFFORDANCE_LABEL,
    "neither": NEITHER_LABEL,
    "spatial_distance": SPATIAL_LABEL,
    "spatial_direction": SPATIAL_LABEL,
    "spatial_none": SPATIAL_LABEL,
    "affordance_grasp_stability": AFFORDANCE_LABEL,
    "affordance_object_blockage": AFFORDANCE_LABEL,
    "affordance_none": AFFORDANCE_LABEL,
}
SUBSET_TO_SUBCATEGORY = {
    "spatial_distance": DISTANCE_SUBCATEGORY,
    "spatial_direction": DIRECTION_SUBCATEGORY,
    "spatial_none": NONE_SUBCATEGORY,
    "affordance_grasp_stability": GRASP_STABILITY_SUBCATEGORY,
    "affordance_object_blockage": OBJECT_BLOCKAGE_SUBCATEGORY,
    "affordance_none": NONE_SUBCATEGORY,
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


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def infer_curation_subcategory(label: str, question: str) -> str | None:
    if label == NEITHER_LABEL:
        return None

    normalized_question = normalize_text(question)
    for subcategory, phrase in SUBCATEGORY_RULES[label].items():
        if normalize_text(phrase) in normalized_question:
            return subcategory
    return NONE_SUBCATEGORY


def is_valid_curation_subcategory(label: str, subcategory: str | None) -> bool:
    if label == NEITHER_LABEL:
        return subcategory is None
    return subcategory in LABEL_TO_SUBCATEGORY_ORDER[label]


def extract_curation_subcategory(
    record: dict[str, Any],
    *,
    path: Path,
    line_number: int,
) -> str | None:
    label = extract_curation_label(record, path=path, line_number=line_number)
    question = record.get("question")
    raw_subcategory = record.get("curation_subcategory", ...)

    if raw_subcategory is ...:
        if not isinstance(question, str):
            if label == NEITHER_LABEL:
                return None
            raise ValueError(
                f"{path}:{line_number} is missing question text required to derive a subcategory."
            )
        return infer_curation_subcategory(label, question)

    if raw_subcategory is None:
        if label == NEITHER_LABEL:
            return None
        raise ValueError(
            f"{path}:{line_number} has null subcategory for non-neither label {label!r}."
        )

    if not isinstance(raw_subcategory, str):
        raise ValueError(f"{path}:{line_number} has an invalid curation subcategory.")

    subcategory = raw_subcategory.strip()
    if not is_valid_curation_subcategory(label, subcategory):
        raise ValueError(
            f"{path}:{line_number} has invalid subcategory {subcategory!r} for label {label!r}."
        )
    return subcategory


def grouping_value_for_record(
    record: dict[str, Any],
    *,
    path: Path,
    line_number: int,
    group_by: str,
) -> str:
    if group_by not in GROUP_BY_CHOICES:
        raise ValueError(f"Unsupported group_by value: {group_by!r}")

    if group_by == "label":
        return extract_curation_label(record, path=path, line_number=line_number)

    subcategory = extract_curation_subcategory(record, path=path, line_number=line_number)
    return subcategory or NO_SUBCATEGORY_GROUP


def empty_subcategory_counts() -> dict[str, dict[str, int]]:
    return {
        label: {subcategory: 0 for subcategory in LABEL_TO_SUBCATEGORY_ORDER[label]}
        for label in ALLOWED_LABELS
    }


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
    return load_group_by_source_index(path, split=split, group_by="label")


def load_subcategory_by_source_index(
    path: str | Path,
    split: str | None = None,
) -> dict[int, str | None]:
    jsonl_path = Path(path)
    subcategory_by_source_index: dict[int, str | None] = {}
    missing = object()

    for line_number, record in _iter_curation_records(jsonl_path, split=split):
        source_index = _coerce_source_index(
            record.get("source_index"),
            path=jsonl_path,
            line_number=line_number,
        )
        subcategory = extract_curation_subcategory(record, path=jsonl_path, line_number=line_number)
        previous = subcategory_by_source_index.get(source_index, missing)
        if previous is not missing and previous != subcategory:
            raise ValueError(
                f"{jsonl_path}:{line_number} conflicts with an existing subcategory for "
                f"source_index {source_index}: {previous!r} vs {subcategory!r}"
            )
        subcategory_by_source_index[source_index] = subcategory

    return dict(sorted(subcategory_by_source_index.items()))


def load_group_by_source_index(
    path: str | Path,
    split: str | None = None,
    group_by: str = "label",
) -> dict[int, str]:
    jsonl_path = Path(path)
    group_by_source_index: dict[int, str] = {}
    missing = object()

    for line_number, record in _iter_curation_records(jsonl_path, split=split):
        source_index = _coerce_source_index(
            record.get("source_index"),
            path=jsonl_path,
            line_number=line_number,
        )
        group_value = grouping_value_for_record(
            record,
            path=jsonl_path,
            line_number=line_number,
            group_by=group_by,
        )

        previous = group_by_source_index.get(source_index, missing)
        if previous is not missing and previous != group_value:
            raise ValueError(
                f"{jsonl_path}:{line_number} conflicts with an existing {group_by} for "
                f"source_index {source_index}: {previous!r} vs {group_value!r}"
            )
        group_by_source_index[source_index] = group_value

    return dict(sorted(group_by_source_index.items()))


def label_for_source_index(source_index: int, label_by_source_index: dict[int, str]) -> str:
    label = label_by_source_index.get(source_index)
    if label is None:
        raise ValueError(
            f"Missing curation label for source_index {source_index}. "
            "Regenerate or complete the curation_results.jsonl file for this split."
        )
    return label


def group_for_source_index(source_index: int, group_by_source_index: dict[int, str], group_by: str) -> str:
    group_value = group_by_source_index.get(source_index)
    if group_value is None:
        raise ValueError(
            f"Missing curation {group_by} for source_index {source_index}. "
            "Regenerate or complete the curation_results.jsonl file for this split."
        )
    return group_value


def load_subset_source_indices(
    path: str | Path,
    subset: str,
    split: str | None = None,
) -> list[int]:
    if subset not in SUBSET_TO_LABEL:
        raise ValueError(f"Unsupported subset: {subset!r}")

    target_label = SUBSET_TO_LABEL[subset]
    subcategory_filter = SUBSET_TO_SUBCATEGORY.get(subset)
    jsonl_path = Path(path)
    indices: list[int] = []

    for line_number, record in _iter_curation_records(jsonl_path, split=split):
        source_index = _coerce_source_index(
            record.get("source_index"),
            path=jsonl_path,
            line_number=line_number,
        )
        label = extract_curation_label(record, path=jsonl_path, line_number=line_number)
        if label != target_label:
            continue
        if subcategory_filter is None:
            indices.append(source_index)
            continue
        subcategory = extract_curation_subcategory(record, path=jsonl_path, line_number=line_number)
        if subcategory == subcategory_filter:
            indices.append(source_index)
    return indices
