#!/usr/bin/env python3
"""
Move known goal-state matching questions from spatial_reasoning to neither.

The recategorization pipeline stores the canonical labels in
`curation_results.jsonl` and materialized subsets in `by_label/*.jsonl`.
This script applies a deterministic post-processing override for questions
that contain a known goal-state phrase, rewrites the canonical results, then
regenerates the per-label subsets and summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from recategorize_robo2vlm import (
        ALLOWED_LABELS,
        count_jsonl_rows,
        summarize_results,
        utc_now_iso,
    )
except ImportError:  # pragma: no cover - used when imported as scripts.<module> in tests
    from scripts.recategorize_robo2vlm import (
        ALLOWED_LABELS,
        count_jsonl_rows,
        summarize_results,
        utc_now_iso,
    )


DEFAULT_MATCH_PHRASE = (
    "Which configuration shows the goal state that the robot should achieve?"
)
DEFAULT_FROM_LABEL = "spatial_reasoning"
DEFAULT_TO_LABEL = "neither"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Relabel goal-state matching Robo2VLM questions from spatial_reasoning "
            "to neither and regenerate the subset files."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory produced by scripts/recategorize_robo2vlm.py.",
    )
    parser.add_argument(
        "--match-phrase",
        default=DEFAULT_MATCH_PHRASE,
        help="Phrase that identifies the misclassified questions.",
    )
    parser.add_argument(
        "--from-label",
        default=DEFAULT_FROM_LABEL,
        choices=ALLOWED_LABELS,
        help="Only move records that currently have this label.",
    )
    parser.add_argument(
        "--to-label",
        default=DEFAULT_TO_LABEL,
        choices=ALLOWED_LABELS,
        help="Destination label for matched records.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def question_matches_phrase(question: str, phrase: str) -> bool:
    return normalize_text(phrase) in normalize_text(question)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL in {path} at line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise SystemExit(f"Expected object in {path} at line {line_number}.")
            records.append(payload)
    return records


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected top-level object in {path}.")
    return payload


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    temp_path = path.parent / f".{path.name}.tmp"
    with temp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")
    temp_path.replace(path)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    temp_path = path.parent / f".{path.name}.tmp"
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path.replace(path)


def build_summary_args(
    results: List[Dict[str, Any]],
    existing_summary: Optional[Dict[str, Any]],
    manifest: Optional[Dict[str, Any]],
) -> argparse.Namespace:
    metadata: Dict[str, Any] = {}
    for source in (manifest, existing_summary):
        if isinstance(source, dict):
            metadata.update(source)

    first_record = results[0] if results else {}
    return argparse.Namespace(
        dataset_name=metadata.get("dataset_name") or first_record.get("source_dataset") or "unknown",
        split=metadata.get("split") or first_record.get("source_split") or "unknown",
        model=metadata.get("model") or first_record.get("curation_model") or "unknown",
        prompt_version=(
            metadata.get("prompt_version")
            or first_record.get("curation_prompt_version")
            or "unknown"
        ),
        streaming=bool(metadata.get("streaming", False)),
    )


def rewrite_label_subsets(output_dir: Path, results: List[Dict[str, Any]]) -> None:
    subset_dir = output_dir / "by_label"
    subset_dir.mkdir(parents=True, exist_ok=True)

    for label in ALLOWED_LABELS:
        label_records = [record for record in results if record.get("curation_label") == label]
        write_jsonl(subset_dir / f"{label}.jsonl", label_records)


def rebuild_summary(
    *,
    output_dir: Path,
    results: List[Dict[str, Any]],
    existing_summary: Optional[Dict[str, Any]],
    manifest: Optional[Dict[str, Any]],
    moved_count: int,
    match_phrase: str,
    from_label: str,
    to_label: str,
) -> Dict[str, Any]:
    args = build_summary_args(results, existing_summary, manifest)
    failed_path = output_dir / "failed_records.jsonl"
    failed_count = (
        int(existing_summary["failed_count"])
        if existing_summary and "failed_count" in existing_summary
        else count_jsonl_rows(failed_path)
    )
    summary = summarize_results(results, failed_count=failed_count, args=args)

    if existing_summary:
        for key in ("new_count", "resume_skipped_count", "runtime_seconds"):
            if key in existing_summary:
                summary[key] = existing_summary[key]
        if "created_at" in existing_summary:
            summary["created_at"] = existing_summary["created_at"]

    summary["updated_at"] = utc_now_iso()
    summary["relabel_moved_count"] = moved_count
    summary["relabel_match_phrase"] = match_phrase
    summary["relabel_from_label"] = from_label
    summary["relabel_to_label"] = to_label
    return summary


def relabel_output_dir(
    output_dir: Path,
    *,
    match_phrase: str = DEFAULT_MATCH_PHRASE,
    from_label: str = DEFAULT_FROM_LABEL,
    to_label: str = DEFAULT_TO_LABEL,
) -> int:
    if from_label == to_label:
        raise SystemExit("--from-label and --to-label must be different.")

    results_path = output_dir / "curation_results.jsonl"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"

    results = load_jsonl(results_path)
    existing_summary = load_json(summary_path)
    manifest = load_json(manifest_path)

    override_timestamp = utc_now_iso()
    override_reason = f"Matched goal-state phrase override: {match_phrase}"
    moved_count = 0

    for record in results:
        question = record.get("question")
        label = record.get("curation_label")
        if not isinstance(question, str):
            continue
        if label != from_label:
            continue
        if not question_matches_phrase(question, match_phrase):
            continue

        record["curation_label"] = to_label
        record["curation_override_from_label"] = from_label
        record["curation_override_to_label"] = to_label
        record["curation_override_reason"] = override_reason
        record["curation_override_timestamp"] = override_timestamp
        moved_count += 1

    if moved_count == 0:
        return 0

    write_jsonl(results_path, results)
    rewrite_label_subsets(output_dir, results)
    summary = rebuild_summary(
        output_dir=output_dir,
        results=results,
        existing_summary=existing_summary,
        manifest=manifest,
        moved_count=moved_count,
        match_phrase=match_phrase,
        from_label=from_label,
        to_label=to_label,
    )
    write_json(summary_path, summary)
    return moved_count


def main() -> int:
    args = parse_args()
    moved_count = relabel_output_dir(
        Path(args.output_dir),
        match_phrase=args.match_phrase,
        from_label=args.from_label,
        to_label=args.to_label,
    )
    if moved_count == 0:
        print("No matching questions required relabeling.", flush=True)
        return 0

    print(
        f"Moved {moved_count} question(s) from {args.from_label} to {args.to_label}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
