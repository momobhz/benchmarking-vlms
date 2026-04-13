from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from robo2vlm_curation import ALLOWED_LABELS, LABEL_TO_SUBCATEGORY_ORDER, infer_curation_subcategory
from scripts.recategorize_robo2vlm import count_jsonl_rows, summarize_results, utc_now_iso


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


def apply_subcategories(results: List[Dict[str, Any]]) -> None:
    for record in results:
        label = record.get("curation_label")
        question = record.get("question")
        if not isinstance(label, str):
            continue
        if not isinstance(question, str):
            record["curation_subcategory"] = None if label == "neither" else "none"
            continue
        record["curation_subcategory"] = infer_curation_subcategory(label, question)


def rewrite_label_subsets(output_dir: Path, results: List[Dict[str, Any]]) -> None:
    subset_dir = output_dir / "by_label"
    subset_dir.mkdir(parents=True, exist_ok=True)

    for label in ALLOWED_LABELS:
        label_records = [record for record in results if record.get("curation_label") == label]
        write_jsonl(subset_dir / f"{label}.jsonl", label_records)

        nested_dir = subset_dir / label
        if nested_dir.exists():
            shutil.rmtree(nested_dir)
        subcategories = LABEL_TO_SUBCATEGORY_ORDER[label]
        if not subcategories:
            continue

        nested_dir.mkdir(parents=True, exist_ok=True)
        for subcategory in subcategories:
            subcategory_records = [
                record
                for record in label_records
                if record.get("curation_subcategory") == subcategory
            ]
            write_jsonl(nested_dir / f"{subcategory}.jsonl", subcategory_records)


def rebuild_summary(
    *,
    output_dir: Path,
    results: List[Dict[str, Any]],
    existing_summary: Optional[Dict[str, Any]],
    manifest: Optional[Dict[str, Any]],
    summary_updates: Optional[Dict[str, Any]] = None,
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
    if summary_updates:
        summary.update(summary_updates)
    return summary


def regenerate_outputs(
    output_dir: Path,
    results: List[Dict[str, Any]],
    *,
    existing_summary: Optional[Dict[str, Any]],
    manifest: Optional[Dict[str, Any]],
    summary_updates: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    apply_subcategories(results)

    results_path = output_dir / "curation_results.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(results_path, results)
    rewrite_label_subsets(output_dir, results)

    summary = rebuild_summary(
        output_dir=output_dir,
        results=results,
        existing_summary=existing_summary,
        manifest=manifest,
        summary_updates=summary_updates,
    )
    write_json(summary_path, summary)
    return summary
