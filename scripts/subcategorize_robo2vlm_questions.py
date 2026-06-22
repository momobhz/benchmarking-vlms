#!/usr/bin/env python3
"""
Assign deterministic subcategories to curated Robo2VLM questions.

This script reads an existing curation output directory, assigns
`curation_subcategory` from the top-level curation label and question text,
rewrites `curation_results.jsonl`, regenerates `by_label/*.jsonl`, creates
nested per-label subcategory files, and rebuilds `summary.json`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for import_path in (REPO_ROOT, SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from vlm_bench.curation.taxonomy import ALLOWED_LABELS, infer_curation_subcategory
from scripts.robo2vlm_postprocess_helpers import (
    load_json,
    load_jsonl,
    regenerate_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign deterministic Robo2VLM subcategories based on the question text "
            "and regenerate the subset files."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory produced by scripts/recategorize_robo2vlm.py.",
    )
    return parser.parse_args()


def subcategorize_output_dir(output_dir: Path) -> dict[str, int]:
    results_path = output_dir / "curation_results.jsonl"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"

    results = load_jsonl(results_path)
    existing_summary = load_json(summary_path)
    manifest = load_json(manifest_path)

    counts = {label: 0 for label in ALLOWED_LABELS}
    for record in results:
        label = record.get("curation_label")
        question = record.get("question")
        previous = record.get("curation_subcategory")
        updated = infer_curation_subcategory(label, question) if isinstance(label, str) and isinstance(question, str) else None
        if previous != updated and label in counts:
            counts[label] += 1
        record["curation_subcategory"] = updated

    regenerate_outputs(
        output_dir,
        results,
        existing_summary=existing_summary,
        manifest=manifest,
    )
    return counts


def main() -> int:
    args = parse_args()
    updated_counts = subcategorize_output_dir(Path(args.output_dir))
    print(
        "Updated subcategories: "
        + ", ".join(f"{label}={count}" for label, count in updated_counts.items()),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
