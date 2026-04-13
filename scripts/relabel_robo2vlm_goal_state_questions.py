#!/usr/bin/env python3
"""
Move known goal-state matching questions from spatial_reasoning to neither.

The recategorization pipeline stores canonical labels in `curation_results.jsonl`
and materialized subsets in `by_label/*.jsonl`. This script applies a
deterministic post-processing override for questions that contain a known
goal-state phrase, then regenerates the subsets and summary. Subcategories are
recomputed as part of the regeneration step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robo2vlm_curation import ALLOWED_LABELS, normalize_text
from scripts.recategorize_robo2vlm import utc_now_iso
from scripts.robo2vlm_postprocess_helpers import (
    load_json,
    load_jsonl,
    regenerate_outputs,
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


def question_matches_phrase(question: str, phrase: str) -> bool:
    return normalize_text(phrase) in normalize_text(question)


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

    regenerate_outputs(
        output_dir,
        results,
        existing_summary=existing_summary,
        manifest=manifest,
        summary_updates={
            "relabel_moved_count": moved_count,
            "relabel_match_phrase": match_phrase,
            "relabel_from_label": from_label,
            "relabel_to_label": to_label,
        },
    )
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
