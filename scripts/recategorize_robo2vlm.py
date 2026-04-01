#!/usr/bin/env python3
"""
Recategorize Robo2VLM questions into spatial reasoning, affordance understanding,
or neither using an OpenAI-backed LLM curator.

The script reads the published Hugging Face dataset by default, classifies each
question with a structured JSON output schema, and writes:

- curation_results.jsonl: one labeled record per question
- by_label/<label>.jsonl: filtered subsets for downstream evaluation
- summary.json: aggregate counts and metadata

Images are not duplicated into the output. Use the question `id` to join the
labels back with the original dataset for VLM evaluation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request


DEFAULT_DATASET_NAME = "keplerccc/Robo2VLM-1"
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_PROMPT_VERSION = "spatial_affordance_v1"
ALLOWED_LABELS = (
    "spatial_reasoning",
    "affordance_understanding",
    "neither",
)
SKIPPED_SOURCE_FIELDS = {
    "id",
    "image",
    "images",
    "choice_images",
    "question",
    "choices",
    "correct_answer",
    "correct_answer_idx",
    "source_dataset",
    "source_split",
    "source_index",
    "original_tag",
    "cache_key",
}

CURATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "label": {
            "type": "string",
            "enum": list(ALLOWED_LABELS),
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "rationale": {
            "type": "string",
            "minLength": 1,
        },
        "signals": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
    },
    "required": ["label", "confidence", "rationale", "signals"],
}

SYSTEM_PROMPT = """
You are curating a robotics VQA benchmark.

Your job is to assign exactly one label to each question:

1. spatial_reasoning
   Use this when the question primarily evaluates geometric or spatial relations:
   relative position, direction, depth, distance, correspondence across views,
   or movement direction in 2D/3D space.

2. affordance_understanding
   Use this when the question primarily evaluates whether an interaction is
   possible, blocked, stable, or physically permitted: reachability, graspability,
   blockage, obstacle interference, or stable manipulation.

3. neither
   Use this for task success, temporal order, action phase, goal-state matching,
   task identification, language instruction matching, or simple robot state
   questions that are not primarily about spatial relations or affordances.

Decision rules:
- Choose the dominant evaluation target, not every concept mentioned.
- A question about "where is X relative to Y" is spatial_reasoning.
- A question about "can the robot reach or stably grasp X" is affordance_understanding.
- A question about "is the gripper open" is neither.
- A question about phases, next action, success, or overall task is neither.
- Original template tags are optional context only. They are not authoritative.

Return only JSON that satisfies the provided schema.
""".strip()


class CurationError(RuntimeError):
    """Raised when a question cannot be curated or the response is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recategorize Robo2VLM questions into spatial, affordance, or neither."
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Hugging Face dataset name to load (default: %(default)s).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to read (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for JSONL labels, filtered subsets, and summary files.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        help="OpenAI model name for the curator (default: %(default)s).",
    )
    parser.add_argument(
        "--prompt-version",
        default=DEFAULT_PROMPT_VERSION,
        help="Version string stored with each curation result.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable that stores the OpenAI API key.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="Base URL for the OpenAI-compatible Responses API.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=90,
        help="HTTP timeout per API call.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Maximum retries for a failed curator API call.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=2.0,
        help="Base backoff in seconds for retries.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Progress/logging checkpoint interval in processed examples.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Process at most this many source examples after start/end filtering.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Inclusive zero-based source index to start from.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Exclusive zero-based source index to stop at.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output directory by skipping completed cache keys.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete prior output files in the target directory before running.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Disable datasets streaming mode.",
    )
    parser.set_defaults(streaming=True)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_dataset_split(dataset_name: str, split: str, streaming: bool) -> Iterable[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The 'datasets' package is required. Install it with 'pip install datasets'."
        ) from exc

    return load_dataset(dataset_name, split=split, streaming=streaming)


def is_json_safe(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, list):
        return all(is_json_safe(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and is_json_safe(item) for key, item in value.items())
    return False


def extract_original_tag(source_row: Dict[str, Any]) -> Optional[str]:
    if isinstance(source_row.get("tag"), str):
        return source_row["tag"]

    metadata = source_row.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("tag"), str):
        return metadata["tag"]

    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict) and isinstance(parsed.get("tag"), str):
            return parsed["tag"]

    return None


def normalize_choices(choices: Any) -> List[str]:
    if choices is None:
        return []
    if isinstance(choices, list):
        normalized = []
        for choice in choices:
            if isinstance(choice, str):
                normalized.append(choice)
            elif isinstance(choice, dict) and "text" in choice:
                normalized.append(str(choice["text"]))
            else:
                normalized.append(str(choice))
        return normalized
    return [str(choices)]


def copy_serializable_source_fields(source_row: Dict[str, Any]) -> Dict[str, Any]:
    copied: Dict[str, Any] = {}
    for key, value in source_row.items():
        if key in SKIPPED_SOURCE_FIELDS:
            continue
        if is_json_safe(value):
            copied[key] = copy.deepcopy(value)
    return copied


def normalize_source_row(
    source_row: Dict[str, Any],
    dataset_name: str,
    split: str,
    source_index: int,
) -> Dict[str, Any]:
    question = source_row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise CurationError(f"Source row at index {source_index} is missing a usable question.")

    normalized = {
        "id": str(source_row.get("id", f"{split}_{source_index}")),
        "question": question.strip(),
        "choices": normalize_choices(source_row.get("choices")),
        "correct_answer": source_row.get("correct_answer", source_row.get("correct_answer_idx")),
        "source_dataset": dataset_name,
        "source_split": split,
        "source_index": source_index,
        "original_tag": extract_original_tag(source_row),
    }
    normalized.update(copy_serializable_source_fields(source_row))
    return normalized


def build_user_prompt(record: Dict[str, Any]) -> str:
    choices = record.get("choices") or []
    if choices:
        formatted_choices = "\n".join(
            f"{idx + 1}. {choice}" for idx, choice in enumerate(choices)
        )
    else:
        formatted_choices = "No answer choices were provided."

    optional_context = (
        f"Original template tag: {record['original_tag']}\n"
        if record.get("original_tag")
        else ""
    )
    return (
        "Curate the following robotics VQA item.\n\n"
        f"Question ID: {record['id']}\n"
        f"{optional_context}"
        f"Question: {record['question']}\n"
        "Choices:\n"
        f"{formatted_choices}\n\n"
        "Classify the question according to the taxonomy."
    )


def make_cache_key(record: Dict[str, Any], model: str, prompt_version: str) -> str:
    digest = hashlib.sha256()
    digest.update(record["id"].encode("utf-8"))
    digest.update(b"\0")
    digest.update(record["question"].encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(record.get("choices", []), sort_keys=True).encode("utf-8"))
    digest.update(b"\0")
    digest.update(model.encode("utf-8"))
    digest.update(b"\0")
    digest.update(prompt_version.encode("utf-8"))
    return digest.hexdigest()


def extract_response_text(response_json: Dict[str, Any]) -> str:
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    outputs = response_json.get("output", [])
    for output_item in outputs:
        if not isinstance(output_item, dict):
            continue
        content_items = output_item.get("content", [])
        for content_item in content_items:
            if not isinstance(content_item, dict):
                continue
            text_value = content_item.get("text")
            if isinstance(text_value, str) and text_value.strip():
                return text_value

    raise CurationError("Responses API returned no text content.")


def validate_curator_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    label = payload.get("label")
    if label not in ALLOWED_LABELS:
        raise CurationError(f"Invalid label: {label!r}")

    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        raise CurationError(f"Invalid confidence: {confidence!r}")

    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise CurationError("Missing rationale.")

    signals = payload.get("signals")
    if not isinstance(signals, list) or not signals:
        raise CurationError("Signals must be a non-empty list.")

    cleaned_signals = []
    for signal in signals:
        if not isinstance(signal, str) or not signal.strip():
            raise CurationError(f"Invalid signal: {signal!r}")
        cleaned_signals.append(signal.strip())

    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "rationale": rationale.strip(),
        "signals": cleaned_signals,
    }


def call_openai_responses_api(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt_version: str,
    record: Dict[str, Any],
    timeout_seconds: int,
) -> Dict[str, Any]:
    user_prompt = build_user_prompt(record)
    payload = {
        "model": model,
        "temperature": 0,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "robo2vlm_question_curation",
                "description": (
                    "Categorize a robotics VQA question as spatial reasoning, "
                    "affordance understanding, or neither."
                ),
                "strict": True,
                "schema": CURATION_SCHEMA,
            }
        },
        "metadata": {
            "benchmark": "Robo2VLM-1",
            "question_id": record["id"],
            "prompt_version": prompt_version,
        },
    }
    request_body = json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(
        url=f"{base_url.rstrip('/')}/responses",
        data=request_body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise CurationError(
            f"Responses API HTTP {exc.code}: {detail.strip() or exc.reason}"
        ) from exc
    except urllib_error.URLError as exc:
        raise CurationError(f"Responses API request failed: {exc}") from exc

    try:
        response_json = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise CurationError(f"Responses API returned non-JSON content: {response_body[:200]}") from exc

    raw_payload = extract_response_text(response_json)
    try:
        structured_payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise CurationError(f"Structured output was not valid JSON: {raw_payload!r}") from exc

    return validate_curator_payload(structured_payload)


def curate_record_with_retries(
    *,
    api_key: str,
    args: argparse.Namespace,
    record: Dict[str, Any],
) -> Tuple[Dict[str, Any], int, float]:
    attempts = 0
    started_at = time.perf_counter()
    while True:
        attempts += 1
        try:
            curated = call_openai_responses_api(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                prompt_version=args.prompt_version,
                record=record,
                timeout_seconds=args.timeout_seconds,
            )
            latency = time.perf_counter() - started_at
            return curated, attempts, latency
        except CurationError:
            if attempts >= args.max_retries:
                latency = time.perf_counter() - started_at
                raise
            sleep_seconds = args.retry_backoff_seconds * (2 ** (attempts - 1))
            time.sleep(sleep_seconds)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True))
        handle.write("\n")


def load_existing_results(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}

    existing: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CurationError(
                    f"Invalid JSONL in {path} at line {line_number}: {exc}"
                ) from exc

            cache_key = payload.get("cache_key")
            if not isinstance(cache_key, str) or not cache_key:
                raise CurationError(
                    f"Missing cache_key in {path} at line {line_number}."
                )
            existing[cache_key] = payload
    return existing


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0

    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def iter_normalized_records(args: argparse.Namespace) -> Iterator[Dict[str, Any]]:
    dataset = load_dataset_split(args.dataset_name, args.split, args.streaming)
    yielded = 0
    for source_index, source_row in enumerate(dataset):
        if source_index < args.start_index:
            continue
        if args.end_index is not None and source_index >= args.end_index:
            break
        if args.max_samples is not None and yielded >= args.max_samples:
            break
        yield normalize_source_row(source_row, args.dataset_name, args.split, source_index)
        yielded += 1


def build_result_record(
    *,
    record: Dict[str, Any],
    curated: Dict[str, Any],
    cache_key: str,
    model: str,
    prompt_version: str,
    attempts: int,
    latency_seconds: float,
) -> Dict[str, Any]:
    result = copy.deepcopy(record)
    result.update(
        {
            "cache_key": cache_key,
            "curation_label": curated["label"],
            "curation_confidence": curated["confidence"],
            "curation_rationale": curated["rationale"],
            "curation_signals": curated["signals"],
            "curation_model": model,
            "curation_prompt_version": prompt_version,
            "curation_attempts": attempts,
            "curation_latency_seconds": round(latency_seconds, 4),
            "curation_timestamp": utc_now_iso(),
        }
    )
    return result


def summarize_results(
    results: Sequence[Dict[str, Any]],
    failed_count: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    label_counts = {label: 0 for label in ALLOWED_LABELS}
    confidence_totals = {label: 0.0 for label in ALLOWED_LABELS}
    tag_breakdown: Dict[str, Dict[str, int]] = {}

    for result in results:
        label = result.get("curation_label")
        if label in label_counts:
            label_counts[label] += 1
            confidence_totals[label] += float(result.get("curation_confidence", 0.0))

        tag = result.get("original_tag") or "unknown"
        tag_counts = tag_breakdown.setdefault(
            tag,
            {allowed_label: 0 for allowed_label in ALLOWED_LABELS},
        )
        if label in tag_counts:
            tag_counts[label] += 1

    label_confidence_means = {}
    for label, count in label_counts.items():
        label_confidence_means[label] = round(confidence_totals[label] / count, 4) if count else 0.0

    return {
        "created_at": utc_now_iso(),
        "dataset_name": args.dataset_name,
        "split": args.split,
        "model": args.model,
        "prompt_version": args.prompt_version,
        "streaming": args.streaming,
        "processed_count": len(results),
        "failed_count": failed_count,
        "label_counts": label_counts,
        "label_confidence_means": label_confidence_means,
        "tag_breakdown": tag_breakdown,
    }


def prepare_output_directory(output_dir: Path, overwrite: bool, resume: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "by_label").mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "curation_results.jsonl"
    if results_path.exists() and not overwrite and not resume:
        raise SystemExit(
            f"{results_path} already exists. Use --resume or --overwrite."
        )


def write_manifest(output_dir: Path, args: argparse.Namespace) -> None:
    manifest = {
        "created_at": utc_now_iso(),
        "dataset_name": args.dataset_name,
        "split": args.split,
        "model": args.model,
        "prompt_version": args.prompt_version,
        "streaming": args.streaming,
        "max_samples": args.max_samples,
        "start_index": args.start_index,
        "end_index": args.end_index,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"Missing API key. Set {args.api_key_env} before running the curator."
        )

    output_dir = Path(args.output_dir)
    prepare_output_directory(output_dir, overwrite=args.overwrite, resume=args.resume)
    write_manifest(output_dir, args)

    results_path = output_dir / "curation_results.jsonl"
    failures_path = output_dir / "failed_records.jsonl"
    summary_path = output_dir / "summary.json"
    subset_dir = output_dir / "by_label"

    existing_results = load_existing_results(results_path) if args.resume else {}
    all_results = list(existing_results.values())
    failed_count = count_jsonl_rows(failures_path) if args.resume else 0
    new_count = 0
    skipped_count = 0
    run_started_at = time.perf_counter()

    for record in iter_normalized_records(args):
        cache_key = make_cache_key(record, args.model, args.prompt_version)
        if cache_key in existing_results:
            skipped_count += 1
            continue

        try:
            curated, attempts, latency_seconds = curate_record_with_retries(
                api_key=api_key,
                args=args,
                record=record,
            )
        except CurationError as exc:
            failure_record = copy.deepcopy(record)
            failure_record.update(
                {
                    "cache_key": cache_key,
                    "curation_error": str(exc),
                    "curation_model": args.model,
                    "curation_prompt_version": args.prompt_version,
                    "curation_timestamp": utc_now_iso(),
                }
            )
            append_jsonl(failures_path, failure_record)
            failed_count += 1
            continue

        result_record = build_result_record(
            record=record,
            curated=curated,
            cache_key=cache_key,
            model=args.model,
            prompt_version=args.prompt_version,
            attempts=attempts,
            latency_seconds=latency_seconds,
        )
        append_jsonl(results_path, result_record)
        append_jsonl(subset_dir / f"{result_record['curation_label']}.jsonl", result_record)
        all_results.append(result_record)
        new_count += 1

        if new_count % args.batch_size == 0:
            elapsed = time.perf_counter() - run_started_at
            print(
                f"Processed {new_count} new examples "
                f"({skipped_count} skipped, {failed_count} failed) "
                f"in {elapsed:.1f}s.",
                flush=True,
            )

    summary = summarize_results(all_results, failed_count, args)
    summary["new_count"] = new_count
    summary["resume_skipped_count"] = skipped_count
    summary["runtime_seconds"] = round(time.perf_counter() - run_started_at, 2)

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"Finished curation: {new_count} new, {skipped_count} resumed, {failed_count} failed.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
