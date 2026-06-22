#!/usr/bin/env python3
"""Run raw image-understanding sanity probes for VQA evaluation outputs.

This diagnostic is intentionally separate from the scored evaluator. It reloads
the images referenced by an existing result JSON and asks the model to explain
or inspect them under several controls:

- correct image + correct answer explanation
- correct image + wrong answer challenge
- blank image + correct answer explanation
- shuffled image + correct answer explanation
- low-level overlay description without answering the multiple-choice question
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
BENCHMARK_DIR = REPO_ROOT / "benchmark"
for import_path in (str(REPO_ROOT), str(SRC_DIR), str(BENCHMARK_DIR)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)


PROBE_CHOICES = (
    "explain_correct",
    "explain_wrong",
    "explain_correct_blank",
    "explain_correct_shuffle",
    "overlay_describe",
)


def load_benchmark_backend() -> dict[str, Any]:
    """Import the vLLM evaluator lazily so --help works without eval deps."""
    from evaluation import (  # noqa: PLC0415
        VQADataset,
        ModelEvaluator,
        build_deranged_index_mapping,
        create_blank_image_like,
        ensure_pil_image,
    )

    class RawPromptModelRunner(ModelEvaluator):
        """Model runner that preserves diagnostic prompts exactly as written."""

        def _get_model_request_data(self, questions):
            modality = "image"
            try:
                return self.model_loader(questions, modality, self.model_id)
            except TypeError:
                return self.model_loader(questions, modality)

        def process_prompt_batch(self, batch_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
            request_data = self._get_model_request_data([item["prompt"] for item in batch_data])
            images = self.prepare_images([item["image"] for item in batch_data])
            inputs = [
                {
                    "prompt": request_data.prompts[i],
                    "multi_modal_data": {"image": images[i]},
                }
                for i in range(len(batch_data))
            ]

            start_time = time.time()
            outputs = self.llm.generate(inputs, sampling_params=self.sampling_params)
            elapsed = time.time() - start_time

            results = []
            for i, output in enumerate(outputs):
                result = dict(batch_data[i])
                result["response"] = output.outputs[0].text
                judgment, passed = classify_response(result["probe_type"], result["response"])
                result["heuristic_judgment"] = judgment
                result["heuristic_pass"] = passed
                result["response_time"] = elapsed / len(outputs)
                results.append(result)
            return results

    return {
        "VQADataset": VQADataset,
        "RawPromptModelRunner": RawPromptModelRunner,
        "build_deranged_index_mapping": build_deranged_index_mapping,
        "create_blank_image_like": create_blank_image_like,
        "ensure_pil_image": ensure_pil_image,
    }


def classify_response(probe_type: str, response: str) -> tuple[str, bool | None]:
    """Return a coarse support/rejection label for quick triage.

    This heuristic helps scan large diagnostic JSON files, but manual review is
    still required before making claims about visual grounding.
    """
    text = " ".join(response.lower().split())
    rejection_markers = (
        "does not support",
        "not supported",
        "not visibly support",
        "not visible",
        "cannot determine",
        "can't determine",
        "insufficient",
        "blank",
        "mismatched",
        "doesn't match",
        "contradicts",
        "incorrect",
        "wrong answer",
        "not enough visual",
        "no visible evidence",
    )
    support_markers = (
        "is correct",
        "answer is correct",
        "supports this answer",
        "visibly supports",
        "supported by",
        "correct because",
        "this answer is supported",
    )

    rejects = any(marker in text for marker in rejection_markers)
    supports = any(marker in text for marker in support_markers)

    if rejects:
        judgment = "rejects_or_insufficient"
    elif supports:
        judgment = "supports"
    else:
        judgment = "unclear"

    if probe_type == "explain_correct":
        return judgment, judgment == "supports"
    if probe_type in ("explain_wrong", "explain_correct_blank", "explain_correct_shuffle"):
        return judgment, judgment == "rejects_or_insufficient"
    return judgment, None


def expected_letter(value: Any) -> str:
    if isinstance(value, int):
        return chr(65 + value)
    text = str(value).strip().upper()
    if text.isdigit():
        return chr(65 + int(text))
    return text


def parse_choices(question: str) -> dict[str, str]:
    choices: dict[str, str] = {}
    for raw_line in question.splitlines():
        line = raw_line.strip()
        if len(line) >= 3 and line[0] in "ABCDE" and line[1] == ".":
            choices[line[0]] = line[2:].strip()
    return choices


def infer_question_type(question: str) -> str:
    normalized = question.lower()
    if "which colored arrow" in normalized:
        return "direction_arrow"
    if "same 3d location" in normalized:
        return "cross_view_correspondence"
    if "closest to the camera" in normalized:
        return "depth_closest"
    if "farthest from the camera" in normalized:
        return "depth_farthest"
    return "generic"


def first_question_line(question: str) -> str:
    return question.splitlines()[0].strip()


def answer_text(letter: str, choices: dict[str, str]) -> str:
    return choices.get(letter, "")


def select_wrong_answer(
    expected: str,
    choices: dict[str, str],
    *,
    strategy: str,
    rng: random.Random,
) -> str:
    wrong_letters = [letter for letter in sorted(choices) if letter != expected]
    if not wrong_letters:
        raise ValueError("Cannot create wrong-answer probe with fewer than two choices.")
    if strategy == "first":
        return wrong_letters[0]
    if strategy == "next":
        ordered = sorted(choices)
        expected_index = ordered.index(expected)
        for offset in range(1, len(ordered)):
            candidate = ordered[(expected_index + offset) % len(ordered)]
            if candidate != expected:
                return candidate
    if strategy == "random":
        return rng.choice(wrong_letters)
    raise ValueError(f"Unsupported wrong-answer strategy: {strategy}")


def build_explanation_prompt(
    *,
    question: str,
    given_letter: str,
    given_text: str,
    probe_type: str,
) -> str:
    if probe_type == "explain_correct":
        answer_intro = "The known correct answer is"
        expectation = (
            "Explain why this answer is correct using only visible evidence from the image. "
            "If the image does not visibly support the answer, say that explicitly."
        )
    elif probe_type == "explain_wrong":
        answer_intro = "The proposed answer is"
        expectation = (
            "Assess whether this proposed answer is supported by visible evidence. "
            "If it is correct, explain why. If it is not supported, say so explicitly "
            "and identify the visual conflict."
        )
    elif probe_type in ("explain_correct_blank", "explain_correct_shuffle"):
        answer_intro = "The known correct answer for the original question is"
        expectation = (
            "Explain whether the current image visibly supports that answer. "
            "If the current image is blank, mismatched, or insufficient, say so explicitly."
        )
    else:
        raise ValueError(f"Unsupported explanation probe: {probe_type}")

    return (
        "You are inspecting a robotics VQA image.\n\n"
        f"{question}\n\n"
        f"{answer_intro} {given_letter}: {given_text}.\n\n"
        f"{expectation}\n"
        "Do not choose a new answer unless the image contradicts the proposed answer."
    )


def build_overlay_prompt(question: str) -> str:
    question_type = infer_question_type(question)
    base = (
        "You are inspecting a robotics VQA image with visual overlays.\n\n"
        f"Original question: {first_question_line(question)}\n\n"
    )
    if question_type == "direction_arrow":
        return (
            base
            + "List each visible colored arrow and describe the direction it points in image coordinates. "
            "Also mention the robot gripper/current position if visible. "
            "Do not answer the multiple-choice question."
        )
    if question_type in ("depth_closest", "depth_farthest"):
        return (
            base
            + "List each visible colored point and describe its approximate 2D image position "
            "(for example top-left, center, lower-right). Mention any visible depth cues, "
            "but do not decide which point is closest or farthest."
        )
    if question_type == "cross_view_correspondence":
        return (
            base
            + "Describe the left and right image panels. Locate the red dot in the left panel "
            "and list the labeled candidate points in the right panel with their approximate "
            "positions. Do not choose the corresponding point."
        )
    return (
        base
        + "Describe the visible evidence relevant to the question. "
        "Do not answer the multiple-choice question."
    )


def load_result_rows(
    result_json: Path,
    *,
    max_examples: int | None,
    question_ids: set[str] | None,
    source_indices: set[int] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with result_json.open("r", encoding="utf-8") as handle:
        result_data = json.load(handle)

    rows = list(result_data.get("responses", []))
    if question_ids is not None:
        rows = [row for row in rows if row.get("question_id") in question_ids]
    if source_indices is not None:
        rows = [row for row in rows if row.get("source_index") in source_indices]
    if max_examples is not None:
        rows = rows[:max_examples]
    if not rows:
        raise ValueError("No rows selected from result JSON.")
    for row in rows:
        if row.get("source_index") is None:
            raise ValueError(f"Selected row lacks source_index: {row.get('question_id')}")
    return result_data, rows


def build_probe_items(
    rows: list[dict[str, Any]],
    dataset: Any,
    probes: list[str],
    *,
    wrong_answer_strategy: str,
    sanity_seed: int,
    image_output_dir: Path | None,
    backend: dict[str, Any],
) -> list[dict[str, Any]]:
    rng = random.Random(sanity_seed)
    shuffle_mapping = (
        backend["build_deranged_index_mapping"](len(rows), sanity_seed) if len(rows) > 1 else None
    )
    items: list[dict[str, Any]] = []

    for eval_index, row in enumerate(rows):
        dataset_item = dataset[eval_index]
        expected = expected_letter(row["expected"])
        choices = parse_choices(row["question"])
        if expected not in choices:
            raise ValueError(f"Expected answer {expected!r} not found in choices for {row['question_id']}")

        wrong_letter = select_wrong_answer(
            expected,
            choices,
            strategy=wrong_answer_strategy,
            rng=rng,
        )

        base_context = {
            "question_id": row["question_id"],
            "source_index": row["source_index"],
            "question": row["question"],
            "question_type": infer_question_type(row["question"]),
            "expected_letter": expected,
            "expected_answer_text": answer_text(expected, choices),
            "choices": choices,
        }

        for probe_type in probes:
            image = dataset_item["image"]
            image_control = "original"
            given_letter = expected
            expected_behavior = "explain_supported_correct_answer"

            if probe_type == "explain_wrong":
                given_letter = wrong_letter
                expected_behavior = "reject_or_flag_unsupported_answer"
            elif probe_type == "explain_correct_blank":
                image = backend["create_blank_image_like"](image)
                image_control = "blank"
                expected_behavior = "state_image_is_insufficient"
            elif probe_type == "explain_correct_shuffle":
                if shuffle_mapping is None:
                    raise ValueError("Shuffled-image probe requires at least two selected examples.")
                donor_eval_index = shuffle_mapping[eval_index]
                image = dataset[donor_eval_index]["image"]
                image_control = "shuffle"
                expected_behavior = "state_image_is_mismatched_or_insufficient"
            elif probe_type == "overlay_describe":
                expected_behavior = "describe_visible_overlay_without_answering"

            if probe_type == "overlay_describe":
                prompt = build_overlay_prompt(row["question"])
                given_answer_text = None
                given_answer_letter = None
            else:
                given_answer_text = answer_text(given_letter, choices)
                given_answer_letter = given_letter
                prompt = build_explanation_prompt(
                    question=row["question"],
                    given_letter=given_letter,
                    given_text=given_answer_text,
                    probe_type=probe_type,
                )

            probe_id = f"{row['question_id']}::{probe_type}"
            image_path = None
            if image_output_dir is not None:
                image_output_dir.mkdir(parents=True, exist_ok=True)
                image_path = image_output_dir / f"{probe_id.replace('/', '_').replace(':', '_')}.png"
                backend["ensure_pil_image"](image).save(image_path)

            items.append(
                {
                    **base_context,
                    "probe_id": probe_id,
                    "probe_type": probe_type,
                    "prompt": prompt,
                    "image": image,
                    "image_control": image_control,
                    "image_source_index": (
                        dataset[shuffle_mapping[eval_index]]["source_index"]
                        if probe_type == "explain_correct_shuffle" and shuffle_mapping is not None
                        else row["source_index"] if image_control == "original" else None
                    ),
                    "given_answer_letter": given_answer_letter,
                    "given_answer_text": given_answer_text,
                    "wrong_answer_letter": wrong_letter if probe_type == "explain_wrong" else None,
                    "wrong_answer_text": answer_text(wrong_letter, choices)
                    if probe_type == "explain_wrong"
                    else None,
                    "expected_behavior": expected_behavior,
                    "saved_image_path": str(image_path) if image_path else None,
                }
            )
    return items


def drop_image_payload(item: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(item)
    cleaned.pop("image", None)
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run image-understanding sanity probes on examples from an evaluation result JSON."
    )
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--model", required=True, help="Hugging Face model id to run.")
    parser.add_argument("--dataset", default=None, help="Dataset name. Defaults to the result JSON dataset.")
    parser.add_argument("--split", default=None, help="Dataset split. Defaults to the result JSON split.")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Maximum result rows to probe before expanding into probe variants.",
    )
    parser.add_argument("--question-id", action="append", dest="question_ids")
    parser.add_argument("--source-index", action="append", type=int, dest="source_indices")
    parser.add_argument(
        "--probe",
        action="append",
        choices=PROBE_CHOICES,
        help="Probe to run. Repeat to select multiple. Defaults to all probes.",
    )
    parser.add_argument(
        "--wrong-answer-strategy",
        choices=("next", "first", "random"),
        default="next",
    )
    parser.add_argument("--sanity-seed", type=int, default=11)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/diagnostics/image_understanding"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-probe-images", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_data, rows = load_result_rows(
        args.result_json,
        max_examples=args.max_examples,
        question_ids=set(args.question_ids) if args.question_ids else None,
        source_indices=set(args.source_indices) if args.source_indices else None,
    )

    backend = load_benchmark_backend()
    dataset_name = args.dataset or result_data["dataset"]
    split = args.split or result_data["split"]
    source_indices = [int(row["source_index"]) for row in rows]
    dataset = backend["VQADataset"](
        dataset_name,
        split=split,
        sample_indices=source_indices,
        subset_name=f"diagnostic_from_{result_data.get('subset', 'unknown')}",
    )

    probes = args.probe or list(PROBE_CHOICES)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{Path(args.result_json).stem}_{args.model.split('/')[-1]}_{timestamp}"
    run_dir = args.output_dir / run_name
    image_dir = run_dir / "images" if args.save_probe_images else None

    probe_items = build_probe_items(
        rows,
        dataset,
        probes,
        wrong_answer_strategy=args.wrong_answer_strategy,
        sanity_seed=args.sanity_seed,
        image_output_dir=image_dir,
        backend=backend,
    )

    runner = backend["RawPromptModelRunner"](
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        prompt_mode="zero_shot",
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    responses: list[dict[str, Any]] = []
    for start in range(0, len(probe_items), args.batch_size):
        batch = probe_items[start : start + args.batch_size]
        batch_results = runner.process_prompt_batch(batch)
        responses.extend(drop_image_payload(item) for item in batch_results)
        print(f"Processed {len(responses)}/{len(probe_items)} probes", flush=True)

    output = {
        "model": args.model,
        "dataset": dataset_name,
        "split": split,
        "source_result_json": str(args.result_json),
        "selected_examples": len(rows),
        "total_probes": len(responses),
        "probes": probes,
        "wrong_answer_strategy": args.wrong_answer_strategy,
        "sanity_seed": args.sanity_seed,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "created_at": timestamp,
        "responses": responses,
    }
    output_path = run_dir / "image_understanding_sanity.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Saved diagnostics to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
