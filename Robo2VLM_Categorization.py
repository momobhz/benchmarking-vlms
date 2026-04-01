#!/usr/bin/env python3
"""Categorize Robo2VLM-1 questions with a local Hugging Face model.

This script rewrites the workflow from `edit_Robo2VLM_Categorization_Colab.ipynb`
into a standalone Python entry point while keeping the original notebook intact.

Install dependencies before running:
    pip install datasets transformers torch accelerate huggingface_hub pillow
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
from pathlib import Path

import torch
from datasets import Features, Image, Value, load_dataset
from huggingface_hub import login
from PIL import Image as PILImage
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = """You are an AI judge tasked with categorizing questions from a robotics VQA dataset into one of three categories: 'spatial', 'affordance', or 'neither'.

Definitions:
- Spatial reasoning: Questions that involve understanding relative positions, directions, distances, orientations, alignments, or geometric relationships between objects, the robot, or the environment. Examples: "What's the relative direction between object X and the gripper?", "How far is the object from the robot?", "Is the object to the left of the gripper?"

- Affordance understanding: Questions that involve understanding what actions or manipulations are possible with objects, such as reachability, graspability, movability, or interaction capabilities. Examples: "Is object X reachable?", "Can the robot grasp this object?", "Is this object movable?"

- Neither: Questions that do not fit into spatial reasoning or affordance understanding, such as general object recognition, counting, or other unrelated topics.

Classify the following question into exactly one category: spatial, affordance, or neither. Respond with only the category name.

Question: {question}

Category:"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Categorize Robo2VLM-1 questions into spatial, affordance, or neither."
    )
    parser.add_argument("--model-name", default="microsoft/phi-2", help="Hugging Face model to use.")
    parser.add_argument("--dataset-name", default="keplerccc/Robo2VLM-1", help="Dataset to stream.")
    parser.add_argument("--split", default="test", help="Dataset split to process.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum number of samples to process. Use -1 to process the full split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for the spatial/affordance index JSON files.",
    )
    parser.add_argument(
        "--hf-login",
        action="store_true",
        help="Prompt for a Hugging Face token before loading the model.",
    )
    parser.add_argument(
        "--preview-first-sample",
        action="store_true",
        help="Print metadata for the first sample after categorization.",
    )
    parser.add_argument(
        "--preview-index",
        type=int,
        default=None,
        help="Print metadata for a specific streamed sample index after categorization.",
    )
    parser.add_argument(
        "--open-images",
        action="store_true",
        help="Open preview images with the default system image viewer.",
    )
    return parser.parse_args()


def check_gpu() -> bool:
    gpu_available = torch.cuda.is_available()
    print("GPU available:", gpu_available)
    if gpu_available:
        print("GPU name:", torch.cuda.get_device_name(0))
    else:
        print("No GPU detected. The model will run on CPU, which may be slow.")
    return gpu_available


def maybe_login(enable_login: bool) -> None:
    if enable_login:
        login()


def load_tokenizer_and_model(model_name: str, use_gpu: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(model_name)
    config.pad_token_id = tokenizer.pad_token_id

    dtype = torch.float16 if use_gpu else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype,
        device_map="auto" if use_gpu else None,
    )
    if not use_gpu:
        model = model.to("cpu")

    print("Model loaded on device:", model.device)
    return tokenizer, model


def load_streaming_dataset(dataset_name: str, split: str):
    features = Features(
        {
            "id": Value("string"),
            "question": Value("string"),
            "choices": Value("string"),
            "correct_answer": Value("int64"),
            "image": Image(decode=False),
        }
    )
    return load_dataset(dataset_name, streaming=True, features=features, split=split)


def classify_question(question: str, tokenizer, model) -> str:
    prompt = PROMPT_TEMPLATE.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            temperature=0.0,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_part = response.split("Category:")[-1].strip().lower()
    if "spatial" in answer_part:
        return "spatial"
    if "affordance" in answer_part:
        return "affordance"
    return "neither"


def categorize_dataset(dataset, tokenizer, model, max_samples: int):
    spatial_indices: list[int] = []
    affordance_indices: list[int] = []

    for index, sample in enumerate(dataset):
        category = classify_question(sample["question"], tokenizer, model)
        if category == "spatial":
            spatial_indices.append(index)
        elif category == "affordance":
            affordance_indices.append(index)

        processed = index + 1
        if processed % 100 == 0:
            print(f"Processed {processed} samples")
        if max_samples >= 0 and processed >= max_samples:
            break

    return spatial_indices, affordance_indices


def save_indices(output_dir: Path, split: str, spatial_indices: list[int], affordance_indices: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_path = output_dir / f"{split}_spatial_indices.json"
    affordance_path = output_dir / f"{split}_affordance_indices.json"

    with spatial_path.open("w", encoding="utf-8") as handle:
        json.dump(spatial_indices, handle)
    with affordance_path.open("w", encoding="utf-8") as handle:
        json.dump(affordance_indices, handle)

    print(f"Finished: spatial {len(spatial_indices)}, affordance {len(affordance_indices)}")
    print("Sample spatial indices:", spatial_indices[:10])
    print("Sample affordance indices:", affordance_indices[:10])
    print("Saved:", spatial_path)
    print("Saved:", affordance_path)


def decode_image(sample) -> PILImage.Image:
    image_bytes = sample["image"]["bytes"]
    return PILImage.open(io.BytesIO(image_bytes))


def print_sample(sample, index: int | None = None) -> None:
    if index is not None:
        print(f"Index: {index}")
    print(f"ID: {sample['id']}")
    print(f"Question: {sample['question']}")
    print(f"Choices: {sample['choices']}")
    print(f"Correct Answer: {sample['correct_answer']}")


def preview_first_sample(dataset, open_images: bool) -> None:
    sample = next(iter(dataset))
    print_sample(sample)
    image = decode_image(sample)
    print(f"Image size: {image.size}")
    if open_images:
        image.show()


def preview_sample_at_index(dataset, target_index: int, open_images: bool) -> None:
    sample = next(itertools.islice(iter(dataset), target_index, None))
    print_sample(sample, index=target_index)
    image = decode_image(sample)
    print(f"Image size: {image.size}")
    if open_images:
        image.show()


def main() -> None:
    args = parse_args()
    use_gpu = check_gpu()
    maybe_login(args.hf_login)

    tokenizer, model = load_tokenizer_and_model(args.model_name, use_gpu)
    dataset = load_streaming_dataset(args.dataset_name, args.split)
    spatial_indices, affordance_indices = categorize_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        model=model,
        max_samples=args.max_samples,
    )
    save_indices(args.output_dir, args.split, spatial_indices, affordance_indices)

    if args.preview_first_sample:
        preview_first_sample(load_streaming_dataset(args.dataset_name, args.split), args.open_images)
    if args.preview_index is not None:
        preview_sample_at_index(
            load_streaming_dataset(args.dataset_name, args.split),
            args.preview_index,
            args.open_images,
        )


if __name__ == "__main__":
    main()
