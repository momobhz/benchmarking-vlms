import os
import ast
import json
import base64
import argparse
import re
import time
import random
import subprocess
import sys
import numpy as np
from datetime import datetime
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import concurrent.futures
from typing import List, Dict, Tuple, Any
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
for import_path in (REPO_ROOT, SRC_DIR):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

# Import from vision_language.py and vision_language_multi_image.py
from vision_language import model_example_map as single_image_models
# from vision_language_multi_image import model_example_map as multi_image_models
# from vision_language_multi_image_copy import model_example_map as multi_image_models

from vllm import SamplingParams


def check_gpu_usage():
    """Check if GPUs are actively in use by other processes."""
    try:
        # Get GPU memory usage information using nvidia-smi
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'])
        memory_info = result.decode('utf-8').strip().split('\n')
        
        # Calculate utilization percentage for each GPU
        gpu_utilization = []
        for info in memory_info:
            used, total = map(int, info.split(','))
            utilization = used / total
            gpu_utilization.append(utilization)
        print(gpu_utilization)
        
        # Consider GPU in use if utilization is above 20%
        # This threshold can be adjusted based on your requirements
        return any(util > 0.20 for util in gpu_utilization)
    except Exception as e:
        print(f"Error checking GPU usage: {e}")
        # If we can't check, assume GPUs are available
        return False


def wait_for_gpu_availability(check_interval=1, max_wait_time=3600):
    """Wait until GPUs are available for use."""
    start_time = time.time()
    waited = False
    
    while check_gpu_usage():
        if not waited:
            print("GPUs are currently in use by other processes. Waiting for availability...")
            waited = True
        
        time.sleep(check_interval)
    
    if waited:
        print("GPUs are now available. Starting evaluation...")


demo_prompt = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B
"""

def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


# vllm imports
from vllm import LLM, SamplingParams

# Model-specific imports for handling images
from transformers import AutoProcessor, AutoTokenizer, pipeline

from vlm_bench.curation.taxonomy import load_subset_source_indices
from vlm_bench.eval.prompts import PROMPT_MODE_CHOICES, apply_prompt_mode

# Constants
DEFAULT_MODELS = [ 
                #   "llava-hf/llava-1.5-7b-hf", 
                #   "llava-hf/llava-v1.6-mistral-7b-hf", 
                #   "Qwen/Qwen2.5-VL-7B-Instruct",   
                #    "google/paligemma-3b-mix-224",
                #    "google/paligemma2-3b-ft-docci-448",
                #    "meta-llama/Llama-3.2-11B-Vision-Instruct",
                #   "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                #   "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
                  "llava-hf/llava-v1.6-34b-hf", 
                  "llava-hf/llava-next-72b-hf", 
                #   "Qwen/Qwen2.5-VL-32B-Instruct",
                #   "Qwen/Qwen2.5-VL-72B-Instruct",
                #   "microsoft/Phi-4-multimodal-instruct",
                  ]
DEFAULT_MAX_BATCH_SIZE = 96
DEFAULT_TENSOR_PARALLEL_SIZE = 4  # Use all 8 A100 GPUs
DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_IMAGE_MODE_INSTRUCTION = (
    "The provided image has two side-by-side panels: the left panel is the "
    "original RGB image, and the right panel is an estimated depth map "
    "generated from that RGB image. Use both panels when helpful."
)

# Initialize the answer extraction model globally for reuse
ANSWER_EXTRACTOR = None

def initialize_answer_extractor(tensor_parallel_size=1):
    """Initialize the Llama 3.2 model for answer extraction."""
    global ANSWER_EXTRACTOR
    if ANSWER_EXTRACTOR is None:
        print("Initializing Llama 3.2 model for answer extraction...")
        ANSWER_EXTRACTOR = LLM(
            model="meta-llama/Llama-3.2-3B-Instruct",  # Using smaller model for extraction
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=1024,  # Smaller context as we only need to extract answers
            gpu_memory_utilization=0.75,
            enforce_eager=True,
        )
    return ANSWER_EXTRACTOR


class VQADataset(Dataset):
    """Load and prepare VQA dataset for evaluation."""
    def __init__(self, dataset_name, split="test", max_samples=None, sample_indices=None, subset_name="full"):
        self.dataset_name = dataset_name
        self.split = split
        self.subset_name = subset_name
        self.sample_indices = sample_indices

        if sample_indices is not None:
            requested_indices = list(sample_indices)
            if max_samples is not None:
                requested_indices = requested_indices[:max_samples]

            self.dataset = self._load_index_subset(requested_indices)
            self.indices = requested_indices
            self.is_streaming = True
        elif max_samples is None:
            self.dataset = load_dataset(dataset_name, split=split)
            self.indices = list(range(len(self.dataset)))
            self.is_streaming = False
        else:
            stream = load_dataset(dataset_name, split=split, streaming=True)
            self.dataset = list(stream.take(max_samples))
            self.indices = list(range(len(self.dataset)))
            self.is_streaming = True

        self.max_samples = len(self.dataset)

    def _load_index_subset(self, requested_indices):
        """Load only the examples referenced by source indices from a split."""
        if not requested_indices:
            return []

        stream = load_dataset(self.dataset_name, split=self.split, streaming=True)
        requested_set = set(requested_indices)
        highest_index = max(requested_set)
        loaded_examples = {}

        for dataset_index, item in enumerate(stream):
            if dataset_index in requested_set:
                loaded_examples[dataset_index] = item
                if len(loaded_examples) == len(requested_set):
                    break
            if dataset_index >= highest_index:
                break

        missing_indices = [index for index in requested_indices if index not in loaded_examples]
        if missing_indices:
            raise ValueError(
                f"Failed to load {len(missing_indices)} requested indices from "
                f"{self.dataset_name} ({self.split} split). Missing head: {missing_indices[:10]}"
            )

        return [loaded_examples[index] for index in requested_indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.sample_indices is not None:
            item = self.dataset[idx]
            source_index = self.indices[idx]
        else:
            item = self.dataset[self.indices[idx]]
            source_index = self.indices[idx]

        return {
            "id": item["id"],
            "question": item["question"],
            "choices": item["choices"],
            "correct_answer": item["correct_answer"],
            "image": item["image"],
            "tag": item.get("tag", "unknown"),
            "source_index": source_index,
        }

    def format_multiple_choice_question(self, item):
        """Format question with multiple choice options."""
        question_text = item["question"]
        choices = item["choices"]

        if isinstance(choices, str):
            try:
                choices = ast.literal_eval(choices)
            except Exception:
                choices = [choices]

        formatted_choices = ""
        choice_letter_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

        for i, choice in enumerate(choices):
            if i < len(choice_letter_map):
                formatted_choices += f"\n{choice_letter_map[i]}. {choice}"

        return f"{question_text}\nChoices:{formatted_choices}"


def ensure_pil_image(image: Any) -> Image.Image:
    """Convert a dataset image payload into a PIL image."""
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(image).convert("RGB")


def create_blank_image_like(image: Any) -> Image.Image:
    """Create a blank image that preserves the source image dimensions."""
    source_image = ensure_pil_image(image)
    return Image.new("RGB", source_image.size, color=(0, 0, 0))


class DepthImageAugmenter:
    """Create RGB + estimated-depth composite images without writing to disk."""

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.pipe = None

    def _resolve_pipeline_device(self):
        if self.device == "auto":
            return 0 if torch.cuda.is_available() else -1
        if self.device in ("cpu", "-1"):
            return -1
        if self.device == "cuda":
            return 0
        if self.device.startswith("cuda:"):
            return int(self.device.split(":", 1)[1])

        try:
            return int(self.device)
        except ValueError:
            return self.device

    def _ensure_pipe(self):
        if self.pipe is None:
            pipeline_device = self._resolve_pipeline_device()
            print(
                f"Initializing depth-estimation pipeline {self.model_name} "
                f"on device {self.device}..."
            )
            self.pipe = pipeline(
                "depth-estimation",
                model=self.model_name,
                device=pipeline_device,
            )
        return self.pipe

    def augment_image(self, image: Any) -> Image.Image:
        original_image = ensure_pil_image(image)
        result = self._ensure_pipe()(original_image)
        depth_image = result["depth"].convert("RGB")

        if depth_image.size != original_image.size:
            try:
                resize_filter = Image.Resampling.BILINEAR
            except AttributeError:
                resize_filter = Image.BILINEAR
            depth_image = depth_image.resize(original_image.size, resize_filter)

        total_width = original_image.width + depth_image.width
        max_height = max(original_image.height, depth_image.height)
        composite_image = Image.new("RGB", (total_width, max_height))
        composite_image.paste(original_image, (0, 0))
        composite_image.paste(depth_image, (original_image.width, 0))
        return composite_image


def format_question_for_image_mode(question: str, image_mode: str) -> str:
    """Add image-layout context when using non-standard visual inputs."""
    if image_mode == "rgb":
        return question
    if image_mode == "rgb_depth":
        return f"{DEPTH_IMAGE_MODE_INSTRUCTION}\n\n{question}"
    raise ValueError(f"Unsupported image mode: {image_mode}")


def apply_image_mode_to_batch(
    batch_items: List[Dict[str, Any]],
    image_mode: str,
    depth_augmenter: Any = None,
) -> List[Dict[str, Any]]:
    """Apply the requested in-memory image transform to evaluation items."""
    transformed_items = [dict(item) for item in batch_items]

    if image_mode == "rgb":
        for item in transformed_items:
            item["image_mode"] = image_mode
        return transformed_items

    if image_mode == "rgb_depth":
        if depth_augmenter is None:
            raise ValueError("rgb_depth image mode requires a depth augmenter.")

        for item in transformed_items:
            item["image"] = depth_augmenter.augment_image(item["image"])
            item["formatted_question"] = format_question_for_image_mode(
                item["formatted_question"],
                image_mode,
            )
            item["image_mode"] = image_mode
            item["depth_model"] = depth_augmenter.model_name
        return transformed_items

    raise ValueError(f"Unsupported image mode: {image_mode}")


def build_deranged_index_mapping(num_items: int, seed: int) -> List[int]:
    """Create a deterministic permutation with no fixed points."""
    if num_items < 2:
        raise ValueError("Shuffled-image sanity checks require at least 2 examples.")

    indices = list(range(num_items))
    shuffled = indices[:]
    rng = random.Random(seed)

    for _ in range(20):
        rng.shuffle(shuffled)
        if all(i != shuffled[i] for i in indices):
            return shuffled

    shift = rng.randrange(1, num_items)
    return indices[shift:] + indices[:shift]


def build_sanity_check_context(dataset: Dataset, sanity_check: str, sanity_seed: int) -> Dict[str, Any]:
    """Prepare any state needed to run a visual sanity check."""
    context: Dict[str, Any] = {
        "mode": sanity_check,
        "seed": sanity_seed,
        "image_mapping": None,
        "repeat_image": None,
        "repeat_source_index": None,
    }

    if sanity_check == "shuffle":
        context["image_mapping"] = build_deranged_index_mapping(len(dataset), sanity_seed)
    elif sanity_check == "repeat_first":
        first_item = dataset[0]
        context["repeat_image"] = first_item["image"]
        context["repeat_source_index"] = first_item.get("source_index")

    return context


def apply_sanity_check_to_batch(
    batch_items: List[Dict[str, Any]],
    dataset: Dataset,
    sanity_context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return batch items with any requested image ablation applied."""
    mode = sanity_context["mode"]
    transformed_items = [dict(item) for item in batch_items]

    if mode == "none":
        for item in transformed_items:
            item["image_source_index"] = item.get("source_index")
            item["sanity_check"] = mode
        return transformed_items

    if mode == "blank":
        for item in transformed_items:
            item["image"] = create_blank_image_like(item["image"])
            item["image_source_index"] = None
            item["sanity_check"] = mode
        return transformed_items

    if mode == "shuffle":
        image_mapping = sanity_context["image_mapping"]
        for item in transformed_items:
            donor_eval_index = image_mapping[item["eval_index"]]
            donor_item = dataset[donor_eval_index]
            item["image"] = donor_item["image"]
            item["image_source_index"] = donor_item.get("source_index")
            item["sanity_check"] = mode
        return transformed_items

    if mode == "repeat_first":
        for item in transformed_items:
            item["image"] = sanity_context["repeat_image"]
            item["image_source_index"] = sanity_context["repeat_source_index"]
            item["sanity_check"] = mode
        return transformed_items

    raise ValueError(f"Unsupported sanity check mode: {mode}")


class ModelEvaluator:
    """Evaluate VLM models on VQA tasks."""
    def __init__(
        self,
        model_id: str,
        device="cuda",
        tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE,
        prompt_mode="cot",
        temperature=0.0,
        max_tokens=10240,
    ):
        self.model_id = model_id
        self.device = device
        self.model_type = self._get_model_key()
        self.prompt_mode = prompt_mode
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        print(f"Initializing model {model_id} of type {self.model_type} with tensor parallelism {tensor_parallel_size}...")
        
        # Use the model loading functionality from vision_language.py or vision_language_multi_image.py
        # Determine which model map to use (single image or multi-image)
        model_key = self._get_model_key()
        
        # Get the appropriate model loading function
        if model_key in single_image_models:
            self.model_loader = single_image_models[model_key]
            self.is_multi_image = False
        # elif model_key in multi_image_models:
        #     self.model_loader = multi_image_models[model_key]
            self.is_multi_image = True
        else:
            raise ValueError(f"Model {model_id} not found in either single or multi-image model maps")
        
        # Initialize model with the appropriate loader
        dummy_question = "What's in this image?"
        model_request_data = self._get_model_request_data([dummy_question])
        
        # Extract engine args and initialize vLLM
        engine_args_dict = model_request_data.engine_args.__dict__.copy()
        # The family-specific loader provides prompt formatting defaults, but
        # evaluation should always instantiate the exact checkpoint requested.
        engine_args_dict["model"] = self.model_id
        engine_args_dict["tensor_parallel_size"] = tensor_parallel_size
        engine_args_dict["max_model_len"] = 4096
        engine_args_dict["gpu_memory_utilization"] = 0.9
        engine_args_dict["enforce_eager"] = False
        
        self.llm = LLM(**engine_args_dict)
        
        # Initialize processor with proper settings
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
    
    # def _get_model_key(self, model_id: str) -> str:
    #     """Determine model type based on model ID."""
    #     model_id_lower = model_id.lower()
    #     if "llava" in model_id_lower:
    #         return "llava"
    #     elif "qwen" in model_id_lower:
    #         return "qwen"
    #     elif "llama" in model_id_lower:
    #         return "llama"
    #     elif "gemma" in model_id_lower:
    #         return "gemma"
    #     else:
    #         return "unknown"
    
    def _get_model_key(self) -> str:
        """Get the appropriate key for the model maps."""
        model_id_lower = self.model_id.lower()

        if "llava-1.5" in model_id_lower:
            return "llava"
        elif "llava-v1.6" in model_id_lower:
            return "llava-next"
        elif "llava-next" in model_id_lower:
            return "llava-next"
        elif "qwen2.5" in model_id_lower:
            return "qwen2_5_vl"
        elif "qwen2" in model_id_lower:
            return "qwen2_vl"
        elif "qwen" in model_id_lower:
            return "qwen_vl"
        elif "llama-4" in model_id_lower:
            return "llama4"
        elif "llama-3.2" in model_id_lower:
            return "mllama"
        elif "gemma-3" in model_id_lower:
            return "gemma3"
        elif "paligemma2" in model_id_lower:
            return "paligemma2"        
        elif "paligemma" in model_id_lower:
            return "paligemma"      
        elif "phi-4" in model_id_lower:
            return "phi4_mm"
        elif "deepseek-vl2" in model_id_lower or "deepseek_vl2" in model_id_lower:
            return "deepseek_vl_v2"
        # # Map the model ID to the correct key in the model maps
        # if "llava-1.5" in model_id_lower:
        #     return "llava_1_5_7b"
        # elif "llava-v1.6-mistral-7b" in model_id_lower:
        #     return "llava_v1_6_mistral_7b"
        # elif "llava-next-72b" in model_id_lower:
        #     return "llava_next_72b"
        # elif "llava-v1.6-34b" in model_id_lower:
        #     return "llava_v1_6_34b"
        # elif "qwen2.5-vl-32b" in model_id_lower:
        #     return "qwen2_5_vl_32b"
        # elif "qwen2.5-vl-72b" in model_id_lower:
        #     return "qwen2_5_vl_72b"
        # elif "qwen2.5-vl-7b" in model_id_lower:
        #     return "qwen2_5_vl"
        # elif "llama-4" in model_id_lower:
        #     return "llama4"
        # elif "llama-4-maverick-17b" in model_id_lower:
        #     return "llama4_maverick"
        # elif "gemma-3" in model_id_lower:
        #     return "gemma3"
        # elif "paligemma2" in model_id_lower:
        #     return "paligemma2"
        else:
            # Default to a similar model if exact match not found
            for key in list(single_image_models.keys()):
                if key.lower() in model_id_lower:
                    return key
            raise 
    
    def _get_model_request_data(self, questions):
        """Get model request data from the appropriate loader."""
        modality = "image"  # We're working with images
        
        instructed_questions = apply_prompt_mode(questions, self.prompt_mode)
        try:
            return self.model_loader(instructed_questions, modality, self.model_id)
        except TypeError:
            return self.model_loader(instructed_questions, modality)
    
    def _get_model_prompt(self, question: str) -> str:
        """Create prompt based on model's loader function."""
        model_request_data = self._get_model_request_data([question])
        return model_request_data.prompts[0]
    
    def prepare_images(self, batch_images):
        """Prepare images for the model with proper preprocessing."""
        processed_images = []
        
        for image in batch_images:
            if isinstance(image, str):  # Path to image
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                # Convert to PIL Image if it's a tensor or array
                image = Image.fromarray(image).convert('RGB')
            
            processed_images.append(image)
            
        return processed_images
            
    def process_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of VQA examples."""
        # For debugging
        print(f"Processing batch of {len(batch_data)} examples...")
        
        prompts = []
        images = []
        
        # Prepare inputs for each item in batch
        for item in batch_data:
            # Format question
            question = item["formatted_question"]
            prompt = self._get_model_prompt(question)
            prompts.append(prompt)
            
            # Process image
            images.append(item["image"])
        
        # Prepare images
        processed_images = self.prepare_images(images)
        
        # Get model request data for batch processing
        start_time = time.time()
        
        # Process batch using model-specific approach
        inputs = []
        for i, prompt in enumerate(prompts):
            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {
                    "image": processed_images[i]
                }
            })
        
        # Run inference with vllm
        outputs = self.llm.generate(
            inputs,
            sampling_params=self.sampling_params,
        )
        batch_elapsed = time.time() - start_time
        
        results = []

        # Prepare data for batch answer extraction
        batch_questions = [item["formatted_question"] for item in batch_data]
        batch_responses = [output.outputs[0].text for output in outputs]
        
        # Extract letter answers in batch
        letter_answers = extract_letter_answer(batch_questions, batch_responses)

        for i, output in enumerate(outputs):
            response_text = output.outputs[0].text
            
            expected_answer = batch_data[i]["correct_answer"]
            if isinstance(expected_answer, int):
                expected_letter = chr(65 + expected_answer)
            else:
                expected_letter = str(expected_answer).upper()          
            
            results.append({
                "question_id": batch_data[i]["id"],
                "question": batch_data[i]["formatted_question"],
                "expected": expected_answer,
                "model": self.model_id,
                "predicted": response_text,
                "predicted_letter": letter_answers[i],
                "correct": letter_answers[i] == expected_letter if letter_answers[i] else False,
                "tag": batch_data[i].get("tag", "unknown"),
                "source_index": batch_data[i].get("source_index"),
                "image_source_index": batch_data[i].get("image_source_index"),
                "sanity_check": batch_data[i].get("sanity_check", "none"),
                "image_mode": batch_data[i].get("image_mode", "rgb"),
                "depth_model": batch_data[i].get("depth_model"),
                "response_time": batch_elapsed / len(outputs),
            })
        
        return results
    
def choice_answer_clean(pred: str):
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred


def extract_letter_answer(queries, predicted_answers):
    """Extract letter answers (A, B, C, D, E) directly from model outputs."""
    if not isinstance(predicted_answers, list):
        return choice_answer_clean(predicted_answers)

    return [choice_answer_clean(answer) for answer in predicted_answers]

def save_model_results(
    model_id,
    accuracy,
    tag_results,
    results,
    dataset,
    sanity_check="none",
    sanity_seed=0,
    image_mode="rgb",
    depth_model=None,
    depth_device=None,
    prompt_mode="cot",
    temperature=0.0,
    max_tokens=10240,
    output_dir=None,
    run_name=None,
    config_path=None,
):
    """Save results for a single model to a JSON file."""
    # Create results directory if it doesn't exist
    results_dir = output_dir or os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create simplified results dictionary focused on accuracy
    results_dict = {
        "model": model_id,
        "dataset": dataset.dataset_name,
        "split": dataset.split,
        "subset": dataset.subset_name,
        "sanity_check": sanity_check,
        "sanity_seed": sanity_seed,
        "image_mode": image_mode,
        "depth_model": depth_model if image_mode == "rgb_depth" else None,
        "depth_device": depth_device if image_mode == "rgb_depth" else None,
        "prompt_mode": prompt_mode,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "run_name": run_name,
        "config_path": config_path,
        "accuracy": accuracy,
        "total_examples": len(results),
        "tag_accuracies": {
            tag: (result["correct"] / result["total"]) * 100 if result["total"] > 0 else 0
            for tag, result in tag_results.items()
        },
        "responses": results
    }
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanity_suffix = f"_{sanity_check}" if sanity_check != "none" else ""
    image_mode_suffix = f"_{image_mode}" if image_mode != "rgb" else ""
    run_prefix = f"{run_name}_" if run_name else ""
    filename = os.path.join(
        results_dir,
        f"{run_prefix}{model_id.split('/')[-1]}_{dataset.subset_name}_{prompt_mode}"
        f"{sanity_suffix}{image_mode_suffix}_{timestamp}.json",
    )
    
    with open(filename, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to {filename}")
    return filename

def evaluate_model(
    model_id,
    dataset,
    max_batch_size=DEFAULT_MAX_BATCH_SIZE,
    tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE,
    sanity_check="none",
    sanity_seed=0,
    image_mode="rgb",
    depth_model=DEFAULT_DEPTH_MODEL,
    depth_device="cpu",
    prompt_mode="cot",
    temperature=0.0,
    max_tokens=10240,
    output_dir=None,
    run_name=None,
    config_path=None,
):
    """Evaluate a single model on the dataset."""
    try:
        evaluator = ModelEvaluator(
            model_id, 
            tensor_parallel_size=tensor_parallel_size,
            prompt_mode=prompt_mode,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        depth_augmenter = None
        if image_mode == "rgb_depth":
            depth_augmenter = DepthImageAugmenter(depth_model, depth_device)
        elif image_mode != "rgb":
            raise ValueError(f"Unsupported image mode: {image_mode}")
        
        results = []
        tag_results = {}
        total_time = 0
        correct = 0
        total = len(dataset)
        sanity_context = build_sanity_check_context(dataset, sanity_check, sanity_seed)
        
        # Process dataset in batches to maximize throughput
        # Start with a reasonable batch size for A100s
        effective_batch_size = max_batch_size
        
        for i in tqdm(range(0, total, effective_batch_size), desc=f"Evaluating {model_id}"):
            batch_items = []
            for j in range(i, min(i + effective_batch_size, total)):
                item = dataset[j]
                item["eval_index"] = j
                batch_items.append(item)
            
            # Format questions for each item in batch
            for item in batch_items:
                item["formatted_question"] = dataset.format_multiple_choice_question(item)

            batch_items = apply_sanity_check_to_batch(batch_items, dataset, sanity_context)
            batch_items = apply_image_mode_to_batch(batch_items, image_mode, depth_augmenter)
            
            # Process batch
            start_time = time.time()
            batch_results = evaluator.process_batch(batch_items)  # Use batch processing with A100s
            end_time = time.time()
            
            batch_time = end_time - start_time
            total_time += batch_time
            
            # Update statistics
            for result in batch_results:
                tag = result["tag"]
                if tag not in tag_results:
                    tag_results[tag] = {"correct": 0, "total": 0}
                
                tag_results[tag]["total"] += 1
                if result["correct"]:
                    correct += 1
                    tag_results[tag]["correct"] += 1
                
                results.append(result)
                
                # # Print progress
                # print(f"Question: {result['question']}")
                # print(f"Predicted: {result['predicted']} (Extracted: {result['predicted_letter']})")
                # print(f"Expected: {result['expected']}")
                # print(f"Correct: {'✓' if result['correct'] else '✗'}")
                # print(f"Response Time: {result['response_time']:.2f}s")
                # print("-" * 40)
            
            # Adaptively increase batch size if things are going well
            if i > 0 and (i % (effective_batch_size * 5)) == 0 and effective_batch_size < max_batch_size:
                effective_batch_size = min(effective_batch_size * 2, max_batch_size)
                print(f"Increasing batch size to {effective_batch_size}")
        
        # Calculate overall accuracy
        accuracy = correct / total * 100 if total > 0 else 0
        avg_response_time = total_time / total if total > 0 else 0
        
        print(f"\n{model_id} Evaluation Complete")
        print(f"Correct Answers: {correct}/{total} ({accuracy:.2f}% accuracy)")
        print(f"Sanity Check: {sanity_check} (seed={sanity_seed})")
        print(f"Image Mode: {image_mode}")
        print(f"Prompt Mode: {prompt_mode}")
        print(f"Temperature: {temperature}")
        print(f"Average Response Time: {avg_response_time:.2f}s")
        
        # Print tag-based breakdown
        print(f"\n{model_id} Success Rate Breakdown by Tag:")
        print("-" * 60)
        print(f"{'Tag':<30} | {'Accuracy':<10} | {'Correct/Total':<15} | {'Avg Time':<10}")
        print("-" * 60)
        
        for tag, result in tag_results.items():
            if result["total"] > 0:
                # Calculate average time for this tag
                tag_times = [r["response_time"] for r in results if r["tag"] == tag]
                avg_tag_time = sum(tag_times) / len(tag_times) if tag_times else 0
                
                tag_accuracy = (result["correct"] / result["total"]) * 100
                print(f"{tag:<30} | {tag_accuracy:>8.2f}% | {result['correct']}/{result['total']} | {avg_tag_time:>8.2f}s")
        
        # Save results for this model
        save_model_results(
            model_id,
            accuracy,
            tag_results,
            results,
            dataset,
            sanity_check=sanity_check,
            sanity_seed=sanity_seed,
            image_mode=image_mode,
            depth_model=depth_model,
            depth_device=depth_device,
            prompt_mode=prompt_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            output_dir=output_dir,
            run_name=run_name,
            config_path=config_path,
        )
        
        return accuracy, tag_results, results
    except Exception as e:
        print(f"Error evaluating {model_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0, {}, []

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate VLM models using vllm on VQA tasks")
    
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=DEFAULT_MODELS,
        help="Model IDs to evaluate (e.g., llava-hf/llava-1.5-7b-hf)"
    )
    
    parser.add_argument(
        "--dataset", 
        default="keplerccc/Robo2VLM-1",
        help="Hugging Face dataset name"
    )
    
    parser.add_argument(
        "--split", 
        default="test",
        help="Dataset split to use (default: test)"
    )

    parser.add_argument(
        "--image_mode",
        "--image-mode",
        dest="image_mode",
        choices=["rgb", "rgb_depth"],
        default="rgb",
        help=(
            "Image input mode. 'rgb' uses the dataset image unchanged. "
            "'rgb_depth' creates an in-memory side-by-side RGB/depth image "
            "for each sample."
        ),
    )

    parser.add_argument(
        "--depth_model",
        "--depth-model",
        dest="depth_model",
        default=DEFAULT_DEPTH_MODEL,
        help="Depth-estimation model used when --image_mode rgb_depth is selected.",
    )

    parser.add_argument(
        "--depth_device",
        "--depth-device",
        dest="depth_device",
        default="cpu",
        help=(
            "Device for the depth-estimation pipeline when --image_mode rgb_depth "
            "is selected. Use 'cpu', 'cuda', 'cuda:0', an integer device id, or 'auto'."
        ),
    )

    parser.add_argument(
        "--subset",
        choices=[
            "full",
            "spatial",
            "affordance",
            "neither",
            "spatial_distance",
            "spatial_direction",
            "spatial_none",
            "affordance_grasp_stability",
            "affordance_object_blockage",
            "affordance_none",
        ],
        default="full",
        help="Evaluate the full split or a curated subset defined in curation_results.jsonl.",
    )

    parser.add_argument(
        "--curation-results",
        default=None,
        help="Path to curation_results.jsonl from recategorize_robo2vlm.py for subset evaluation.",
    )
    
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None,
        help="Maximum number of samples to evaluate (default: 400)"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=DEFAULT_MAX_BATCH_SIZE,
        help=f"Batch size for evaluation (default: {DEFAULT_MAX_BATCH_SIZE})"
    )

    parser.add_argument(
        "--prompt-mode",
        dest="prompt_mode",
        choices=PROMPT_MODE_CHOICES,
        default="cot",
        help="Prompting strategy for multiple-choice evaluation.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature passed to vLLM.",
    )

    parser.add_argument(
        "--max_tokens",
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=10240,
        help="Maximum number of generated tokens per answer.",
    )
    
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=DEFAULT_TENSOR_PARALLEL_SIZE,
        help=f"Number of GPUs to use for tensor parallelism (default: {DEFAULT_TENSOR_PARALLEL_SIZE})"
    )
    
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code for custom models"
    )

    parser.add_argument(
        "--sanity_check",
        choices=["none", "blank", "shuffle", "repeat_first"],
        default="none",
        help=(
            "Optional image ablation to test whether the model is using visual input. "
            "'blank' replaces every image with a black image of the same size, "
            "'shuffle' swaps images across examples with a deterministic derangement, "
            "and 'repeat_first' feeds the first evaluation image to every question."
        ),
    )

    parser.add_argument(
        "--sanity_seed",
        type=int,
        default=0,
        help="Seed used for deterministic sanity-check setups such as shuffled images.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where result JSON files should be written.",
    )

    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional experiment/run name stored in results and used as filename prefix.",
    )

    parser.add_argument(
        "--config-path",
        default=None,
        help="Optional path to the workflow config that launched this evaluation.",
    )
    
    return parser.parse_args()


def load_subset_indices(args):
    """Resolve the evaluation subset and its source indices."""
    if args.subset == "full":
        return None, "full"

    if not args.curation_results:
        raise ValueError(
            "Subset evaluation now requires --curation-results pointing to the "
            "curation_results.jsonl file produced by scripts/recategorize_robo2vlm.py."
        )

    indices = load_subset_source_indices(args.curation_results, args.subset, split=args.split)
    print(f"Loaded {len(indices)} {args.subset} indices from {args.curation_results}")
    return indices, args.subset

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Check if GPUs are available, wait if they're in use
    # wait_for_gpu_availability()
    
    subset_indices, subset_name = load_subset_indices(args)

    print(f"Evaluating {len(args.models)} models on {args.dataset} ({args.split} split)")
    print(f"Subset: {subset_name}")
    print(f"Max samples: {args.max_samples}, Batch size: {args.batch_size}, Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Prompt mode: {args.prompt_mode}, Temperature: {args.temperature}, Max tokens: {args.max_tokens}")
    print(f"Sanity check: {args.sanity_check} (seed={args.sanity_seed})")
    print(f"Image mode: {args.image_mode}")
    if args.run_name:
        print(f"Run name: {args.run_name}")
    if args.output_dir:
        print(f"Output dir: {args.output_dir}")
    if args.image_mode == "rgb_depth":
        print(f"Depth model: {args.depth_model} on {args.depth_device}")
    print(f"Models: {', '.join(args.models)}")
    
    # Load dataset
    dataset = VQADataset(
        args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        sample_indices=subset_indices,
        subset_name=subset_name,
    )
    print(f"Loaded {len(dataset)} examples for evaluation")
    
    # Create results directory
    os.makedirs(os.path.join(SCRIPT_DIR, "results"), exist_ok=True)
    
    # Evaluate each model
    results = {}
    for model_id in args.models:
        print(f"\n{'='*80}")
        print(f"EVALUATING {model_id}")
        print(f"{'='*80}")
        
        try:
            model_results = evaluate_model(
                model_id, 
                dataset, 
                max_batch_size=args.batch_size,
                tensor_parallel_size=args.tensor_parallel_size,
                sanity_check=args.sanity_check,
                sanity_seed=args.sanity_seed,
                image_mode=args.image_mode,
                depth_model=args.depth_model,
                depth_device=args.depth_device,
                prompt_mode=args.prompt_mode,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                output_dir=args.output_dir,
                run_name=args.run_name,
                config_path=args.config_path,
            )
            results[model_id] = model_results
            
        except Exception as e:
            print(f"Error evaluating {model_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[model_id] = (0.0, {}, [])
    
    # Print final accuracy summary
    print("\nEVALUATION SUMMARY")
    print("="*50)
    print(f"{'Model':<35} | {'Accuracy':<10}")
    print("-"*50)
    failed_models = []
    for model_id, (accuracy, _, _) in results.items():
        print(f"{model_id:<35} | {accuracy:>8.2f}%")
        if len(dataset) > 0 and not results[model_id][2]:
            failed_models.append(model_id)

    if failed_models:
        print("\nFailed model(s): " + ", ".join(failed_models), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
