import os
import ast
import json
import base64
import argparse
import re
import time
import random
import subprocess
import numpy as np
from datetime import datetime
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import concurrent.futures
from typing import List, Dict, Tuple, Any
from tqdm import tqdm

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
from transformers import AutoProcessor, AutoTokenizer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_SPATIAL_INDICES_FILE = os.path.join(REPO_ROOT, "test_spatial_indices.json")
DEFAULT_AFFORDANCE_INDICES_FILE = os.path.join(REPO_ROOT, "test_affordance_indices.json")

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


class ModelEvaluator:
    """Evaluate VLM models on VQA tasks."""
    def __init__(self, model_id: str, device="cuda", tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE):
        self.model_id = model_id
        self.device = device
        self.model_type = self._get_model_key()
        
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
        engine_args_dict = model_request_data.engine_args.__dict__
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
            temperature=0.7,
            max_tokens=10240,
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
        
        # Prepend the instruction to each question
        # instructed_questions = [
        #     f"Answer the following multiple choice question by selecting the letter (A, B, C, D, or E). Reason step by step about the answer, and show your work, for each step. Only after that, proceed to the final answer. Please answer the question and provide the correct option letter, e.g., A, B, C, D, E, at the end. {q}" 
        #     for q in questions
        # ]
        instructed_questions = [
            f"Answer the following multiple choice question by selecting the letter (A, B, C, D, or E). ONLY output the correct option letter, i.e., A, B, C, D, E. {q}" 
            for q in questions
        ]
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
                "response_time": (time.time() - start_time) / len(outputs),
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

def save_model_results(model_id, accuracy, tag_results, results, dataset):
    """Save results for a single model to a JSON file."""
    # Create results directory if it doesn't exist
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create simplified results dictionary focused on accuracy
    results_dict = {
        "model": model_id,
        "dataset": dataset.dataset_name,
        "split": dataset.split,
        "subset": dataset.subset_name,
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
    filename = os.path.join(
        results_dir,
        f"{model_id.split('/')[-1]}_{dataset.subset_name}_{timestamp}.json",
    )
    
    with open(filename, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to {filename}")
    return filename

def evaluate_model(model_id, dataset, max_batch_size=DEFAULT_MAX_BATCH_SIZE, tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE):
    """Evaluate a single model on the dataset."""
    try:
        evaluator = ModelEvaluator(
            model_id, 
            tensor_parallel_size=tensor_parallel_size
        )
        
        results = []
        tag_results = {}
        total_time = 0
        correct = 0
        total = len(dataset)
        
        # Process dataset in batches to maximize throughput
        # Start with a reasonable batch size for A100s
        effective_batch_size = max_batch_size
        
        for i in tqdm(range(0, total, effective_batch_size), desc=f"Evaluating {model_id}"):
            batch_items = [dataset[j] for j in range(i, min(i + effective_batch_size, total))]
            
            # Format questions for each item in batch
            for item in batch_items:
                item["formatted_question"] = dataset.format_multiple_choice_question(item)
            
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
        save_model_results(model_id, accuracy, tag_results, results, dataset)
        
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
        "--subset",
        choices=["full", "spatial", "affordance"],
        default="full",
        help="Evaluate the full split or only the spatial/affordance subset defined by source indices.",
    )

    parser.add_argument(
        "--indices_file",
        default=None,
        help="Optional JSON file containing source indices for subset evaluation.",
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
    
    return parser.parse_args()


def load_subset_indices(args):
    """Resolve the evaluation subset and its source indices."""
    if args.subset == "full":
        return None, "full"

    if args.split != "test":
        raise ValueError(
            f"{args.subset} indices are defined for the test split, but --split was set to '{args.split}'."
        )

    if args.indices_file:
        indices_path = args.indices_file
    elif args.subset == "spatial":
        indices_path = DEFAULT_SPATIAL_INDICES_FILE
    else:
        indices_path = DEFAULT_AFFORDANCE_INDICES_FILE

    with open(indices_path, "r") as handle:
        indices = json.load(handle)

    if not isinstance(indices, list) or not all(isinstance(index, int) for index in indices):
        raise ValueError(f"Subset index file must contain a JSON list of integers: {indices_path}")

    print(f"Loaded {len(indices)} {args.subset} indices from {indices_path}")
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
                tensor_parallel_size=args.tensor_parallel_size
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
    for model_id, (accuracy, _, _) in results.items():
        print(f"{model_id:<35} | {accuracy:>8.2f}%")

if __name__ == "__main__":
    main()
