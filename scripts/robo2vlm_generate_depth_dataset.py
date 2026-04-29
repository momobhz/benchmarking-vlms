import os
import torch
from datasets import load_dataset, Dataset
from transformers import pipeline
from PIL import Image


def add_depth_image(example):
    original_image = example['image'].convert("RGB")

    result = depth_pipe(original_image)
    depth_pil = result["depth"]
    depth_rgb = depth_pil.convert("RGB")

    total_width = original_image.width + depth_rgb.width
    max_height = max(original_image.height, depth_rgb.height)

    new_image = Image.new('RGB', (total_width, max_height))
    new_image.paste(original_image, (0, 0))
    new_image.paste(depth_rgb, (original_image.width, 0))

    example['image'] = new_image
    return example


def main():
    device = 0 if torch.cuda.is_available() else -1
    depth_pipe = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device
    )

    dataset = load_dataset("keplerccc/Robo2VLM-1", split="test")

    augmented_dataset = dataset.map(
        add_depth_image,
        fn_kwargs={"pipe": depth_pipe},
        batched=False
    )

    output_dir = "/work/courses/3dv/team43/robo2vlm_qwen_eval_cache/datasets/depth_dataset"

    os.makedirs(output_dir, exist_ok=True)
    augmented_dataset.save_to_disk(output_dir)

if __name__ == "__main__":
    main()
