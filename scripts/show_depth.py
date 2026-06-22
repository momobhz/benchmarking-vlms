#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import os
from typing import Iterable, Optional, Tuple

import torch
from datasets import load_dataset, load_from_disk
from PIL import Image
from transformers import pipeline


def _ensure_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    # datasets Image feature sometimes returns dicts in edge cases
    if isinstance(img, dict):
        if "path" in img and img["path"] and os.path.exists(img["path"]):
            return Image.open(img["path"])
        if "bytes" in img and img["bytes"]:
            from io import BytesIO

            return Image.open(BytesIO(img["bytes"]))
    raise TypeError(f"Unsupported image type: {type(img)}")


def _make_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    left = left.convert("RGB")
    right = right.convert("RGB")
    h = max(left.height, right.height)
    w = left.width + right.width
    out = Image.new("RGB", (w, h))
    out.paste(left, (0, 0))
    out.paste(right, (left.width, 0))
    return out


def _split_concatenated(img: Image.Image) -> Tuple[Image.Image, Image.Image]:
    img = img.convert("RGB")
    mid = img.width // 2
    left = img.crop((0, 0, mid, img.height))
    right = img.crop((mid, 0, img.width, img.height))
    return left, right


def _show_pair(
    original: Image.Image,
    depth_vis: Image.Image,
    *,
    title: str,
    save_path: Optional[str],
    no_gui: bool,
) -> None:
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        _make_side_by_side(original, depth_vis).save(save_path)

    if no_gui:
        return

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(title)
    axes[0].imshow(original)
    axes[0].set_title("original")
    axes[0].axis("off")
    axes[1].imshow(depth_vis)
    axes[1].set_title("depth (visualization)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


def _iter_streamed(ds: Iterable[dict], start_idx: int, num: int) -> Iterable[Tuple[int, dict]]:
    it = itertools.islice(ds, start_idx, start_idx + num)
    for offset, ex in enumerate(it):
        yield start_idx + offset, ex


def main() -> None:
    ap = argparse.ArgumentParser(description="Preview depth images (streaming or local saved dataset).")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--stream", action="store_true", help="Stream from HF dataset and generate depth on-the-fly.")
    src.add_argument("--from-disk", type=str, help="Path to a dataset saved with datasets.save_to_disk().")

    ap.add_argument("--dataset", type=str, default="keplerccc/Robo2VLM-1", help="HF dataset name (stream mode).")
    ap.add_argument("--split", type=str, default="test", help="Dataset split (stream mode).")
    ap.add_argument("--idx", type=int, default=0, help="Index to show (stream: skip to idx).")
    ap.add_argument("--num", type=int, default=1, help="Number of samples to show (stream mode).")

    ap.add_argument(
        "--model",
        type=str,
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="Depth estimation model id.",
    )
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")

    ap.add_argument("--save", type=str, default=None, help="Optional path to save a side-by-side PNG/JPG.")
    ap.add_argument("--save-dir", type=str, default=None, help="If set, save each sample to this directory.")
    ap.add_argument("--no-gui", action="store_true", help="Do not open a window; only save if --save/--save-dir set.")
    ap.add_argument(
        "--assume-concat",
        action="store_true",
        help="In --from-disk mode: assume stored image is concat(original|depth) and split it for display.",
    )

    args = ap.parse_args()

    if args.save and args.save_dir:
        raise SystemExit("Use either --save or --save-dir (not both).")
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.device == "cpu":
        device = -1
    elif args.device == "cuda":
        device = 0
    else:
        device = 0 if torch.cuda.is_available() else -1

    depth_pipe = None
    if args.stream:
        depth_pipe = pipeline("depth-estimation", model=args.model, device=device)

        ds = load_dataset(args.dataset, split=args.split, streaming=True)
        for i, ex in _iter_streamed(ds, args.idx, args.num):
            original = _ensure_pil(ex["image"]).convert("RGB")
            depth_vis = depth_pipe(original)["depth"]

            save_path = args.save
            if args.save_dir:
                save_path = os.path.join(args.save_dir, f"depth_preview_{i:06d}.png")

            _show_pair(
                original,
                depth_vis,
                title=f"{args.dataset}:{args.split} idx={i}",
                save_path=save_path,
                no_gui=args.no_gui,
            )
        return

    ds = load_from_disk(args.from_disk)
    ex = ds[args.idx]
    img = _ensure_pil(ex["image"]).convert("RGB")

    if args.assume_concat:
        original, depth_vis = _split_concatenated(img)
    else:
        original, depth_vis = img, img

    save_path = args.save
    if args.save_dir:
        save_path = os.path.join(args.save_dir, f"depth_dataset_{args.idx:06d}.png")

    _show_pair(
        original,
        depth_vis,
        title=f"from-disk idx={args.idx}",
        save_path=save_path,
        no_gui=args.no_gui,
    )


if __name__ == "__main__":
    main()
