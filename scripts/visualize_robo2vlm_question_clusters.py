#!/usr/bin/env python3
"""Embed Robo2VLM questions and visualize category structure in 2D with UMAP.

This script uses the existing spatial and affordance index files to assign
cluster labels to the Robo2VLM split:

- spatial_reasoning
- affordance_understanding
- neither

It then embeds every question with a sentence-transformer model, projects the
embeddings to two dimensions with UMAP, and writes both a CSV of coordinates
and a scatter plot.

Example:
    python3 scripts/visualize_robo2vlm_question_clusters.py \
        --output-dir outputs/robo2vlm_umap

Dependencies:
    pip install datasets sentence-transformers umap-learn matplotlib scikit-learn tqdm
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_DATASET_NAME = "keplerccc/Robo2VLM-1"
DEFAULT_SPLIT = "test"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SPATIAL_LABEL = "spatial_reasoning"
AFFORDANCE_LABEL = "affordance_understanding"
NEITHER_LABEL = "neither"

LABEL_ORDER = [SPATIAL_LABEL, AFFORDANCE_LABEL, NEITHER_LABEL]
LABEL_COLORS = {
    SPATIAL_LABEL: "#1f77b4",
    AFFORDANCE_LABEL: "#ff7f0e",
    NEITHER_LABEL: "#7f7f7f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed Robo2VLM questions and visualize the category split with UMAP."
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Hugging Face dataset name to load (default: %(default)s).",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Dataset split to visualize (default: %(default)s).",
    )
    parser.add_argument(
        "--spatial-indices",
        type=Path,
        default=REPO_ROOT / "full_test_spatial_indices.json",
        help="JSON file containing source indices labeled as spatial reasoning.",
    )
    parser.add_argument(
        "--affordance-indices",
        type=Path,
        default=REPO_ROOT / "full_test_affordance_indices.json",
        help="JSON file containing source indices labeled as affordance understanding.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the saved projection CSV, plot image, and summary JSON.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence-transformers model name or local path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Embedding batch size (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Embedding device override, for example 'cpu', 'cuda', or 'mps'.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of split examples to process.",
    )
    parser.add_argument(
        "--streaming",
        dest="streaming",
        action="store_true",
        help="Stream the dataset instead of materializing it locally.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Disable datasets streaming mode and materialize the split locally.",
    )
    parser.set_defaults(streaming=True)
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="Optional PCA dimension before UMAP; set to 0 to disable.",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=30,
        help="UMAP n_neighbors parameter (default: %(default)s).",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.05,
        help="UMAP min_dist parameter (default: %(default)s).",
    )
    parser.add_argument(
        "--umap-metric",
        default="cosine",
        help="UMAP metric (default: %(default)s).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for UMAP and plotting order (default: %(default)s).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=4.0,
        help="Scatter point size in the output plot (default: %(default)s).",
    )
    parser.add_argument(
        "--point-alpha",
        type=float,
        default=0.35,
        help="Scatter transparency in the output plot (default: %(default)s).",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=12.0,
        help="Output figure width in inches (default: %(default)s).",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=9.0,
        help="Output figure height in inches (default: %(default)s).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Saved figure DPI (default: %(default)s).",
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Also save the dense question embeddings as a .npy file.",
    )
    return parser.parse_args()


def load_index_set(path: Path) -> set[int]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"Expected {path} to contain a JSON list of source indices.")
    return {int(index) for index in raw}


def validate_label_sets(spatial_indices: set[int], affordance_indices: set[int]) -> None:
    overlap = spatial_indices & affordance_indices
    if overlap:
        head = sorted(overlap)[:10]
        raise ValueError(
            "Spatial and affordance index files overlap. "
            f"First overlapping indices: {head}"
        )


def label_for_index(
    source_index: int,
    spatial_indices: set[int],
    affordance_indices: set[int],
) -> str:
    if source_index in spatial_indices:
        return SPATIAL_LABEL
    if source_index in affordance_indices:
        return AFFORDANCE_LABEL
    return NEITHER_LABEL


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def load_question_records(
    dataset_name: str,
    split: str,
    spatial_indices: set[int],
    affordance_indices: set[int],
    streaming: bool,
    max_samples: int | None = None,
) -> list[dict[str, object]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The 'datasets' package is required. Install it with "
            "'pip install datasets'."
        ) from exc

    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    if not streaming and max_samples is not None:
        limit = min(max_samples, len(dataset))
        dataset = dataset.select(range(limit))

    records: list[dict[str, object]] = []
    for source_index, row in enumerate(dataset):
        if streaming and max_samples is not None and source_index >= max_samples:
            break
        records.append(
            {
                "source_index": source_index,
                "id": row.get("id", str(source_index)),
                "question": row["question"],
                "label": label_for_index(
                    source_index=source_index,
                    spatial_indices=spatial_indices,
                    affordance_indices=affordance_indices,
                ),
            }
        )
    return records


def infer_device(device: str | None) -> str | None:
    if device:
        return device

    try:
        import torch
    except ImportError:
        return None

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_questions(
    questions: Iterable[str],
    model_name: str,
    batch_size: int,
    device: str | None,
):
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "The 'sentence-transformers' package is required. Install it with "
            "'pip install sentence-transformers'."
        ) from exc

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        list(questions),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def maybe_apply_pca(embeddings, pca_components: int):
    if pca_components <= 0 or embeddings.shape[1] <= pca_components:
        return embeddings

    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise SystemExit(
            "The 'scikit-learn' package is required for PCA preprocessing. "
            "Install it with 'pip install scikit-learn'."
        ) from exc

    reducer = PCA(n_components=pca_components, random_state=0)
    return reducer.fit_transform(embeddings)


def run_umap(embeddings, n_neighbors: int, min_dist: float, metric: str, random_state: int):
    try:
        import umap
    except ImportError as exc:
        raise SystemExit(
            "The 'umap-learn' package is required. Install it with "
            "'pip install umap-learn'."
        ) from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=True,
    )
    return reducer.fit_transform(embeddings)


def write_projection_csv(records: list[dict[str, object]], coordinates, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_index", "id", "label", "question", "umap_x", "umap_y"])
        for record, (x_coord, y_coord) in zip(records, coordinates):
            writer.writerow(
                [
                    record["source_index"],
                    record["id"],
                    record["label"],
                    record["question"],
                    float(x_coord),
                    float(y_coord),
                ]
            )


def plot_projection(
    records: list[dict[str, object]],
    coordinates,
    output_path: Path,
    point_size: float,
    point_alpha: float,
    figure_width: float,
    figure_height: float,
    dpi: int,
    random_state: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "Matplotlib is required for plotting. Install it with "
            "'pip install matplotlib'."
        ) from exc

    labels = np.asarray([record["label"] for record in records], dtype=object)
    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(records))
    shuffled_coordinates = coordinates[order]
    shuffled_labels = labels[order]

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), constrained_layout=True)
    for label in LABEL_ORDER:
        mask = shuffled_labels == label
        if not np.any(mask):
            continue
        ax.scatter(
            shuffled_coordinates[mask, 0],
            shuffled_coordinates[mask, 1],
            s=point_size,
            alpha=point_alpha,
            c=LABEL_COLORS[label],
            label=label.replace("_", " "),
            linewidths=0,
            rasterized=True,
        )

        centroid = shuffled_coordinates[mask].mean(axis=0)
        ax.text(
            centroid[0],
            centroid[1],
            label.replace("_", "\n"),
            fontsize=10,
            fontweight="bold",
            color=LABEL_COLORS[label],
            ha="center",
            va="center",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2},
        )

    ax.set_title("Robo2VLM Question Embeddings Projected with UMAP")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=True)
    ax.grid(alpha=0.15, linewidth=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    records: list[dict[str, object]],
    args: argparse.Namespace,
    embedding_shape: tuple[int, ...],
    output_path: Path,
) -> None:
    counts = Counter(record["label"] for record in records)
    summary = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "embedding_model": args.embedding_model,
        "num_questions": len(records),
        "embedding_shape": list(embedding_shape),
        "streaming": args.streaming,
        "label_counts": {label: counts.get(label, 0) for label in LABEL_ORDER},
        "spatial_indices_file": str(args.spatial_indices),
        "affordance_indices_file": str(args.affordance_indices),
        "umap": {
            "n_neighbors": args.umap_neighbors,
            "min_dist": args.umap_min_dist,
            "metric": args.umap_metric,
            "random_state": args.random_state,
        },
        "pca_components": args.pca_components,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    import numpy as np

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spatial_indices = load_index_set(args.spatial_indices)
    affordance_indices = load_index_set(args.affordance_indices)
    validate_label_sets(spatial_indices, affordance_indices)

    records = load_question_records(
        dataset_name=args.dataset_name,
        split=args.split,
        spatial_indices=spatial_indices,
        affordance_indices=affordance_indices,
        streaming=args.streaming,
        max_samples=args.max_samples,
    )
    if not records:
        raise SystemExit("No questions were loaded from the requested dataset split.")

    questions = [str(record["question"]) for record in records]
    device = infer_device(args.device)
    embeddings = embed_questions(
        questions=questions,
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        device=device,
    )

    reduced_input = maybe_apply_pca(embeddings, args.pca_components)
    coordinates = run_umap(
        embeddings=reduced_input,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.random_state,
    )
    coordinates = np.asarray(coordinates, dtype=np.float32)

    stem = f"{sanitize_name(args.split)}_{sanitize_name(args.embedding_model)}_umap"
    csv_path = args.output_dir / f"{stem}.csv"
    plot_path = args.output_dir / f"{stem}.png"
    summary_path = args.output_dir / f"{stem}_summary.json"
    embeddings_path = args.output_dir / f"{stem}_embeddings.npy"

    write_projection_csv(records, coordinates, csv_path)
    plot_projection(
        records=records,
        coordinates=coordinates,
        output_path=plot_path,
        point_size=args.point_size,
        point_alpha=args.point_alpha,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        dpi=args.dpi,
        random_state=args.random_state,
    )
    write_summary(records, args, embeddings.shape, summary_path)

    if args.save_embeddings:
        np.save(embeddings_path, embeddings)

    print(f"Saved projection CSV to {csv_path}")
    print(f"Saved plot to {plot_path}")
    print(f"Saved summary to {summary_path}")
    if args.save_embeddings:
        print(f"Saved embeddings to {embeddings_path}")


if __name__ == "__main__":
    main()
