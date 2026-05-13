#!/usr/bin/env python3
"""Create an interactive Plotly scatter plot from clustered Robo2VLM CSV data."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for import_path in (REPO_ROOT, SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from vlm_bench.curation.taxonomy import NO_SUBCATEGORY_GROUP


LABEL_ORDER = [
    "spatial_reasoning",
    "affordance_understanding",
    "neither",
]
LABEL_COLORS = {
    "spatial_reasoning": "#1f77b4",
    "affordance_understanding": "#ff7f0e",
    "neither": "#7f7f7f",
}
SUBCATEGORY_ORDER = [
    "distance",
    "direction",
    "grasp_stability",
    "object_blockage",
    "none",
    NO_SUBCATEGORY_GROUP,
]
SUBCATEGORY_COLORS = {
    "distance": "#2b8cbe",
    "direction": "#8856a7",
    "grasp_stability": "#31a354",
    "object_blockage": "#e6550d",
    "none": "#969696",
    NO_SUBCATEGORY_GROUP: "#c7c7c7",
}
REQUIRED_COLUMNS = {"source_index", "id", "label", "question", "umap_x", "umap_y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an interactive Plotly scatter plot from Robo2VLM UMAP CSV data."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV file produced by visualize_robo2vlm_question_clusters.py.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Path to write the interactive HTML plot. Defaults next to the input CSV.",
    )
    parser.add_argument(
        "--title",
        default="Robo2VLM Question Embeddings Projected with UMAP",
        help="Figure title.",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=5.0,
        help="Marker size for the scatter plot.",
    )
    parser.add_argument(
        "--marker-opacity",
        type=float,
        default=0.6,
        help="Marker opacity for the scatter plot.",
    )
    parser.add_argument(
        "--include-plotlyjs",
        choices=["cdn", "directory", "embed"],
        default="cdn",
        help="How to include Plotly JS in the exported HTML.",
    )
    parser.add_argument(
        "--color-by",
        choices=["label", "subcategory"],
        default="label",
        help="Color the interactive plot by the top-level label or deterministic subcategory.",
    )
    return parser.parse_args()


def default_output_html(input_csv: Path) -> Path:
    return input_csv.with_name(f"{input_csv.stem}_interactive.html")


def load_projection_rows(input_csv: Path) -> list[dict[str, object]]:
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - fieldnames
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(f"Input CSV is missing required columns: {missing_text}")
        has_subcategory_column = "subcategory" in fieldnames

        rows: list[dict[str, object]] = []
        for row in reader:
            rows.append(
                {
                    "source_index": int(row["source_index"]),
                    "id": row["id"],
                    "label": row["label"],
                    "subcategory": (row.get("subcategory", "") if has_subcategory_column else None),
                    "question": row["question"],
                    "umap_x": float(row["umap_x"]),
                    "umap_y": float(row["umap_y"]),
                }
            )
    return rows


def display_group_value(row: dict[str, object], color_by: str) -> str:
    if color_by == "label":
        return str(row["label"])
    subcategory = row.get("subcategory")
    return str(subcategory) if isinstance(subcategory, str) and subcategory else NO_SUBCATEGORY_GROUP


def build_figure(
    rows: list[dict[str, object]],
    title: str,
    color_by: str,
    marker_size: float,
    marker_opacity: float,
):
    try:
        import plotly.express as px
    except ImportError as exc:
        raise SystemExit(
            "The 'plotly' package is required. Install it with 'pip install plotly'."
        ) from exc

    color_column = "label" if color_by == "label" else "subcategory_display"
    category_orders = {"label": LABEL_ORDER}
    color_discrete_map = LABEL_COLORS
    if color_by == "subcategory":
        category_orders = {"subcategory_display": SUBCATEGORY_ORDER}
        color_discrete_map = SUBCATEGORY_COLORS

    fig = px.scatter(
        rows,
        x="umap_x",
        y="umap_y",
        color=color_column,
        category_orders=category_orders,
        color_discrete_map=color_discrete_map,
        hover_name="id",
        hover_data={
            "label": True,
            "subcategory": True,
            "question": True,
            "source_index": True,
            "umap_x": ":.3f",
            "umap_y": ":.3f",
            "subcategory_display": color_by == "subcategory",
        },
        render_mode="webgl",
        title=title,
        labels={
            "umap_x": "UMAP-1",
            "umap_y": "UMAP-2",
            "label": "Category",
            "subcategory_display": "Subcategory",
            "source_index": "Source Index",
        },
    )
    fig.update_traces(marker={"size": marker_size, "opacity": marker_opacity})
    fig.update_layout(legend_title_text="Category" if color_by == "label" else "Subcategory")
    return fig


def main() -> None:
    args = parse_args()
    output_html = args.output_html or default_output_html(args.input_csv)

    rows = load_projection_rows(args.input_csv)
    if not rows:
        raise SystemExit(f"No rows found in input CSV: {args.input_csv}")
    if args.color_by == "subcategory" and all(row.get("subcategory") is None for row in rows):
        raise SystemExit(
            "Input CSV does not include subcategory values. "
            "Regenerate it with visualize_robo2vlm_question_clusters.py."
        )
    for row in rows:
        row["subcategory_display"] = display_group_value(row, args.color_by)

    fig = build_figure(
        rows=rows,
        title=args.title,
        color_by=args.color_by,
        marker_size=args.marker_size,
        marker_opacity=args.marker_opacity,
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    include_plotlyjs = True if args.include_plotlyjs == "embed" else args.include_plotlyjs
    fig.write_html(output_html, include_plotlyjs=include_plotlyjs)

    print(f"Saved interactive plot to {output_html}")


if __name__ == "__main__":
    main()
