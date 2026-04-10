#!/usr/bin/env python3
"""Create an interactive Plotly scatter plot from clustered Robo2VLM CSV data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


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

        rows: list[dict[str, object]] = []
        for row in reader:
            rows.append(
                {
                    "source_index": int(row["source_index"]),
                    "id": row["id"],
                    "label": row["label"],
                    "question": row["question"],
                    "umap_x": float(row["umap_x"]),
                    "umap_y": float(row["umap_y"]),
                }
            )
    return rows


def build_figure(
    rows: list[dict[str, object]],
    title: str,
    marker_size: float,
    marker_opacity: float,
):
    try:
        import plotly.express as px
    except ImportError as exc:
        raise SystemExit(
            "The 'plotly' package is required. Install it with 'pip install plotly'."
        ) from exc

    fig = px.scatter(
        rows,
        x="umap_x",
        y="umap_y",
        color="label",
        category_orders={"label": LABEL_ORDER},
        color_discrete_map=LABEL_COLORS,
        hover_name="id",
        hover_data={
            "label": True,
            "question": True,
            "source_index": True,
            "umap_x": ":.3f",
            "umap_y": ":.3f",
        },
        render_mode="webgl",
        title=title,
        labels={
            "umap_x": "UMAP-1",
            "umap_y": "UMAP-2",
            "label": "Category",
            "source_index": "Source Index",
        },
    )
    fig.update_traces(marker={"size": marker_size, "opacity": marker_opacity})
    fig.update_layout(legend_title_text="Category")
    return fig


def main() -> None:
    args = parse_args()
    output_html = args.output_html or default_output_html(args.input_csv)

    rows = load_projection_rows(args.input_csv)
    if not rows:
        raise SystemExit(f"No rows found in input CSV: {args.input_csv}")

    fig = build_figure(
        rows=rows,
        title=args.title,
        marker_size=args.marker_size,
        marker_opacity=args.marker_opacity,
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    include_plotlyjs = True if args.include_plotlyjs == "embed" else args.include_plotlyjs
    fig.write_html(output_html, include_plotlyjs=include_plotlyjs)

    print(f"Saved interactive plot to {output_html}")


if __name__ == "__main__":
    main()
