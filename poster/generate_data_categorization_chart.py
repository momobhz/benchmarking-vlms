#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/vlm_mpl_config")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURATION_DIR = ROOT / "outputs" / "robo2vlm_spatial_affordance"
CURATION_DIR = Path(os.environ.get("CURATION_DIR", DEFAULT_CURATION_DIR)).expanduser()
SUMMARY_PATH = CURATION_DIR / "summary.json"
CURATION_PATH = CURATION_DIR / "curation_results.jsonl"

OUT_DIR = ROOT / "output" / "pdf"
OUT_PNG = OUT_DIR / "data_categorization_chart.png"
OUT_TRANSPARENT_PNG = OUT_DIR / "data_categorization_chart_transparent.png"
OUT_PDF = OUT_DIR / "data_categorization_chart.pdf"

TEXT = "#111111"
LINE = "#2B2B2B"
BG = "#FFFFFF"
SPATIAL = "#4C78A8"
SPATIAL_LIGHT = "#D9E7F7"
AFFORDANCE = "#F28E2B"
AFFORDANCE_LIGHT = "#FBE2C3"
NEITHER = "#A0A0A0"
NEITHER_LIGHT = "#E8E8E8"
OTHER = "#D9D9D9"


def sample_count(count: int) -> str:
    return f"{count:,}".replace(",", "'") + " samples"


def wrapped_label(text: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(text, width))


def load_counts() -> tuple[int, dict[str, int], dict[str, Counter]]:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    total = int(summary["processed_count"])
    label_counts = dict(summary["label_counts"])

    nested: dict[str, Counter] = defaultdict(Counter)
    phrase_rules = {
        "spatial_reasoning": {
            "Distance / depth": "point",
            "Direction arrows": "which colored arrow",
        },
        "affordance_understanding": {
            "Object blockage": "obstacle blocking",
            "Grasp stability": "stable",
        },
    }
    with CURATION_PATH.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            label = row.get("curation_label") or row.get("label")
            question = " ".join((row.get("question") or "").split()).lower()
            if label in phrase_rules:
                matched = "Other"
                for subcategory, phrase in phrase_rules[label].items():
                    if phrase in question:
                        matched = subcategory
                        break
                nested[label][matched] += 1
    return total, label_counts, nested


def box(ax, center, size, title, subtitle, face, edge=LINE, title_size=16, subtitle_size=13, title_linespacing=0.92):
    cx, cy = center
    w, h = size
    x, y = cx - w / 2, cy - h / 2
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.016",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.3,
        transform=ax.transAxes,
        zorder=3,
    )
    ax.add_patch(patch)
    title_lines = title.count("\n") + 1
    title_y = cy + h * (0.18 if title_lines == 1 else 0.22)
    subtitle_y = cy - h * (0.24 if title_lines == 1 else 0.34)
    ax.text(
        cx,
        title_y,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=TEXT,
        linespacing=title_linespacing,
        zorder=4,
    )
    ax.text(
        cx,
        subtitle_y,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=subtitle_size,
        fontweight="bold",
        color=TEXT,
        zorder=4,
    )
    return patch


def connector(ax, points, color=LINE, lw=1.0):
    xs, ys = zip(*points)
    ax.plot(xs, ys, transform=ax.transAxes, color=color, linewidth=lw, solid_capstyle="round", zorder=1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total, labels, nested = load_counts()

    spatial = labels["spatial_reasoning"]
    affordance = labels["affordance_understanding"]
    neither = labels["neither"]
    spatial_sub = nested["spatial_reasoning"]
    affordance_sub = nested["affordance_understanding"]

    fig = plt.figure(figsize=(10.2, 3.25), dpi=300, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    source_y = 0.835
    judge_y = 0.635
    box(ax, (0.50, source_y), (0.31, 0.145), "Robo2VLM-1 test split", sample_count(total), "#F7F7F7", title_size=17, subtitle_size=14)
    box(ax, (0.50, judge_y), (0.26, 0.130), "LLM curation", "gpt-4.1-mini", "#FFFFFF", title_size=16, subtitle_size=13)
    connector(ax, [(0.50, source_y - 0.073), (0.50, judge_y + 0.065)], lw=1.0)

    top_y = 0.390
    cats = [
        ("Spatial reasoning", spatial, SPATIAL_LIGHT, SPATIAL, 0.20, (0.255, 0.160), 15.0, 13.2, 0.92),
        ("Affordance\nunderstanding", affordance, AFFORDANCE_LIGHT, AFFORDANCE, 0.50, (0.280, 0.178), 14.4, 12.6, 0.78),
        ("Neither / excluded", neither, NEITHER_LIGHT, "#666666", 0.80, (0.255, 0.160), 15.0, 13.2, 0.92),
    ]
    category_fork_y = 0.525
    for title, count, fill, edge, x, size, title_size, subtitle_size, linespacing in cats:
        connector(ax, [(0.50, judge_y - 0.065), (0.50, category_fork_y), (x, category_fork_y), (x, top_y + 0.080)], color=edge, lw=1.0)
        box(
            ax,
            (x, top_y),
            size,
            wrapped_label(title, 20),
            sample_count(count),
            fill,
            edge=edge,
            title_size=title_size,
            subtitle_size=subtitle_size,
            title_linespacing=linespacing,
        )

    sub_y = 0.135
    sub_boxes = [
        (0.125, "Distance / depth", spatial_sub["Distance / depth"], SPATIAL_LIGHT, SPATIAL, spatial),
        (0.365, "Direction arrows", spatial_sub["Direction arrows"], SPATIAL_LIGHT, SPATIAL, spatial),
        (0.595, "Object blockage", affordance_sub["Object blockage"], AFFORDANCE_LIGHT, AFFORDANCE, affordance),
        (0.835, "Grasp stability", affordance_sub["Grasp stability"], AFFORDANCE_LIGHT, AFFORDANCE, affordance),
    ]
    for x, title, count, fill, edge, parent_count in sub_boxes:
        parent_x = 0.20 if "spatial" in title.lower() or title in {"Distance / depth", "Direction arrows"} else 0.50
        fork_y = 0.255
        connector(ax, [(parent_x, top_y - 0.080), (parent_x, fork_y), (x, fork_y), (x, sub_y + 0.070)], color=edge, lw=0.85)
        box(ax, (x, sub_y), (0.175, 0.145), wrapped_label(title, 16), sample_count(count), fill, edge=edge, title_size=12.6, subtitle_size=10.9)

    fig.savefig(OUT_PNG, dpi=300, facecolor=BG, bbox_inches=None)
    fig.savefig(OUT_TRANSPARENT_PNG, dpi=300, transparent=True, facecolor="none", bbox_inches=None)
    fig.savefig(OUT_PDF, dpi=300, facecolor=BG, bbox_inches=None)
    print(OUT_PNG)
    print(OUT_TRANSPARENT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
