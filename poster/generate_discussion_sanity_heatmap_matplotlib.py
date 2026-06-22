#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import textwrap
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/vlm_mpl_config")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "pdf"
OUT_PNG = OUT_DIR / "discussion_sanity_heatmap_matplotlib.png"
OUT_PDF = OUT_DIR / "discussion_sanity_heatmap_matplotlib.pdf"

QWEN_SANITY = ROOT / (
    "runs/diagnostics/image_understanding/"
    "qwen25_3b_spatial_cot_200_t0_Qwen2.5-VL-3B-Instruct_spatial_cot_"
    "20260513_170901_Qwen2.5-VL-3B-Instruct_20260517_140842/"
    "image_understanding_sanity.json"
)
DEEPSEEK_SANITY = ROOT / (
    "runs/diagnostics/image_understanding/"
    "deepseek_vl2_tiny_spatial_cot_200_t0_deepseek-vl2-tiny_spatial_cot_"
    "20260513_163446_deepseek-vl2-tiny_spatial_cot_20260517_141222/"
    "image_understanding_sanity.json"
)
if not DEEPSEEK_SANITY.exists():
    DEEPSEEK_SANITY = ROOT / (
        "runs/diagnostics/image_understanding/"
        "deepseek_vl2_tiny_spatial_cot_200_t0_deepseek-vl2-tiny_spatial_cot_"
        "20260513_163446_deepseek-vl2-tiny_20260517_141222/"
        "image_understanding_sanity.json"
    )

BG = "#FFFFFF"
LINE = "#111111"
TEXT = "#111111"
MUTED = TEXT
RED = "#C94F4F"
YELLOW = "#F0D36A"
GREEN = "#3F9F6B"

PROBES = [
    ("explain_correct", "Support\ncorrect"),
    ("explain_wrong", "Reject\nwrong"),
    ("explain_correct_blank", "Reject\nblank"),
    ("explain_correct_shuffle", "Reject\nshuffle"),
]


def load_counts(path: Path) -> dict[str, Counter]:
    data = json.loads(path.read_text(encoding="utf-8"))
    counts: dict[str, Counter] = {}
    for row in data["responses"]:
        counts.setdefault(row["probe_type"], Counter())[row["heuristic_judgment"]] += 1
    return counts


def probe_score(counts: Counter, probe: str) -> float:
    desired_label = "supports" if probe == "explain_correct" else "rejects_or_insufficient"
    desired = counts.get(desired_label, 0)
    contradictory = counts.get("supports", 0) if probe != "explain_correct" else counts.get("rejects_or_insufficient", 0)
    unclear = counts.get("unclear", 0)
    total = max(1, desired + contradictory + unclear)
    return (desired - contradictory) / total


def add_box(ax, x, y, w, h, facecolor, edgecolor=LINE, lw=0.8, zorder=1):
    patch = Rectangle(
        (x, y),
        w,
        h,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        transform=ax.transAxes,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def add_wrapped(ax, x, y, text, width_chars, *, size=8, weight="normal", color=TEXT, va="top", ha="left"):
    ax.text(
        x,
        y,
        "\n".join(textwrap.wrap(text, width_chars)),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=size,
        fontweight=weight,
        color=color,
        linespacing=1.05,
    )


def add_centered(ax, x, y, text, *, size=11, weight="bold", color=TEXT, va="center"):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va=va,
        fontsize=size,
        fontweight=weight,
        color=color,
        linespacing=0.94,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        ("Qwen2.5\nVL-3B", load_counts(QWEN_SANITY)),
        ("DeepSeek\nVL2-tiny", load_counts(DEEPSEEK_SANITY)),
    ]
    matrix = [[probe_score(counts[probe], probe) for probe, _ in PROBES] for _, counts in rows]

    cmap = LinearSegmentedColormap.from_list("grounding_risk", [RED, YELLOW, GREEN], N=256)
    norm = Normalize(vmin=-1, vmax=1)

    fig = plt.figure(figsize=(8.0, 2.75), dpi=300, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.text(
        0.035,
        0.92,
        "Grounding Stress Test",
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        color=TEXT,
        va="top",
    )
    ax.text(
        0.035,
        0.805,
        "Image evidence or question prior?",
        transform=ax.transAxes,
        fontsize=11.8,
        fontweight="bold",
        color=TEXT,
        va="top",
    )

    legend_items = [(RED, "contradicts"), (YELLOW, "unclear"), (GREEN, "expected")]
    lx, ly = 0.53, 0.84
    for idx, (color, label) in enumerate(legend_items):
        x = lx + idx * 0.155
        add_box(ax, x, ly, 0.035, 0.055, color, lw=0.7)
        ax.text(
            x + 0.043,
            ly + 0.028,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color=TEXT,
        )

    x0, y_head = 0.035, 0.565
    model_w, cell_w = 0.17, 0.19
    head_h, row_h = 0.155, 0.255
    gap = 0.0

    add_box(ax, x0, y_head, model_w, head_h, "white")
    add_centered(ax, x0 + model_w / 2, y_head + head_h / 2, "Model", size=12.2)

    for i, (_, title) in enumerate(PROBES):
        x = x0 + model_w + gap + i * (cell_w + gap)
        y = y_head
        add_box(ax, x, y, cell_w, head_h, "white")
        add_centered(ax, x + cell_w / 2, y + head_h / 2, title, size=11.4)

    for ridx, (model, _) in enumerate(rows):
        y = y_head - (ridx + 1) * (row_h + gap)
        add_box(ax, x0, y, model_w, row_h, "white")
        add_centered(ax, x0 + model_w / 2, y + row_h / 2, model, size=11.6)

        for i, value in enumerate(matrix[ridx]):
            x = x0 + model_w + gap + i * (cell_w + gap)
            add_box(ax, x, y, cell_w, row_h, cmap(norm(value)), edgecolor=LINE, lw=0.8)

    fig.savefig(OUT_PNG, dpi=300, facecolor=BG, bbox_inches=None)
    fig.savefig(OUT_PDF, dpi=300, facecolor=BG, bbox_inches=None)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
