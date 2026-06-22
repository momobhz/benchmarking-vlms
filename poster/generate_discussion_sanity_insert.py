#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "pdf"
OUT_PNG = OUT_DIR / "discussion_sanity_insert_compact.png"
OUT_PDF = OUT_DIR / "discussion_sanity_insert_compact.pdf"

QWEN_SANITY = ROOT / (
    "runs/diagnostics/image_understanding/"
    "qwen25_3b_spatial_cot_200_t0_Qwen2.5-VL-3B-Instruct_spatial_cot_"
    "20260513_170901_Qwen2.5-VL-3B-Instruct_20260517_140842/"
    "image_understanding_sanity.json"
)
DEEPSEEK_SANITY = ROOT / (
    "runs/diagnostics/image_understanding/"
    "deepseek_vl2_tiny_spatial_cot_200_t0_deepseek-vl2-tiny_spatial_cot_"
    "20260513_163446_deepseek-vl2-tiny_20260517_141222/"
    "image_understanding_sanity.json"
)

# Poster insert: 255 mm x 58 mm at 300 dpi.
W, H = 3012, 685

COLORS = {
    "bg": "#DCE8F7",
    "panel": "#FFFFFF",
    "ink": "#111111",
    "muted": "#343A40",
    "subtle": "#667085",
    "line": "#97B2D5",
    "blue": "#1F5FAC",
    "blue_dark": "#164A8A",
    "red": "#D95F59",
    "yellow": "#F2D46B",
    "green": "#3BA46B",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    path = Path("/System/Library/Fonts/Supplemental") / name
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


F_TITLE = font(54, True)
F_SUB = font(28)
F_HEAD = font(23, True)
F_HEAD_SMALL = font(19, True)
F_MODEL = font(27, True)
F_BODY = font(23)
F_BODY_B = font(23, True)
F_TINY = font(17, True)


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
    # Desired behavior pulls green, contradictory behavior pulls red, unclear stays neutral yellow.
    return max(-1.0, min(1.0, (desired - contradictory) / total))


def lerp(a: int, b: int, t: float) -> int:
    return round(a + (b - a) * t)


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def mix(c1: str, c2: str, t: float) -> str:
    a, b = hex_to_rgb(c1), hex_to_rgb(c2)
    return rgb_to_hex(tuple(lerp(a[i], b[i], t) for i in range(3)))


def heat_color(score: float) -> str:
    if score < 0:
        return mix(COLORS["red"], COLORS["yellow"], score + 1)
    return mix(COLORS["yellow"], COLORS["green"], score)


def lighten(color: str, amount: float) -> str:
    return mix(color, "#FFFFFF", amount)


def text(draw: ImageDraw.ImageDraw, xy, value: str, fnt, fill: str = COLORS["ink"], anchor: str | None = None):
    draw.text(xy, value, font=fnt, fill=fill, anchor=anchor)


def wrapped(draw: ImageDraw.ImageDraw, xy, value: str, fnt, width_px: int, fill: str, gap: int = 4) -> int:
    x, y = xy
    avg = max(8, int(fnt.size * 0.52))
    for paragraph in value.split("\n"):
        for line in textwrap.wrap(paragraph, max(1, width_px // avg)) or [""]:
            draw.text((x, y), line, font=fnt, fill=fill)
            y += fnt.size + gap
    return y


def rect(draw: ImageDraw.ImageDraw, xy, fill: str, outline: str | None = None, width: int = 1):
    draw.rectangle(xy, fill=fill, outline=outline, width=width)


def rounded(draw: ImageDraw.ImageDraw, xy, radius: int, fill: str, outline: str | None = None, width: int = 1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def gradient_cell(draw: ImageDraw.ImageDraw, xy, score: float):
    x0, y0, x1, y1 = xy
    base = heat_color(score)
    top = lighten(base, 0.36)
    bottom = lighten(base, 0.06)
    h = y1 - y0
    for i in range(h):
        color = mix(top, bottom, i / max(1, h - 1))
        draw.line((x0, y0 + i, x1, y0 + i), fill=color)
    draw.rounded_rectangle(xy, radius=13, outline="#FFFFFF", width=3)


def draw_scale(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int):
    for i in range(w):
        score = -1 + 2 * (i / max(1, w - 1))
        draw.line((x + i, y, x + i, y + h), fill=heat_color(score))
    rect(draw, (x, y, x + w, y + h), fill=None, outline=COLORS["line"], width=2)
    text(draw, (x, y + h + 8), "worse", F_TINY, COLORS["muted"])
    text(draw, (x + w // 2, y + h + 8), "unclear", F_TINY, COLORS["muted"], anchor="ma")
    text(draw, (x + w, y + h + 8), "better", F_TINY, COLORS["muted"], anchor="ra")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qwen = load_counts(QWEN_SANITY)
    deepseek = load_counts(DEEPSEEK_SANITY)

    canvas = Image.new("RGB", (W, H), COLORS["bg"])
    draw = ImageDraw.Draw(canvas)

    text(draw, (38, 26), "Grounding Stress-Test Matrix", F_TITLE, COLORS["blue"])
    wrapped(
        draw,
        (40, 88),
        "Cell hue summarizes sanity behavior: green = expected grounding response, red = contradictory support, yellow = unclear / neutral.",
        F_SUB,
        1880,
        COLORS["muted"],
    )
    draw_scale(draw, 2260, 44, 610, 30)

    probes = [
        ("explain_correct", "Supports correct image", "support"),
        ("explain_wrong", "Rejects wrong answer", "reject"),
        ("explain_correct_blank", "Rejects blank image", "reject"),
        ("explain_correct_shuffle", "Rejects shuffled image", "reject"),
    ]
    rows = [
        ("Qwen2.5-VL-3B", qwen, "Sees absence, misses mismatch"),
        ("DeepSeek-VL2 tiny", deepseek, "Rejects controls, weak positive grounding"),
    ]

    x0, y0 = 40, 170
    model_w, cell_w, fp_w = 330, 390, 560
    head_h, row_h = 92, 152
    gap = 10

    # Headers
    rounded(draw, (x0, y0, x0 + model_w, y0 + head_h), 10, COLORS["panel"], COLORS["line"], 2)
    text(draw, (x0 + 22, y0 + 32), "Model", F_HEAD, COLORS["ink"])
    for i, (_, title, expected) in enumerate(probes):
        x = x0 + model_w + gap + i * (cell_w + gap)
        rounded(draw, (x, y0, x + cell_w, y0 + head_h), 10, COLORS["panel"], COLORS["line"], 2)
        wrapped(draw, (x + 18, y0 + 17), title, F_HEAD_SMALL, cell_w - 36, COLORS["ink"], 1)
        text(draw, (x + 18, y0 + 60), f"Expected: {expected}", F_TINY, COLORS["blue_dark"])

    fp_x = x0 + model_w + gap + 4 * (cell_w + gap)
    rounded(draw, (fp_x, y0, fp_x + fp_w, y0 + head_h), 10, COLORS["panel"], COLORS["line"], 2)
    text(draw, (fp_x + 18, y0 + 31), "Failure fingerprint", F_HEAD_SMALL, COLORS["ink"])

    for ridx, (model, counts, fingerprint) in enumerate(rows):
        y = y0 + head_h + gap + ridx * (row_h + gap)
        rounded(draw, (x0, y, x0 + model_w, y + row_h), 10, COLORS["panel"], COLORS["line"], 2)
        draw.rectangle((x0, y, x0 + 12, y + row_h), fill=COLORS["blue"] if ridx == 0 else "#7C3AED")
        wrapped(draw, (x0 + 28, y + 42), model, F_MODEL, model_w - 48, COLORS["ink"], 2)

        for i, (probe, _, _) in enumerate(probes):
            x = x0 + model_w + gap + i * (cell_w + gap)
            gradient_cell(draw, (x, y, x + cell_w, y + row_h), probe_score(counts[probe], probe))

        rounded(draw, (fp_x, y, fp_x + fp_w, y + row_h), 10, COLORS["panel"], COLORS["line"], 2)
        wrapped(draw, (fp_x + 18, y + 38), fingerprint, F_BODY_B, fp_w - 36, COLORS["blue_dark"] if ridx == 0 else "#6D28D9", 4)

    note_y = y0 + head_h + gap + 2 * (row_h + gap) + 32
    rect(draw, (40, note_y, W - 40, note_y + 92), COLORS["blue"], None, 0)
    wrapped(
        draw,
        (70, note_y + 25),
        "One-glance takeaway: accuracy, control robustness, and explanation grounding are different failure axes.",
        F_BODY_B,
        W - 140,
        "#FFFFFF",
        4,
    )

    canvas.save(OUT_PNG)
    canvas.save(OUT_PDF, "PDF", resolution=300.0)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
