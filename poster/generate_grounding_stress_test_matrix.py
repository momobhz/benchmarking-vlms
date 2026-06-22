#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "pdf"
OUT_PNG = OUT_DIR / "grounding_stress_test_matrix.png"
OUT_PDF = OUT_DIR / "grounding_stress_test_matrix.pdf"

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
SOURCE_408 = ROOT / "tmp" / "source_000408_depth.png"

W, H = 4961, 3508  # A3 landscape at 300 dpi.
M = 170

COLORS = {
    "paper": "#F7F9FC",
    "panel": "#FFFFFF",
    "ink": "#0F172A",
    "muted": "#475569",
    "subtle": "#64748B",
    "line": "#CBD5E1",
    "line2": "#E2E8F0",
    "green": "#16A34A",
    "red": "#DC2626",
    "gray": "#94A3B8",
    "teal": "#0F766E",
    "blue": "#2563EB",
    "purple": "#7C3AED",
    "pink_bg": "#FDF2F8",
    "pink_line": "#FBCFE8",
    "green_bg": "#ECFDF5",
    "green_line": "#99F6E4",
    "blue_bg": "#EFF6FF",
    "blue_line": "#BFDBFE",
    "purple_bg": "#F5F3FF",
    "purple_line": "#DDD6FE",
    "amber_bg": "#FFFBEB",
    "amber_line": "#FDE68A",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    path = Path("/System/Library/Fonts/Supplemental") / name
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


F_TITLE = font(88, True)
F_SUBTITLE = font(42, False)
F_H2 = font(54, True)
F_H3 = font(38, True)
F_BODY = font(34, False)
F_BODY_B = font(34, True)
F_SMALL = font(27, False)
F_SMALL_B = font(27, True)
F_TINY = font(23, False)
F_TINY_B = font(23, True)


def load_counts(path: Path) -> dict[str, Counter]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_probe: dict[str, Counter] = {}
    for row in data["responses"]:
        by_probe.setdefault(row["probe_type"], Counter())[row["heuristic_judgment"]] += 1
    return by_probe


def desired_breakdown(counts: Counter, probe: str) -> tuple[int, int, int]:
    desired = counts.get("supports" if probe == "explain_correct" else "rejects_or_insufficient", 0)
    contradictory = counts.get("supports", 0) if probe != "explain_correct" else counts.get("rejects_or_insufficient", 0)
    unclear = counts.get("unclear", 0)
    return desired, contradictory, unclear


def rounded(draw: ImageDraw.ImageDraw, xy, radius: int, fill: str, outline: str | None = None, width: int = 1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def text(draw: ImageDraw.ImageDraw, xy, value: str, fnt, fill: str = COLORS["ink"], anchor: str | None = None):
    draw.text(xy, value, font=fnt, fill=fill, anchor=anchor)


def wrapped(draw: ImageDraw.ImageDraw, xy, value: str, fnt, width_px: int, fill: str = COLORS["ink"], line_gap: int = 8):
    x, y = xy
    avg_char = max(8, int(fnt.size * 0.52))
    lines: list[str] = []
    for paragraph in value.split("\n"):
        lines.extend(textwrap.wrap(paragraph, max(1, width_px // avg_char)) or [""])
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y


def fit_image(path: Path, box: tuple[int, int, int, int], crop: bool = False) -> Image.Image:
    img = Image.open(path).convert("RGB")
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if crop:
        scale = max(bw / img.width, bh / img.height)
    else:
        scale = min(bw / img.width, bh / img.height)
    nw, nh = int(img.width * scale), int(img.height * scale)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    if crop:
        left = max(0, (nw - bw) // 2)
        top = max(0, (nh - bh) // 2)
        return img.crop((left, top, left + bw, top + bh))
    out = Image.new("RGB", (bw, bh), "white")
    out.paste(img, ((bw - nw) // 2, (bh - nh) // 2))
    return out


def draw_dot_strip(draw: ImageDraw.ImageDraw, x: int, y: int, desired: int, bad: int, unclear: int):
    colors = [COLORS["green"]] * desired + [COLORS["red"]] * bad + [COLORS["gray"]] * unclear
    colors = colors[:20] + [COLORS["gray"]] * max(0, 20 - len(colors))
    r, gap = 13, 9
    for i, color in enumerate(colors):
        cx = x + (i % 10) * (2 * r + gap)
        cy = y + (i // 10) * (2 * r + gap)
        draw.ellipse((cx, cy, cx + 2 * r, cy + 2 * r), fill=color)


def draw_probe_cell(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    desired: int,
    bad: int,
    unclear: int,
    label: str,
):
    rounded(draw, (x, y, x + w, y + h), 24, "#FFFFFF", COLORS["line"], 3)
    draw_dot_strip(draw, x + 36, y + 34, desired, bad, unclear)
    text(draw, (x + 36, y + 112), f"{desired}/20", F_H3, COLORS["green"])
    text(draw, (x + 158, y + 122), label, F_SMALL_B, COLORS["muted"])
    if bad:
        text(draw, (x + 36, y + 162), f"{bad}/20 contrad.", F_TINY_B, COLORS["red"])
    if unclear:
        text(draw, (x + 36, y + 194), f"{unclear}/20 unclear", F_TINY_B, COLORS["gray"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qwen = load_counts(QWEN_SANITY)
    deep = load_counts(DEEPSEEK_SANITY)

    canvas = Image.new("RGB", (W, H), COLORS["paper"])
    draw = ImageDraw.Draw(canvas)

    # Header
    rounded(draw, (M, 110, W - M, 405), 36, COLORS["panel"], COLORS["line"], 4)
    draw.rectangle((M, 110, W - M, 132), fill=COLORS["teal"])
    text(draw, (M + 60, 166), "Grounding Stress-Test Matrix", F_TITLE)
    text(
        draw,
        (M + 64, 276),
        "Sanity probes separate answer accuracy, control robustness, and explanation grounding.",
        F_SUBTITLE,
        COLORS["muted"],
    )
    rounded(draw, (W - M - 1010, 172, W - M - 60, 348), 28, COLORS["green_bg"], COLORS["green_line"], 4)
    wrapped(
        draw,
        (W - M - 970, 210),
        "One-glance insight: a model can reject blank images yet still rationalize wrong or mismatched visual evidence.",
        F_BODY_B,
        890,
        COLORS["teal"],
        7,
    )

    # Matrix panel
    matrix = (M, 455, W - M, 2090)
    rounded(draw, matrix, 36, COLORS["panel"], COLORS["line"], 4)
    text(draw, (M + 52, 510), "1. Probe outcomes", F_H2)
    text(draw, (M + 52, 580), "Green = expected grounding behavior observed, red = contradictory support, gray = heuristic unclear.", F_BODY, COLORS["muted"])

    probes = [
        ("explain_correct", "Supports\ncorrect image", "support"),
        ("explain_wrong", "Rejects\nwrong answer", "reject"),
        ("explain_correct_blank", "Rejects\nblank image", "reject"),
        ("explain_correct_shuffle", "Rejects\nshuffled image", "reject"),
    ]
    x_model, y_header = M + 60, 690
    col_w, gap = 575, 24
    start_x = x_model + 600
    for i, (_, title, expected) in enumerate(probes):
        x = start_x + i * (col_w + gap)
        rounded(draw, (x, y_header, x + col_w, y_header + 170), 24, COLORS["blue_bg"], COLORS["blue_line"], 3)
        wrapped(draw, (x + 30, y_header + 24), title, F_H3, col_w - 60, COLORS["ink"], 3)
        text(draw, (x + 30, y_header + 118), f"Expected: {expected}", F_SMALL_B, COLORS["blue"])

    fp_x = start_x + 4 * (col_w + gap)
    rounded(draw, (fp_x, y_header, W - M - 60, y_header + 170), 24, COLORS["amber_bg"], COLORS["amber_line"], 3)
    text(draw, (fp_x + 30, y_header + 34), "Failure fingerprint", F_H3)
    wrapped(draw, (fp_x + 30, y_header + 91), "Compact interpretation of the probe pattern.", F_SMALL_B, 570, COLORS["muted"])

    rows = [
        ("Qwen2.5-VL-3B", qwen, COLORS["blue"], "Sees absence,\nmisses mismatch", "Detects blank inputs, but often accepts wrong answers or shuffled evidence."),
        ("DeepSeek-VL2 tiny", deep, COLORS["purple"], "Rejects controls,\nweak positive grounding", "Flags blank and shuffled controls, but rarely phrases correct-image explanations as support."),
    ]
    y0 = 910
    for ridx, (model, counts, color, fingerprint, fp_text) in enumerate(rows):
        y = y0 + ridx * 515
        rounded(draw, (M + 60, y, M + 560, y + 390), 28, "#FFFFFF", COLORS["line"], 3)
        draw.rectangle((M + 60, y, M + 78, y + 390), fill=color)
        wrapped(draw, (M + 105, y + 56), model, F_H3, 390, COLORS["ink"], 5)
        text(draw, (M + 105, y + 180), "n = 20 probes per type", F_SMALL_B, COLORS["subtle"])
        text(draw, (M + 105, y + 232), "Sanity source: spatial CoT", F_SMALL, COLORS["subtle"])

        for i, (probe, _, label) in enumerate(probes):
            desired, bad, unclear = desired_breakdown(counts[probe], probe)
            draw_probe_cell(draw, start_x + i * (col_w + gap), y, col_w, 390, desired, bad, unclear, label)

        rounded(draw, (fp_x, y, W - M - 60, y + 390), 28, "#FFFFFF", COLORS["line"], 3)
        wrapped(draw, (fp_x + 34, y + 42), fingerprint, F_H3, 520, color, 5)
        wrapped(draw, (fp_x + 34, y + 160), fp_text, F_BODY, 550, COLORS["muted"], 7)

    # Legend
    legend_y = 1982
    for x, label, color in [
        (M + 60, "expected behavior observed", COLORS["green"]),
        (M + 560, "contradictory support", COLORS["red"]),
        (M + 980, "heuristic unclear", COLORS["gray"]),
    ]:
        draw.ellipse((x, legend_y, x + 30, legend_y + 30), fill=color)
        text(draw, (x + 44, legend_y - 2), label, F_SMALL_B, COLORS["muted"])

    # Anomaly panel
    anomaly = (M, 2160, W - M, 3295)
    rounded(draw, anomaly, 36, COLORS["panel"], COLORS["line"], 4)
    text(draw, (M + 52, 2215), "2. Misleading-question anomaly", F_H2)
    text(draw, (M + 52, 2285), "A qualitative stress case for ambiguity and insufficient image grounding.", F_BODY, COLORS["muted"])

    img_box = (M + 60, 2370, M + 1880, 2760)
    rounded(draw, img_box, 24, "#FFFFFF", COLORS["line"], 3)
    inset = fit_image(SOURCE_408, (img_box[0] + 24, img_box[1] + 28, img_box[2] - 24, img_box[3] - 28), crop=False)
    canvas.paste(inset, (img_box[0] + 24, img_box[1] + 28))
    text(draw, (img_box[0] + 22, img_box[3] + 28), "source_index 408 image context", F_SMALL_B, COLORS["muted"])

    # Flow boxes
    bx = M + 2000
    by = 2370
    bw = 820
    rounded(draw, (bx, by, bx + bw, by + 215), 26, COLORS["amber_bg"], COLORS["amber_line"], 4)
    text(draw, (bx + 34, by + 30), "Question prior", F_H3, COLORS["ink"])
    wrapped(draw, (bx + 34, by + 88), "Modified prompt: pick up the object. The target is underspecified.", F_BODY, bw - 70, COLORS["muted"], 6)

    arrow_x = bx + bw + 58
    draw.line((arrow_x, by + 108, arrow_x + 190, by + 108), fill=COLORS["subtle"], width=8)
    draw.polygon([(arrow_x + 190, by + 108), (arrow_x + 155, by + 88), (arrow_x + 155, by + 128)], fill=COLORS["subtle"])

    bx2 = bx + bw + 260
    rounded(draw, (bx2, by, bx2 + bw, by + 215), 26, COLORS["pink_bg"], COLORS["pink_line"], 4)
    text(draw, (bx2 + 34, by + 30), "Model explanation", F_H3, COLORS["ink"])
    wrapped(draw, (bx2 + 34, by + 88), "Assumes frying pan as the target and justifies the red arrow as motion toward it.", F_BODY, bw - 70, COLORS["muted"], 6)

    rounded(draw, (bx, by + 290, bx + 2 * bw + 260, by + 560), 26, COLORS["green_bg"], COLORS["green_line"], 4)
    text(draw, (bx + 34, by + 324), "Interpretation", F_H3, COLORS["teal"])
    wrapped(
        draw,
        (bx + 34, by + 390),
        "The failure mode is not merely an incorrect answer. The explanation can be coherent while being under-grounded: it follows a plausible linguistic prior instead of demonstrating that the target and arrow are visually verified in the scene.",
        F_BODY_B,
        2 * bw + 190,
        COLORS["teal"],
        7,
    )

    # Bottom thesis
    rounded(draw, (M + 60, 3046, W - M - 60, 3226), 28, "#0F172A", None, 0)
    wrapped(
        draw,
        (M + 100, 3090),
        "Takeaway: accuracy, control robustness, and explanation grounding are separable axes. Sanity evaluations are most useful when they reveal which axis fails.",
        F_H3,
        W - 2 * M - 200,
        "#FFFFFF",
        7,
    )

    text(draw, (M, 3375), "Generated from local sanity artifacts. Counts are heuristic labels over 20 selected examples per probe type.", F_SMALL, COLORS["subtle"])
    text(draw, (W - M, 3375), "300 dpi A3 landscape PDF", F_SMALL_B, COLORS["subtle"], anchor="ra")

    canvas.save(OUT_PNG)
    canvas.save(OUT_PDF, "PDF", resolution=300.0)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
