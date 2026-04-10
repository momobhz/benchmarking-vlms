import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_robo2vlm_question_clusters import (
    default_output_html,
    load_projection_rows,
)


def test_default_output_html_uses_interactive_suffix():
    input_csv = Path("outputs/test_model_umap.csv")
    assert default_output_html(input_csv) == Path("outputs/test_model_umap_interactive.html")


def test_load_projection_rows_requires_expected_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("source_index,id,label\n0,a,neither\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_projection_rows(csv_path)


def test_load_projection_rows_parses_types(tmp_path):
    csv_path = tmp_path / "good.csv"
    csv_path.write_text(
        (
            "source_index,id,label,question,umap_x,umap_y\n"
            "7,abc,spatial_reasoning,Where is the mug?,1.25,-2.5\n"
        ),
        encoding="utf-8",
    )

    rows = load_projection_rows(csv_path)

    assert rows == [
        {
            "source_index": 7,
            "id": "abc",
            "label": "spatial_reasoning",
            "question": "Where is the mug?",
            "umap_x": 1.25,
            "umap_y": -2.5,
        }
    ]
