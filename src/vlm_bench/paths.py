from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root when running from an editable checkout."""
    return Path(__file__).resolve().parents[2]


def default_eval_script() -> Path:
    return repo_root() / "benchmark" / "evaluation.py"


def default_script(name: str) -> Path:
    return repo_root() / "scripts" / name
