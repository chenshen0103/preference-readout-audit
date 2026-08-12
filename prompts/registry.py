"""Prompt template loading. Templates are frozen files; IDs are filenames.
Drafted before any model outputs were observed (SPEC §46; OQ-3)."""
from __future__ import annotations

from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parent

FAMILY_DIRS = {
    "forced_choice": "forced_choice",
    "rating": "rating",
    "stated_preference": "stated_preference",
}


def load_template(template_id: str) -> str:
    for sub in FAMILY_DIRS.values():
        p = _PROMPT_DIR / sub / f"{template_id}.txt"
        if p.exists():
            return p.read_text(encoding="utf-8")
    raise FileNotFoundError(f"prompt template not found: {template_id}")


def list_templates(family: str) -> list[str]:
    d = _PROMPT_DIR / FAMILY_DIRS[family]
    return sorted(p.stem for p in d.glob("*.txt"))
