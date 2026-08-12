"""Strict structured-output parsing for secondary elicitation methods.
Malformed output is FLAGGED, never coerced (SPEC test T7, §46)."""
from __future__ import annotations

import json
import re


def parse_json_number(raw: str, key: str,
                      lo: float | None = None,
                      hi: float | None = None) -> tuple[float | None, str]:
    """Return (value, status). status == 'ok' only for a clean parse."""
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return None, "parse_error: no JSON object found"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, f"parse_error: invalid JSON ({e.msg})"
    if not isinstance(obj, dict) or key not in obj:
        return None, f"parse_error: missing key '{key}'"
    v = obj[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None, f"parse_error: '{key}' not numeric"
    if lo is not None and v < lo:
        return None, f"parse_error: '{key}'={v} below {lo}"
    if hi is not None and v > hi:
        return None, f"parse_error: '{key}'={v} above {hi}"
    return float(v), "ok"


def parse_forced_choice_text(raw: str) -> tuple[str | None, str]:
    """For generate-based forced choice (API fallback). Accepts only an
    unambiguous leading A/B token."""
    text = raw.strip()
    m = re.match(r"^([AB])\b", text)
    if not m:
        return None, "parse_error: no unambiguous A/B"
    return m.group(1), "ok"
