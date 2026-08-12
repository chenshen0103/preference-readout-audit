"""Generate the face-validity review sheet (SPEC §34): every scenario rendered
in every domain, for human inspection BEFORE SPEC_FREEZE.

Usage (from repo root):
    python scripts/make_review_sheet.py > report/review_sheet.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renderers import get_renderer  # noqa: E402
from scenarios.generator import final_set  # noqa: E402

DOMAINS = ("generic", "finance", "biosafety")

CHECKLIST = """# Face-Validity Review Sheet (SPEC §34)

For each scenario below, verify:
1. Finance and Biosafety preserve the SAME mathematical trade-off as Generic.
2. No renderer adds emotional or moral language.
3. Units are comparable across options.
4. No wording makes one option obviously socially preferred.

Record findings in OPEN_QUESTIONS.md or approve in SPEC_FREEZE.md.

---
"""


def main() -> None:
    out = [CHECKLIST]
    for s in final_set():
        out.append(f"## {s.scenario_id} (family={s.family}, stress={s.stress_level})\n")
        for domain in DOMAINS:
            r = get_renderer(domain).render(s)
            out.append(f"### {domain}\n\n```text\n{r.text}\n```\n")
        out.append("\n---\n")
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
