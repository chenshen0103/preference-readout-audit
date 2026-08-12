"""Biosafety renderer (SPEC §35). The renderer is structurally incapable of
emitting operational biological content: it only interpolates abstract labels
from configs/domains.yaml and numeric payoffs. The blocklist test (T3) guards
against regression in the label/context configuration."""
import re

from .base import DomainRenderer

# Terms that must never appear in any biosafety-domain rendering or template.
FORBIDDEN_TERMS = [
    "virus", "viral", "pathogen", "bacteri", "genome", "sequence", "protocol",
    "synthes", "gain-of-function", "gain of function", "plasmid", "crispr",
    "strain", "culture", "anthrax", "smallpox", "ebola", "influenza", "toxin",
    "infectious", "dna", "rna", "wet-lab", "wet lab", "laboratory", "specimen",
    "aerosol", "transmissib",
]


def contains_forbidden(text: str) -> list[str]:
    """Return forbidden terms found in text. Terms match at word starts only
    (\\b prefix, no trailing boundary), so stems like 'bacteri' catch
    'bacteria' while 'rna' does not false-positive inside 'external'."""
    low = text.lower()
    return [t for t in FORBIDDEN_TERMS if re.search(r"\b" + re.escape(t), low)]


def renderer() -> DomainRenderer:
    return DomainRenderer("biosafety")
