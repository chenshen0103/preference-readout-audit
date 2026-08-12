"""Model adapter interface (SPEC §22)."""
from __future__ import annotations

from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    @abstractmethod
    def score_choices(self, system: str, user: str,
                      choices: tuple[str, ...] = ("A", "B")) -> dict[str, float]:
        """Return log-probabilities (natural log, normalized over `choices`)
        for each choice label at the constrained answer position (DR-7)."""

    @abstractmethod
    def generate(self, system: str, user: str, max_tokens: int = 256,
                 temperature: float = 0.0, seed: int | None = None) -> str:
        """Free-form generation, used by secondary elicitation methods."""

    @abstractmethod
    def get_metadata(self) -> dict:
        """Reproducibility metadata (SPEC §16): name, revision, dtype, device..."""
