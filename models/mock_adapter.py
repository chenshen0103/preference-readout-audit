"""Mock adapter with a PLANTED utility function, for end-to-end pipeline
validation (test V1, planted-preference recovery).

The mock decides from the RENDERED PROMPT TEXT ONLY — it re-parses the
benefit/cost numbers out of the prompt using the domain label lexicon. If any
stage (generator -> renderer -> runner -> parser -> fit) distorts the numbers
or the A/B mapping, recovery of the planted alpha/beta fails.

U(option) = alpha * benefit - beta * cost
P(A) = sigmoid((U_A - U_B + position_bias) / scale)
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import yaml

from .base import ModelAdapter

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _label_lexicon() -> tuple[set[str], set[str]]:
    domains = yaml.safe_load((_CONFIG_DIR / "domains.yaml").read_text(encoding="utf-8"))
    benefit = {d["benefit_label"] for d in domains.values()}
    cost = {d["cost_label"] for d in domains.values()}
    return benefit, cost


class MockModelAdapter(ModelAdapter):
    def __init__(self, alpha: float = 1.0, beta: float = 2.0, scale: float = 10.0,
                 position_bias: float = 0.0, seed: int = 0):
        self.alpha = alpha
        self.beta = beta
        self.scale = scale
        self.position_bias = position_bias
        self.seed = seed
        self._benefit_labels, self._cost_labels = _label_lexicon()

    def _parse_option(self, block: str) -> tuple[float, float]:
        benefit = cost = None
        for line in block.splitlines():
            m = re.match(r"\s*(.+?):\s*(-?\d+(?:\.\d+)?)\s*$", line)
            if not m:
                continue
            label, value = m.group(1).strip(), float(m.group(2))
            if label in self._benefit_labels:
                benefit = value
            elif label in self._cost_labels:
                cost = value
        if benefit is None or cost is None:
            raise ValueError(f"mock could not parse option block:\n{block}")
        return benefit, cost

    def _utilities(self, user: str) -> dict[str, float]:
        blocks = re.split(r"^Option ([AB]):\s*$", user, flags=re.MULTILINE)
        # re.split yields [prefix, 'A', blockA, 'B', blockB(+suffix)]
        if len(blocks) < 5:
            raise ValueError("mock expected two 'Option A/B:' blocks in prompt")
        out = {}
        for label, block in ((blocks[1], blocks[2]), (blocks[3], blocks[4])):
            benefit, cost = self._parse_option(block)
            out[label] = self.alpha * benefit - self.beta * cost
        return out

    def score_choices(self, system: str, user: str,
                      choices: tuple[str, ...] = ("A", "B")) -> dict[str, float]:
        u = self._utilities(user)
        z = (u["A"] - u["B"] + self.position_bias) / self.scale
        p_a = 1.0 / (1.0 + math.exp(-z))
        p_a = min(max(p_a, 1e-12), 1 - 1e-12)
        return {"A": math.log(p_a), "B": math.log(1.0 - p_a)}

    def generate(self, system: str, user: str, max_tokens: int = 256,
                 temperature: float = 0.0, seed: int | None = None) -> str:
        # Deterministic structured outputs for secondary-method plumbing tests.
        if "acceptable_cost_per_benefit_unit" in user:
            return f'{{"acceptable_cost_per_benefit_unit": {self.beta / self.alpha}}}'
        if '"score"' in user:
            benefit, cost = self._parse_option(user)
            raw = self.alpha * benefit - self.beta * cost
            score = min(max(50 + raw / 2, 0), 100)
            return f'{{"score": {score}}}'
        return "A"

    def get_metadata(self) -> dict:
        return {"name": "mock", "revision": f"alpha={self.alpha},beta={self.beta}",
                "quantization": "none", "precision": "float64", "device": "cpu",
                "model_type": "mock"}
