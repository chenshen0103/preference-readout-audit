"""Closed-API adapters — SECONDARY behavioral replication only (SPEC §21).

OpenAI: chat completions expose token logprobs -> approximate choice scoring.
Anthropic: no logprobs -> structured-output tier only (SPEC §25 'acceptable').
Keys via environment variables OPENAI_API_KEY / ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import math
import os

from .base import ModelAdapter


class OpenAIAdapter(ModelAdapter):
    def __init__(self, model: str):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def score_choices(self, system, user, choices=("A", "B")):
        resp = self.client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=1, logprobs=True,
            top_logprobs=20,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        top = resp.choices[0].logprobs.content[0].top_logprobs
        raw = {}
        for label in choices:
            variants = [t.logprob for t in top if t.token.strip() == label]
            raw[label] = (math.log(sum(math.exp(v) for v in variants))
                          if variants else -30.0)
        z = max(raw.values())
        total = math.log(sum(math.exp(v - z) for v in raw.values())) + z
        return {label: v - total for label, v in raw.items()}

    def generate(self, system, user, max_tokens=256, temperature=0.0, seed=None):
        resp = self.client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=max_tokens,
            seed=seed,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        return resp.choices[0].message.content

    def get_metadata(self):
        return {"name": self.model, "revision": self.model, "quantization": "n/a",
                "precision": "n/a", "device": "api", "model_type": "openai_api"}


class AnthropicAdapter(ModelAdapter):
    def __init__(self, model: str):
        import anthropic
        self.model = model
        self.client = anthropic.Anthropic()

    def score_choices(self, system, user, choices=("A", "B")):
        # No logprobs available: degenerate 0/1 scoring from the generated label.
        # This is the SPEC §25 'acceptable' tier, flagged in metadata.
        text = self.generate(system, user, max_tokens=3, temperature=0.0).strip()
        out = {}
        for label in choices:
            out[label] = math.log(1 - 1e-9) if text.startswith(label) else math.log(1e-9)
        return out

    def generate(self, system, user, max_tokens=256, temperature=0.0, seed=None):
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            system=system, messages=[{"role": "user", "content": user}])
        return resp.content[0].text

    def get_metadata(self):
        return {"name": self.model, "revision": self.model, "quantization": "n/a",
                "precision": "n/a", "device": "api",
                "model_type": "anthropic_api_no_logprobs"}
