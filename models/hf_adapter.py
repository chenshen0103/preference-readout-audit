"""HuggingFace adapter for open-weight models on the DGX (SPEC §19-§23, DR-6/DR-7).

V100 note: Volta has no bf16 support — use float16 (Qwen2.5 is fp16-safe;
avoid Gemma in fp16). Requires: torch, transformers, accelerate.
"""
from __future__ import annotations

import math

from .base import ModelAdapter


class HFModelAdapter(ModelAdapter):
    def __init__(self, hf_id: str, revision: str = "main", dtype: str = "float16",
                 device_map: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.hf_id = hf_id
        self.revision = revision
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id, revision=revision,
            torch_dtype=getattr(torch, dtype), device_map=device_map)
        self.model.eval()
        self.dtype = dtype
        # Resolve the exact commit hash for reproducibility metadata (§16).
        self.commit_hash = getattr(self.model.config, "_commit_hash", None)

    def _first_token_ids(self, label: str) -> list[int]:
        """First-token ids for the label variants with/without leading space (DR-7)."""
        ids = set()
        for variant in (label, " " + label):
            toks = self.tokenizer.encode(variant, add_special_tokens=False)
            if toks:
                ids.add(toks[0])
        return sorted(ids)

    def _answer_logits(self, system: str, user: str):
        torch = self._torch
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits[0, -1, :]
        return torch.log_softmax(logits.float(), dim=-1)

    def score_choices(self, system: str, user: str,
                      choices: tuple[str, ...] = ("A", "B")) -> dict[str, float]:
        torch = self._torch
        logprobs = self._answer_logits(system, user)
        raw = {}
        for label in choices:
            ids = self._first_token_ids(label)
            raw[label] = torch.logsumexp(logprobs[ids], dim=0).item()
        # Normalize over the choice set.
        z = max(raw.values())
        total = math.log(sum(math.exp(v - z) for v in raw.values())) + z
        return {label: v - total for label, v in raw.items()}

    def generate(self, system: str, user: str, max_tokens: int = 256,
                 temperature: float = 0.0, seed: int | None = None) -> str:
        torch = self._torch
        if seed is not None:
            torch.manual_seed(seed)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        kwargs = dict(max_new_tokens=max_tokens,
                      pad_token_id=self.tokenizer.eos_token_id)
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature)
        else:
            kwargs.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(input_ids, **kwargs)
        return self.tokenizer.decode(out[0, input_ids.shape[1]:],
                                     skip_special_tokens=True)

    def get_metadata(self) -> dict:
        return {"name": self.hf_id, "revision": self.commit_hash or self.revision,
                "quantization": "none", "precision": self.dtype,
                "device": str(self.model.device), "model_type": "hf_causal_lm"}
