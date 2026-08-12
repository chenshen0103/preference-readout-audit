from .base import ModelAdapter  # noqa: F401


def build_adapter(cfg: dict) -> ModelAdapter:
    kind = cfg["adapter"]
    if kind == "mock":
        from .mock_adapter import MockModelAdapter
        return MockModelAdapter(alpha=cfg.get("alpha", 1.0), beta=cfg.get("beta", 2.0),
                                scale=cfg.get("scale", 10.0), seed=cfg.get("seed", 0))
    if kind == "hf":
        from .hf_adapter import HFModelAdapter
        return HFModelAdapter(hf_id=cfg["hf_id"], revision=cfg.get("revision", "main"),
                              dtype=cfg.get("dtype", "float16"),
                              device_map=cfg.get("device_map", "auto"))
    if kind == "openai":
        from .api_adapter import OpenAIAdapter
        return OpenAIAdapter(model=cfg["model"])
    if kind == "anthropic":
        from .api_adapter import AnthropicAdapter
        return AnthropicAdapter(model=cfg["model"])
    raise KeyError(f"unknown adapter kind: {kind}")
