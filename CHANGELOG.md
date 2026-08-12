# CHANGELOG

## 0.1.0-prefreeze — 2026-08-12
- Initial scaffold: schema, deterministic scenario generator (40 final / 10 pilot),
  3 domain renderers, 3+2 prompt templates, mock/HF/API model adapters,
  resumable batch runner, metrics M1-M5, diagnostics SC1-SC4, primary figure.
- Full test suite (T1-T10 + V1 planted-preference recovery): 24 tests passing.
- Sweep grid calibrated for beta/alpha in [0.5, 2] (see OQ-8 for pilot recalibration).
