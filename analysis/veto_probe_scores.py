#!/usr/bin/env python3
"""Provenance script: how analysis/veto/probe_scores.npy was produced.

Trains per-layer logistic probes (L32-57) on the validity-set residuals
(192 traces; the probe validated for cross-scenario, cross-label transfer)
and applies them, frozen, to the veto-batch residuals; scores are z-scaled
per layer and sign-adjusted so positive = PROFIT semantically.

Inputs analysis/{validity,veto}/residuals.npy are gitignored (30-120 MB);
regenerate them first with validity_diagnostics.py and veto_tests.py.
Originally run inline on 2026-08-15; boundary/probe_scores.npy is the
analogous output of boundary_probe.py.
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

Xv = np.load("analysis/validity/residuals.npy").astype(np.float32)
mv = json.load(open("analysis/validity/meta.json"))
yv = np.array([(m["semantic_choice"] == "x") == (m["order"] == "orig")
               for m in mv], dtype=int)
X = np.load("analysis/veto/residuals.npy").astype(np.float32)
meta = json.load(open("analysis/veto/meta.json"))
order = np.array([m["order"] for m in meta])

S = np.full((len(meta), 60), np.nan)
for l in range(32, 58):
    sc = StandardScaler().fit(Xv[:, l, :])
    clf = LogisticRegression(C=0.01, max_iter=2000).fit(
        sc.transform(Xv[:, l, :]), yv)
    raw = clf.decision_function(sc.transform(X[:, l, :]))
    raw = raw / (np.std(raw) + 1e-9)
    S[:, l] = np.where(order == "orig", raw, -raw)
np.save("analysis/veto/probe_scores.npy", S)
print("saved analysis/veto/probe_scores.npy", S.shape)
