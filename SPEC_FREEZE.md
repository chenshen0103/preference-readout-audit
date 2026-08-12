# SPEC_FREEZE.md

STATUS: **NOT FROZEN** — main experiment MUST NOT begin (SPEC §17).

Freeze checklist (complete at Day 1, after pilot calibration / OQ-8):

- [ ] All items in OPEN_QUESTIONS.md resolved or explicitly deferred
- [ ] Scenario generator frozen; `scenarios/frozen/{final,pilot}.json` committed
- [ ] Prompt templates frozen (face-validity review per SPEC §34 recorded below)
- [ ] Model checkpoint pinned to exact commit hash in configs/models.yaml
- [ ] Primary endpoint frozen: M1 cross-domain switch-point invariance (DR-3)
- [ ] Metrics M1-M5 formulas frozen (DECISIONS.md DR-3/DR-4)
- [ ] Parser frozen
- [ ] All tests green (including V1 planted-preference recovery)

At freeze, record:

```
date:
git_commit:
scenario_count: 40 final / 10 pilot
model_versions:
primary_metric: M1 (DR-3)
planned_statistical_analysis: logistic regression, cluster-robust SE by scenario (DR-9)
face_validity_reviewers:
```

Any change after freeze requires a version bump of `experiment_version`
and an entry in CHANGELOG.md.
