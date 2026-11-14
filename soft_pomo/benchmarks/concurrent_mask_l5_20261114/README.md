# Concurrent routing comparison archive (2026-11-14)

Run from `soft_pomo`:

```powershell
python benchmarks/concurrent_mask_l5_20261114/report.py
```

This is a curated **Test80** archive for the current cross-PC experiments, not an archive of every historical experiment. All RL runs use simultaneous routing, vehicle weight 100 or 200, late penalty lambda=5, return mask ON, and epoch1000 checkpoints.

- Pure RL: C101, R101, RC101; weights100/200; seeds42?46 (30 runs).
- Kim RL-example (legacy dispatch): R101 weight100 seeds42?46; RC101 weight100 seeds42?45; RC101 weight200 seed42. This is the earlier New Kim with a fixed RL example, WITHOUT NEW-head dispatch bias.
- Kim NEW-head bias: RC101 weight100 seed42, completed Test80. This is the current treatment; do not pool it with legacy Kim.
- LKH-3: each case, scored at both weights on the identical Test80 customer-instance names. One solution set, NOT five seeds. Reward is recomputed using the matching RL reward denominator. Solver optimization is not claimed to optimize this weighted RL objective.
- Missing settings are absent, NOT zero. C concurrent Kim results and other unfinished/unverified evaluations are not invented. Sequential Kim is deliberately excluded.

`runs.json` includes per-instance reward/vehicle count/travel time/lateness, run means, source paths and SHA256, seed and method labels. Pure RL configs are included. These sources are archival provenance paths on the original PC; the report works without those files. No checkpoints, API credentials, request/response caches or full replay traces are needed.

Compare only identical case, weight, lambda, mask, Test80 names and decode protocol. Report includes actual seeds; RC legacy Kim n=4 and NEW-head n=1 are not five-seed comparisons. CI uses sample SD (ddof=1), 1.96*SD/sqrt(n), over seed-level Test80 means. This is a normal-approximation CI, not a Student-t CI. Reward is negative (higher is better); positive cost = -reward. All printed metrics use two decimals.

LKH sources: RC pilot-history Test80 (alpha1); C strength200 Test80 (alpha2); R strength050 history Test80 (alpha0.5). Correction: the initial export mistakenly used C alpha1 LKH results; this version uses C alpha2 matching the target dataset. Instance-name equality with Pure RL is checked on export. C/R portable examples can have different source routing/weights: reference-example provenance does not redefine the target evaluation setting.

To compare new-PC results, calculate the mean over exactly these 80 instance names for each seed first; then aggregate seeds. Do not compare training-log averages with these Test results. Regenerate guidance causally at ACC disclosure when required; do not reuse another policy's state-dependent ACC response.
