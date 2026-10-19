# Concurrent fleet v9

Fresh SELECT-on experiment. V8 and Pair runs are preserved.

- Whole-sector customer/report embeddings; one token per disclosed report.
- Learned STOP_DISPATCH token with shared decoder keys and phase-specific query.
- Dispatch context includes fleet count, free demand, operating residual capacity,
  and free-customer fraction. No hidden multiplier or future arrival is observed.
- STOP requires enough aggregate capacity, except at the fleet limit.
- Depot return requires enough other capacity, except when no free customer fits.
- Explicit dispatch closure also handles zero-duration transitions.
- Existing slack clamps, terminal reward, full-graph REINFORCE, optimizer and
  training budget are unchanged. Vehicle cost is NOT increased to 200 minutes.
- This is a combined architecture/mask experiment, not a single-factor ablation.
- Guidance adapter changes are deferred; this run is pure RL only.

Run: `python -m experiments.concurrent_fleet_v9.train --out RESULT --weights 10`
Stop: create `RESULT/STOP`; checkpoint is saved after the current update.
Diagnostics: training.jsonl every batch, early_history for first 20 updates and
every 20 thereafter, fixed TRAIN greedy probes in checks. Final test80 is one-shot.

Action slots retain the existing tensor layout: 0..N-1 NEW, N..2N routing
destinations (including depot), 2N+1 STOP_DISPATCH. Event tokens are never actions.
