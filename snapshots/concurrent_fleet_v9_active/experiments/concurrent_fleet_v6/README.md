# Concurrent Fleet V6

This isolated revision replaces v5's reuse of the depot-return logit for
SELECT_EXISTING with an independent dispatch-stop score. It does not change
the environment, objective, action masks, routing decoder, or training budget.

The score is a zero-initialized linear head (four weights and one bias),
followed by the same tanh/logit-clipping scale as routing scores. Inputs are
dispatched vehicles / fleet limit, unassigned demand / fleet limit, active
vehicles' residual capacity / fleet limit, and unassigned customer fraction.
These are observable dispatch statistics, not future event outcomes.

V5 checkpoints are not resume-compatible with v6. Use a new output directory.
The active v5 experiment is not restarted or modified by this revision.

```powershell
python -m unittest experiments.concurrent_fleet_v6.test_policy
python -m experiments.concurrent_fleet_v6.train --out result/concurrent_fleet_v6_train20 --weights 5 10
```

The training command is intentionally not launched while v5 is running.
Performance optimizations and reward or feasibility-mask changes are deferred.
