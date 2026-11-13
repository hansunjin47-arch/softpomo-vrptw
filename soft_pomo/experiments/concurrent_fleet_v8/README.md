# Concurrent Fleet V8

Fresh-run performance revision of v7. Environment semantics, SELECT head,
terminal objective, REINFORCE baseline, AMP, encoder batch, rollout count,
and full-graph single-backward training are unchanged. No reward-to-go.

## Changes

- Cache done/next-vehicle/mask while the fleet state is unchanged. Invalidate
  at both transition-processing phases; preserve both `_process` calls.
- Cache attention K/V projections per event encoding and autocast mode for
  one batch. Clear the cache at every encode_events call. No detach and no
  sharing of encoder inputs across the batch dimension.
- Save checkpoints every 20 updates (one epoch) and on STOP/smoke completion.
  Unplanned interruption can lose up to 19 completed updates. Existing
  per-batch metrics and early/fixed probe logging remain enabled.

Mixed event stages still use v7's stage-selection computation. torch.compile
and new reward or action-masking rules were not introduced.

## Verification

`python -m unittest experiments.concurrent_fleet_v8.test_equivalence -v`

Eight tests passed: five hand-calculated/fixed-route checks (including all
20 real training scenarios), and CPU, CUDA FP32, CUDA AMP comparisons of
state transitions, probabilities, and gradients. Optimizer updates are
compared in FP32. CUDA AMP uses tolerance because accumulation order changes.

`python -m analysis.benchmark_fleet_v8 --version v7` (then v8 separately)

Six batch128 x rollout100 updates from the same v7 checkpoint, each version
in a new process: rewards matched across the six batches. Excluding the first
batch, average elapsed times were about 0.933 s (v7) and 0.778 s (v8).
Maximum allocated memory over these batches: 19012 MiB vs 17982 MiB.
This short benchmark does not establish steady-state memory or throughput.

A separate eight-update full-size fresh initialization smoke run was used
before launching the real experiment. Its checkpoint is not used for training.

## Fresh Training

```powershell
python -m experiments.concurrent_fleet_v8.train --out result/concurrent_fleet_v8_fresh_train20_20261018 --weights 5 10 1 20
```

Each lambda starts from its configured seed and runs 1000 epochs followed by
80 single-rollout test evaluations. The four settings share one sequential
runner and one GPU, not separate overlapping jobs. Prior v7 runs are preserved.
