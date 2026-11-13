# Vehicle/Destination Pair Variant

Separate experiment; all SELECT v7/v8 source, checkpoints and results are retained.
No explicit STOP/SELECT output and no change to reward weights or fleet limit.

## Actions And Time

- Initial action: `(NEW, customer)` or `(existing vehicle, destination)`.
- NEW immediately dispatches a vehicle from the depot at time zero. With no
  vehicles only NEW is valid. The existing alternative includes depot return.
- Choosing an existing pair closes dispatch and commits that vehicle's next leg.
  Its customer is reserved immediately and capacity is charged once. The current
  leg/service is unchanged. The pending leg departs automatically after service.
- The event loop handles all other vehicles in chronological order, including
  pending departures before the next policy decision. Actual travel time uses
  the multiplier at actual departure, never retimes an already departed arc.
- At the vehicle limit or with no unassigned customers dispatch closes automatically.
- After dispatch, the environment selects a ready vehicle (ID order for ties),
  and the policy selects only its destination. Dispatch stays closed even in
  zero-duration fixtures where time remains zero.

## Policy And Training

One shared decoder scores vehicle/destination pairs, followed by a joint softmax.
Existing queries during dispatch use the committed current destination and remaining
capacity. No arrival/available timestamp, hidden multiplier or future event stage
is exposed. Public events are gated by the actual global decision clock.
For a future committed leg, both arrival-derived slack channels are zero (unknown);
the base travel feature remains. Ordinary ready-vehicle decoding matches v8.
The five SELECT parameters are removed; shared encoder/decoder initialization is
unchanged. The number of existing pairs grows with deployed vehicles, so initial
stop probability differs structurally from a single SELECT alternative.

Training remains batch128, 100 sampled rollouts, 2500 episodes/epoch, seed42,
full-graph REINFORCE with per-episode mean/std baseline, Adam, AMP/GradScaler,
gradient clipping=1, 1000 epochs, then test80 one-shot greedy. Only lambda5 is
started for this comparison. The smoke checkpoint is not reused.
State/KV caching remains enabled. Queries for wholly invalid action groups are
not decoded. Different action space implies different sampling and learning paths.
Valid queries are packed within each encoder batch (encoder batch is NOT shared).
Decoder attention uses PyTorch SDPA with the same scale and masks, no dropout.
This avoids retaining a dense attention matrix for every padded vehicle query.
FP32/AMP probability and gradient parity are tested within numerical tolerance;
floating-point trajectories need not be bitwise identical. No rollout reduction,
gradient accumulation, detached replay or activation checkpointing is used.

Status/checkpoint replacement retries transient reader locks for up to ~2 seconds.
Resume archives and reconciles logs beyond the saved checkpoint update.
The original v8 writer is not changed. Persistent I/O errors remain fatal.
Epoch reward remains best-of-100 for continuity; compare `training.jsonl` reward
and other metrics at matched update ranges as well, not only this curve.

```text
python -m unittest experiments.concurrent_fleet_pair.test_pair -v
python -m experiments.concurrent_fleet_pair.train --out result/UNIQUE_PAIR_RUN --weights 5
```

Early history columns: `time, vehicle, destination, new_vehicle, live`.
Vehicle indices are zero-based; customer/depot IDs follow the original convention.
Model or environment changes must use a new output/protocol, not resume v8.
