# Current Pure RL V9 Review Snapshot

## Customer-Only LLM Integration Update

The separate [v9 guided adapter](experiments/concurrent_fleet_v9_guided/README.md)
now includes the [concurrent ontology](experiments/concurrent_fleet_v9_guided/ONTOLOGY.md),
separate train/test prompt builders and [online one-shot evaluation](experiments/concurrent_fleet_v9_guided/ONLINE.md).
Accident disclosure interrupts the simulation clock to refresh confidence before
the next routing decision. Depot, NEW and STOP receive no direct confidence bias.
API latency pauses the simulator; paid calls require explicit opt-in. Tests use
mock responses only. This addition does not change the active pure-RL source.
The deferred guidance notes below describe the original pure-RL snapshot stage.

    python -m unittest experiments.concurrent_fleet_v9_guided.test_guidance experiments.concurrent_fleet_v9_guided.test_ontology experiments.concurrent_fleet_v9_guided.test_prompts experiments.concurrent_fleet_v9_guided.test_online -v

This is the October 19 diagnostic-plan implementation. V8 is preserved separately.
All 15 source files recorded by the vehicle200 protocol match their SHA256.
Dependencies, configs and train20/test80 data are included; no checkpoints,
API keys or bulk training logs are published.

## Changes

- Shared customer/report sector embeddings; one token per public report.
- Learned STOP token, shared decoder keys, fleet-conditioned dispatch query.
- STOP capacity mask and weak depot mask with fleet-limit/no-fit exceptions.
- Explicit dispatch closure, including zero-duration transitions.
- Return-completion accounting, slack clamps, AMP and full-graph backward retained.
- Batch128 x rollout100, 2500 episodes/epoch, seed42 and 1000 epochs unchanged.
- No actual multiplier or hidden future arrival time enters policy input.
- Clamp removal and guidance-adapter refactoring are deferred: pure RL only.

## Active Cost Experiment

Baseline v9 (121.21 minutes/vehicle) was paused at update1471 during epoch74.
The 200-minute run starts fresh, not from that checkpoint:

    python -m experiments.concurrent_fleet_v9.train_vehicle200 --out runs/v9_200 --weights 10

Run from THIS snapshot directory. Objective:
J = (200*K + T + 10*L)/1696.95 + 500*U/N.
The ordinary train.py retains the baseline vehicle cost.
Original protocols under result/ are provenance, not new output destinations.
Creating STOP in an output directory requests checkpointed termination.

Log reports distinguish rollout mean, best-of-100 and fixed TRAIN greedy probes.
Report reward, fleet count, travel time, total lateness (including depot),
late-visit count and unserved. Compare different weights with common-cost rescores.
No completed test80 performance is claimed.

## Verification

    python verify_v9.py
    python -m unittest experiments.concurrent_fleet_v9.test_v9 -v

Tests cover capacity masks, fleet-limit exception, zero-duration closure, whole
sector reports, pre-disclosure invariance and full-graph gradients. They do not
establish exhaustive scalar/tensor parity or final model quality.
The inherited verify_snapshot.py checks historical v8 inputs only.
