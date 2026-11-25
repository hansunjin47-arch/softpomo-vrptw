# Concurrent routing: version and execution map

Snapshot: 2026-11-25. Run commands from `soft_pomo/` with Python 3.12 and a
CUDA-compatible PyTorch installation. `requirements-code.txt` lists dependencies.
Use a separate checkout when another process is training; changing files does not
update code already imported by that process.

## Which version is which?

| ID | Implementation | Status / purpose |
|---|---|---|
| concurrent-base | `experiments/concurrent_fleet_pair/policy.py` | Original concurrent Pure RL comparator |
| concurrent-kim | `experiments/concurrent_fleet_pair_kim/run.py` | Anchored routing confidence; no NEW-head bias |
| concurrent-newhead | `experiments/concurrent_fleet_pair_kim/confidence_head_guidance.py` | Proposed: fixed highest-confidence NEW head + subsequent group confidence |
| fleet-pure | `experiments/concurrent_fleet_pair/fleet_context.py` | Stage 1 Pure RL: public fleet attention in current query |
| fleet-proposed | `experiments/concurrent_fleet_pair_kim/fleet_context.py` | Stage 1 + NEW-head / routing confidence; current local research version |
| customer-graph | `experiments/concurrent_fleet_pair/customer_graph.py` | Stage 2 prototype; vehicle-to-customer message residual, not a validated full heterogeneous GNN |

`v1`...`v9` directories are historical implementations AND inherited dependencies.
Do not delete them or select the highest version number as the current entry point.
The sequential/old two-phase workflow in the legacy README is a different protocol.

## Shared action semantics

Dispatch occurs only at time zero. NEW chooses a first customer and creates a
vehicle; choosing an existing-vehicle action closes dispatch (subject to capacity
mask). After dispatch, the simulator selects the next vehicle by event timing and
the policy chooses its destination. Already committed customers are masked.
Return protection and minimum dispatch capacity are distinct constraints.

NEW heads are each group's highest-confidence customer, not `group[0]`. Ties use
group order. Once selected/masked, there is no runner-up promotion. The 2026-11-25
optimization computes heads when confidence is loaded, rather than per decision;
it does not change the bias semantics. Later accident guidance affects routing.
The CPU regression matched probabilities and gradients across a full trajectory.

## Portable commands: baseline / existing Proposed

```powershell
python -m experiments.concurrent_fleet_pair_kim.portable validate --case rc101
python -m experiments.concurrent_fleet_pair_kim.portable train --case rc101 --method pure --seed 46 --vehicle-cost 100 --dispatch-capacity-mask --lr-decay
# Proposed requires an appropriate generated confidence artifact:
python -m experiments.concurrent_fleet_pair_kim.portable train --case rc101 --method newhead --seed 46 --vehicle-cost 100 --dispatch-capacity-mask --lr-decay --artifact PATH_TO_CACHE
```

These are the existing baseline portable commands, not fleet-context commands.
`pure`, `kim`, `newhead` select different treatments. Inspect cache provenance:
reference type (RL only versus RL+LKH), objective, prompt version and groups matter.
Do not assume these defaults recreate every dated experiment.

## Portable commands: stage 1 / stage 2 Pure RL

```powershell
python -m analysis.train_fleet_context_portable --model fleet --late 5 --out result/rc_fleet_l5_s46 --validate-only
python -m analysis.test_fleet_context_20261124
python -m analysis.test_customer_graph_20261125
python -u -m analysis.train_fleet_context_portable --model fleet --late 5 --out result/rc_fleet_l5_s46
```

Use `--late 10` with another output directory for the second objective. Use
`--model customer_graph` for the prototype, also with a fresh directory.
The launcher trains then evaluates Pure RL Test80 automatically. Seed46,
vehicle100/travel1, epochs1000, LR1e-4 through500 then linear to1e-5, advantage std,
return mask and dispatch capacity mask ON. `--resume` requires unchanged sources.
Do not load a fleet checkpoint into the graph architecture.

## Current local Proposed wrappers: artifact-dependent, NOT portable launchers

- `analysis/train_kim_fleet_context_20261124.py`: actual local stage-1 training wrapper.
- `analysis/evaluate_fleetctx_l5_test_20261125.py`: completed lambda5 causal Test80 runner.

These preserve the research commands, but require local `server_export/`, cache
and checkpoint paths. A clean clone alone cannot execute them. The local trainer
has no resume CLI; do not restart into the same output directory. The portable
Pure RL launcher must NOT be used to evaluate a Proposed checkpoint.

Current local lambda5 uses RL+LKH example cache, while lambda10 uses RL-only cache.
Thus their comparison changes reference content as well as lateness weight.
Stage-1 lambda5 completed training and Test80; lambda10 was still training at this
snapshot. Original Proposed and fleet Proposed are distinct models. Test statistics
are over 80 scenarios for one seed, not uncertainty over training seeds.

Initial evaluation responses can be reused only for identical requests. Accident
responses require the actual disclosure-time state; do not reuse a response from
a different route state. Explicit API authorization remains necessary.

## Diagnostics and interpretation

The new `analysis/diagnose_fleetctx*`, `check_fleetctx_relocation_prefix*`,
`probe_fleetctx_relocation*` and `continue_fleetctx_relocation*` scripts require the
saved result artifacts on the research PC. They are not training entry points.
They separate fixed-assignment order changes, fixed-fleet customer relocation,
original-prefix legality, and single-action policy continuation.
Full-event oracle route improvements are not causal Test improvements. No oracle
Test actions should be used as training labels for the reported comparison.

## Organization and reproducibility

No implementation folders were moved: imports and running jobs remain intact.
Historical documentation is retained under `docs/legacy`. `source_manifest.json`
records updated source hashes. Existing data and older result archives are retained;
this update does not upload new model checkpoints, LLM response caches or API keys.
Always record commit, config, prompt/cache identity and checkpoint hash. Keep
baseline, fleet and graph outputs in separate directories.
