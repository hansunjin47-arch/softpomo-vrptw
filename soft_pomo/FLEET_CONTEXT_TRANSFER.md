# Fleet context and customer graph — 2026-11-25

Use branch `code-sync-20261113`. Work in a separate checkout if an experiment
is already running on the destination PC; do not update code underneath it.

## Included implementations

- `experiments/concurrent_fleet_pair/fleet_context.py`: stage 1, public active
  fleet attention added to the existing selected-vehicle query. Includes the
  query-packing fix (`pack_queries=False`) used by the running experiments.
- `experiments/concurrent_fleet_pair_kim/fleet_context.py`: the same policy
  composed with existing Proposed confidence/NEW-head guidance.
- `experiments/concurrent_fleet_pair/customer_graph.py`: stage 2 prototype,
  stage 1 plus one vehicle-to-customer message-passing residual into customer
  embeddings and dynamic decoder keys/values. This is NOT a separate static
  customer-customer GNN, and NOT a full vehicle-vehicle heterogeneous graph.

Both preserve environment-selected routing vehicles, dispatch rules, time,
action masks and reward. No hidden arrival/available times are read. Graph
edges use nominal travel from last node and target, capacity minus demand,
target equality and pending-destination equality. Residual output projections
start at zero. Full training batch size is unchanged; graph computation is
chunked over rollouts to limit temporary memory.

## Portable Pure RL launch (no LLM/API needed)

Activate the destination Python 3.12 environment with CUDA-compatible PyTorch.
Install `requirements-code.txt` if needed. From `soft_pomo`:

```powershell
python -m analysis.train_fleet_context_portable --model fleet --late 5 --out result/rc_fleet_l5_s46 --validate-only
python -m analysis.test_fleet_context_20261124
python -m analysis.test_customer_graph_20261125

# Stage 1 comparison, only if needed on this PC:
python -u -m analysis.train_fleet_context_portable --model fleet --late 5 --out result/rc_fleet_l5_s46

# Stage 2 graph prototype, separate fresh training:
python -u -m analysis.train_fleet_context_portable --model customer_graph --late 5 --out result/rc_graph_l5_s46
```

These are alternative training commands, not a request to run both. `--late 10`
selects the other objective; use a different output directory. `--resume`
resumes an unchanged run with a checkpoint. Do not resume a fleet checkpoint
as a graph model. Further architecture edits should use a new output directory.

Settings: RC101 Train20/Test80, seed46, vehicle100/travel1, late5 or late10,
fresh1000 epochs, return mask and dispatch-capacity mask ON, advantage std
normalization ON, learning rate1e-4 through epoch500 then linear to1e-5 at1000.
Final greedy Test80 runs automatically. Outputs include checkpoint/config
hashes, per-instance metrics and mean ±1.96 sample SD/sqrt80. This interval
describes scenario variation for one trained seed, not training-seed variation.

Inputs reuse `experiment_inputs/new_head/rc101/dataset`, byte-verified against
the current server experiment. New configs are in `experiment_inputs/fleet_context`.
`analysis/fleet_context_trainer.py` is the frozen server trainer with only its
historical default-config file dependency removed; optimization is unchanged.

## Proposed and further GNN development

The composed Proposed policy is included, but this portable launcher is Pure
RL only. Proposed requires the matching confidence cache and causal online
Test runner; do not substitute Pure RL evaluation or reuse mismatched accident
vehicle-state responses. No cache, checkpoint, secret or paid API invocation
is added by this handoff.

For additional static customer GNN work, preserve the current files as
comparisons and add a new policy module/model option. Keep objective, masks,
training schedule and data fixed. CPU regression tests cover zero-residual
equivalence, public-state sensitivity, hidden-time independence, fleet-node
permutation and checkpointed full-trajectory gradients. The stage 2 prototype
also passed a CPU forward check at batch128 ×100 rollouts. Production graph
GPU memory/speed and final performance have not yet been established.
