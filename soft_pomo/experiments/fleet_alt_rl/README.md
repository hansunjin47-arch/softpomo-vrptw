# Local B diagnosis on RC101

Use the code-sync-20261113 branch and activate the project Python environment with PyTorch. From the soft_pomo directory run:

```powershell
python -m analysis.diagnose_alt_b_20261126 experiment_inputs/fleet_context/b_guided_l5/diagnostic_weights.pt result/b_diagnosis/train20.json
```

The supplied 6th-epoch Q-network is a snapshot of the local Guided Double DQN, training seed 46 and objective weights 100/1/5. The evaluator runs on CPU, uses the original Train20 confidence responses offline and writes metrics for five action selection rules. Stochastic modes sample four routes per case; greedy modes evaluate one. The sharper mode divides the entire guided score by 0.25 before softmax. This preserves the greedy ranking. These are training-instance diagnostics.

For a fresh GPU run on this PC:

```powershell
python -u -m experiments.fleet_alt_rl.train --method b --out result/b_local_s46
```

The training command saves a full resumable checkpoint every epoch. The bundled compact snapshot contains model weights and replay counts for evaluation. To diagnose a full checkpoint generated on the other PC use its checkpoint.pt as the first argument. Use a stable checkpoint copy if training concurrently.
