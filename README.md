# SoftPOMO-VRPTW

POMO-based reinforcement learning for the Vehicle Routing Problem with Time Windows (VRPTW), augmented with LLM soft-cluster confidence bias.

---

## Overview

Standard POMO learns routing from scratch via RL. This extension adds an LLM-guided soft-clustering layer:

1. **Kim(2006) clustering** divides customers into K geographic clusters (one per vehicle).
2. **LLM** scores each customer within its cluster (0–1) based on time-window urgency and real-time events (rain, accidents).
3. **Confidence bias** is applied at every routing step (not just depot dispatch), guiding the RL policy without hard constraints.

### RL-Recluster pipeline (recommended)

```
Phase 1 (500 epochs, pure RL)
    ↓  checkpoint-500.pt
gen_test_rl_cluster.py          ← inference → RL-derived clusters → LLM confidence
    ↓  {name}_rl_cluster.json
Phase 2 (epochs 501–1000, RL + LLM bias on RL clusters)
    ↓  result_soft_rl_recluster/
```

This eliminates Kim(2006) geographic clusters and instead uses RL-derived route groupings that reflect the model's actual learned behaviour.

---

## File structure

```
soft_pomo/
├── train_soft_cluster.py       # Main train / eval script
├── SoftClusterLLMModule.py     # LLM prompt builders & cluster confidence API
├── SoftClusterOntology.py      # VRPTW ontology context for LLM prompts
├── gen_test_rl_cluster.py      # Phase 1 → RL-cluster cache generator
├── eval_soft_plug.py           # Plug-in inference eval (checkpoint → LLM results)
├── build_fewshot_cache.py      # Build few-shot experience caches from episode data
├── build_rl_cluster_cache.py   # Batch RL-cluster cache builder
├── extract_rl_routes.py        # Extract best routes from a checkpoint
├── validate_llm_cache.py       # Validate / inspect llm_cache JSON files
├── train_soft_pomo.py          # Baseline: POMO without LLM (reference)
└── scripts/                    # Experiment runner scripts (not core)
```

---

## Requirements

- Python 3.10+
- PyTorch (CUDA recommended)
- Ollama running locally with `deepseek-r1:32b` (or another model)
- Solomon benchmark data under `../data/Solomon/`

Install Python dependencies (same environment as `original_POMO`):

```bash
pip install torch numpy
```

---

## Usage

### Phase 1 — Pure RL training (500 epochs)

```bash
python train_soft_cluster.py --benchmark r1 --epochs 500 --no-llm \
    --result-dir result_soft_rl_recluster
```

> `--no-llm` disables LLM bias so Phase 1 is pure RL. The checkpoint is used as the seed for RL-cluster generation.

### Generate RL-cluster cache

Run inference with the Phase 1 checkpoint, cluster by RL routes, score with LLM:

```bash
python gen_test_rl_cluster.py \
    --benchmark r1 \
    --resume result_soft_rl_recluster/r102-r112/checkpoint-500.pt \
    --result-dir result_soft_rl_recluster \
    --model deepseek-r1:32b --aug 8
```

Output: `result_soft_rl_recluster/llm_cache/{name}_rl_cluster.json` for all train + test instances.

### Phase 2 — RL training with LLM cluster bias (epochs 501–1000)

```bash
python train_soft_cluster.py --benchmark r1 --epochs 1000 \
    --resume result_soft_rl_recluster/r102-r112/checkpoint-500.pt \
    --result-dir result_soft_rl_recluster \
    --llm-cache-dir result_soft_rl_recluster/llm_cache \
    --experience-refresh-epoch 999999
```

`--experience-refresh-epoch 999999` disables mid-training experience refresh (not needed when using pre-generated RL clusters).

### Other benchmarks

Replace `--benchmark r1` with `c1` or `rc1`.

---

## LLM cache structure

Each instance has a cluster cache file in `{result_dir}/llm_cache/`:

| File pattern | Description |
|---|---|
| `{name}_rl_cluster.json` | RL-derived clusters + LLM confidence (highest priority) |
| `{name}_cluster.json` | Kim(2006) clusters + LLM confidence |
| `{name}_fewshot_cluster.json` | Cluster cache built with episode few-shot examples |
| `{name}_experience.txt` | Episode-derived few-shot text (input to fewshot cache build) |

Cache format (compact, per-cluster):
```json
{"0": {"12": 0.85, "37": 0.60}, "1": {"5": 0.92, "21": 0.45}, ...}
```

---

## Benchmark instances

Solomon benchmark with stochastic events added:

- **Base**: standard Solomon instance (e.g. `R101`, `C101`)
- **RAIN**: base + rain event on a subset of road segments (2 variants: `_RAIN_A`, `_RAIN_B`)
- **ACC**: base + traffic accident on one edge (`_ACC_A`, `_ACC_B`)

Train on R102–R112 (R1 benchmark); test on R101.

---

## Result directories

| Directory | Description |
|---|---|
| `result_soft_pure_rl/` | Pure RL baseline (no LLM) |
| `result_soft_deepseek/` | Kim clusters + DeepSeek LLM |
| `result_soft_rl_deepseek/` | RL-recluster + DeepSeek LLM (previous run) |
| `result_soft_rl_recluster/` | Current RL-recluster experiment |

---

## Key design choices

**Why RL clusters instead of Kim(2006)?**
Kim clusters are geographic. RL clusters reflect actual learned routing behaviour — they respect TW feasibility and travel patterns learned over training, so LLM confidence scores computed on them are more actionable.

**Why soft bias instead of hard constraints?**
RL action space stays fully open. The confidence bias nudges the policy toward urgent customers without preventing the agent from learning better deviations.

**Reward config F** (default):
```
r = -(Lt / Lt_max) - 0.333*(D / D_max) - (K / N)
    Lt_max=5.0, D_max=3.0
```
Prioritises lateness reduction over travel distance.
