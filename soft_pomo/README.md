# SoftPOMO-VRPTW

POMO-based reinforcement learning for the Vehicle Routing Problem with Time Windows (VRPTW),
augmented with an LLM-driven soft-clustering confidence bias and a lightweight VRPTW ontology.
Target: generalization to held-out Solomon benchmark instances (C1/R1/RC1) for a paper
submission to *Transportation Research Part E*.

---

## Pipeline

```
Phase 1 (epochs 1-500, pure RL, no LLM)
    train_vrptw.py  →  result/config_E_{bench}_final_weightedsum_b32_aug8/.../checkpoint-500.pt
        ↓
Phase 1→2 bridge: cluster by the model's OWN routing, not geography
    train_soft_cluster.py --cache-only  (cluster_mode='rl')
        → runs the Phase-1 checkpoint's best solution per instance
        → each vehicle's visited nodes become that vehicle's "cluster"
        → ONE combined LLM call scores every cluster of an instance at once
        → result_soft/llm_cache_.../{name}_rl_cluster.json
        ↓
Phase 2 (epochs 501-1000, RL + LLM confidence bias)
    train_soft_cluster.py --resume checkpoint-500.pt --llm-cache-dir ...
        → additive per-(vehicle, node) logit bias at every decode step
        → result_soft/{result-dir}/{pool}/checkpoint-*.pt
```

Phase 2 never hard-constrains the policy — the LLM score only nudges logits
(`score = score + llm_bias`, see `train_vrptw_llm.py`'s `_Decoder.forward`), so the
RL agent can still override a bad LLM judgment.

---

## Where to look, by question

| Question | File |
|---|---|
| How does the RL model / decoder work? | `train_vrptw_llm.py` (`VRPTWModel`, `_Decoder`, `VRPTWLLMTrainer`) |
| How does Phase 1 (pure RL, no LLM) train? | `train_vrptw.py` — also owns `_REWARD_CONFIGS[_R1/_RC1]` (A–K, RF) and `_load_model_flexible` |
| How does Phase 2 (LLM-biased) train, and all CLI flags? | `train_soft_cluster.py` — `SoftClusterTrainer`, `main()` argparse block |
| How is the LLM prompted, and how is its JSON parsed? | `SoftClusterLLMModule.py` — `build_all_clusters_prompt`, `parse_all_clusters_confidence`, `get_all_clusters_confidence` |
| How is the raw Ollama HTTP call made? | `VRPTWLLMModule.py` — `query_llm` |
| What ontology/domain concepts feed the prompt? | `SoftClusterOntology.py` (cluster-aware) / `VRPTWOntology.py` (base TBox/ABox) |
| How is the environment/reward/data loading defined? | `vrptw_env.py` (`VRPTWEnv`, `load_solomon`, `make_batch`) |
| How is the REINFORCE advantage computed? | `advantage.py` (`hybrid_advantage`, lexicographic epsilon-constraint preference) |
| How are TW-augmented / event variant instances generated? | `generate_vrptw.py`, `generate_r1_tw_variants.py`, `generate_c_rc_tw_variants.py` |
| How do I sanity-check a cache file? | `validate_llm_cache.py` |
| How do I compare against a strong heuristic baseline? | `lkh_eval_*.py` (LKH-3 — a heuristic local-search solver, not an exact/optimal one; treat its output as a strong reference point, not a proven optimum) |

---

## The LLM confidence mechanism (Phase 2 bias)

For each instance, ALL of that instance's vehicle-clusters are scored in a **single
combined LLM call** (`get_all_clusters_confidence` → `build_all_clusters_prompt`).
Three independently-toggleable pieces make up the score:

1. **Misfit gate** (Step 1, OFF by default — opt in via `--use-misfit` as an ablation):
   when on, the LLM flags customers that don't belong in this vehicle's cluster at all;
   flagged customers get a hardcoded `MISFIT_BIAS = -1.0` and are excluded from Step 2's
   priority ranking. By default this step is skipped entirely — every customer gets a
   plain Step 2 score, and "misfit" is defined purely by the depot-cutoff comparison
   below (scored below depot → `MISFIT_BIAS`), not by a separate LLM-flagged list.
2. **Priority score** (Step 2, 0.0–1.0 per customer): reflects visit urgency for THIS
   vehicle, informed by the current RL route order, TW slack, and any disruption
   events (rain/accident) active on that instance.
3. **Depot score** ("0", same 0.0–1.0 scale as customers): the newest piece. The LLM
   scores the depot itself, once per vehicle, functioning as a cutoff — any customer
   scored below it is converted to `MISFIT_BIAS` at tensor-build time
   (`_build_cluster_conf_tensor` in `train_soft_cluster.py`), signaling "this vehicle
   should wrap up, another vehicle should take this customer." The prompt requires
   the LLM to score depot as **exactly 0.0** when the vehicle's own performance looks
   fine (not a vague low-but-nonzero number), and — when a cutoff is intended — to
   place it **numerically above** the customers being pushed away, since only
   "0" `<` a customer's own score, not stated intent, has any effect on training.
   This exists because customers in every OTHER vehicle's cluster always contribute
   zero bias to this vehicle's decision (the (K, N+1) tensor only fills in a
   cluster's own members), so an un-scored depot (fixed at 0) was systematically
   disadvantaged against same-cluster customers in ties — see "Known quirks" below.

All three pieces are threaded through **two parallel code paths** in
`train_soft_cluster.py` that must both be kept in sync: the `rl_instance` tag
(base + RAIN2 instances) and the separately-branched `rl_acc` tag (ACC2 instances,
~line 690). A live mid-episode accident re-score (`_refresh_clusters_accident`) is a
third, currently un-synced path — it does not thread depot scores.

---

## Cache file format

`{llm-cache-dir}/{name}_rl_cluster.json`:

```json
{
  "0": {"54": 1.0, "39": 0.9, "23": 0.8},
  "1": {"...": "..."},
  "_stats": {"0": {"late_count": 0, "total_late": 0.0, "travel_time": 45.6,
                    "node_detail": {"54": {"slack": 127.2, "leg_time": 22.8}}}},
  "_depot": {"0": 0.0, "1": 0.4}
}
```

- Top-level numeric keys (`"0"`, `"1"`, ...) are cluster/vehicle indices; each maps
  `{node_id: confidence}` (misfit-flagged or below-depot-cutoff nodes get `-1.0`).
- `_stats`: per-cluster actual-performance summary computed from the current RL best
  solution, injected into the prompt as `[Actual performance]` / `[RL visit order]`
  so the LLM scores against ground truth, not blind guesses.
- `_depot`: per-cluster depot score (see mechanism above). Missing/empty on any file
  means that instance never got a depot judgment (defaults to 0 bias everywhere).

A cache with every score `== 1.0` is a **placeholder** — `train_soft_cluster.py`
detects this (`is_placeholder`) and calls the LLM fresh instead of trusting it.

---

## Key `train_soft_cluster.py` CLI flags

| Flag | Meaning |
|---|---|
| `--benchmark {c1,rc1,r1,mixed,c2}` | Train pool + held-out test instance (see below) |
| `--config {A..K,RF}` | Reward weighting, defined in `train_vrptw.py::_REWARD_CONFIGS[_R1/_RC1]` |
| `--start-mode {det_conf,random,confidence_topk}` | How a vehicle's cluster is locked in. `det_conf` (legacy default) assumes vehicle *k* always visits cluster *k* and never re-verifies — breaks once the policy drifts from the initial clustering. `random`/`confidence_topk` track the cluster dynamically from whichever node the vehicle actually starts on (no bias on that first pick, correct alignment from the second stop on) — **use this one**. |
| `--use-misfit` | Opt into the separate LLM-flagged Step-1 misfit gate (ablation) — OFF by default |
| `--bias-strength` | Scalar multiplier on the additive LLM logit bias (default 5.0) |
| `--resume PATH` | Warm-start from a checkpoint (`--resume .../checkpoint-500.pt`) |
| `--result-dir` / `--llm-cache-dir` | Output / cache directory overrides |
| `--cache-only` | Generate/refresh LLM caches for the whole benchmark, then exit (no training) |
| `--refresh-cache` | Force a fresh LLM call even for non-placeholder caches |
| `--pomo N` | POMO rollout count (must be passed explicitly; kept identical across LLM-on/off runs so it never confounds the comparison) |
| `--acc-ratio` | Fraction of training batches drawn from ACC-event instances (default 0.1) |

## Benchmark train/test split

| `--benchmark` | Train pool | Held-out test |
|---|---|---|
| `r1` | R102–R112 (+ `_RAIN2`, `_ACC2` variants) | R101 (base / `_ACC2` / `_RAIN2`) |
| `rc1` | RC102–RC108 (+ variants) | RC101 |
| `c1` | C102–C109 (+ variants) | C101 |

`_RAIN2` = one rain-event variant per base instance; `_ACC2` = one accident-event
variant. Both are generated on the fly (not static files) with event severity
qualitatively described to the LLM (exact slowdown multiplier is withheld — the LLM
must reason about it, not read it off).

---

## Known quirks / open issues (see git history / session logs for detail)

- **`gpt-oss:20b` reasoning instability on long prompts**: for a small fraction of
  instances, the model's "thinking" phase can spiral until it fills the entire
  `num_ctx` (`prompt_tokens + eval_tokens == num_ctx` exactly, at `temperature=0.0`
  so it's deterministic *within* a process run) and gets truncated before emitting
  JSON. A plain retry — new process, sometimes with the Ollama server restarted —
  usually succeeds on some later attempt; there is no per-instance special-casing.
- **Depot-score reliability**: the LLM occasionally misreads or ignores the injected
  `[Actual performance]` stats (e.g. claiming "all routes on time" when `_stats`
  shows a late customer), and depot scores were, before the current prompt wording,
  observed to come out uniform across every cluster in a scenario regardless of
  which specific vehicle was overloaded — i.e. acting like a scenario-wide dial
  rather than a per-vehicle judgment. The current prompt (exact-0.0 rule + explicit
  "score above the customers you mean to exclude" + cross-vehicle-parity note) was
  added to address this; verify against fresh cache stats before trusting a training
  run's depot behavior.
- **`rl_acc` cache path lags `rl_instance`**: any future change to the confidence/
  depot pipeline must be applied to BOTH `cluster_mode == 'rl'` branches in
  `train_soft_cluster.py` (search `rl_instance` and `rl_acc`) plus the live
  `_refresh_clusters_accident` path, or ACC-tagged instances will silently miss it.
