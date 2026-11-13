"""
train_soft_cluster.py — Soft-Clustering POMO for VRPTW.

LLM assigns per-cluster confidence scores that bias the RL policy at every routing step.
Clusters come from Kim(2006) geographic grouping (default) or from RL-derived routes
(RL-recluster pipeline via gen_test_rl_cluster.py).

Architecture vs original_POMO:
  original_POMO : LLM → top-K global start nodes, bias at depot dispatch only
  soft_pomo     : K clusters (one per vehicle), LLM scores per cluster,
                  bias = cluster_conf[k, node] applied at every step for vehicle k,
                  RL action space fully open (no hard zone constraint)

RL-recluster pipeline (two-phase training):
  Phase 1  →  --no-llm --epochs 500          (pure RL, save checkpoint-500.pt)
  generate →  gen_test_rl_cluster.py         (RL routes → clusters → LLM confidence)
  Phase 2  →  --resume checkpoint-500.pt \   (RL + LLM bias on RL-derived clusters)
               --epochs 1000 \
               --llm-cache-dir <cache>

Quick run:
  python train_soft_cluster.py --benchmark r1
  python train_soft_cluster.py --benchmark r1 --no-llm
  python train_soft_cluster.py --benchmark r1 --test-only
"""
from __future__ import annotations

import os
import sys
import math
import json
import random
import argparse
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def _parse_cluster_cache(data: dict):
    """Parse cluster cache JSON — two formats supported.
    Old: {"clusters": [[...], ...], "confidence": {"node": score, ...}}
    New: {"0": {"node": score, ...}, "1": {...}, ..., "_stats": {...}}  (compact,
    written by _dump_cluster_cache / extract_rl_routes.routes_to_cluster_json)
    Returns (clusters: list[list[int]], flat_conf: dict[int, float]) -- any
    non-numeric top-level key (e.g. "_stats", read separately via
    _parse_cluster_stats) is ignored here, not treated as a cluster index.
    """
    if 'clusters' in data:
        clusters  = data['clusters']
        flat_conf = {int(k): v for k, v in data['confidence'].items()}
    else:
        keys      = sorted((k for k in data if not k.startswith('_')), key=lambda x: int(x))
        clusters  = [[int(n) for n in data[k]] for k in keys]
        flat_conf = {int(n): s for k in keys for n, s in data[k].items()}
    return clusters, flat_conf


def _parse_cluster_stats(data: dict):
    """Read the optional {"_stats": {cluster_idx: {late_count, total_late,
    travel_time, node_detail}}} block written by extract_rl_routes.routes_to_
    cluster_json. Returns {cluster_idx (int): {...}} or {} if absent (older
    caches without it)."""
    raw = data.get('_stats', {})
    if not isinstance(raw, dict):
        return {}
    return {int(k): v for k, v in raw.items()}


def _parse_cluster_depot(data: dict):
    """Read the optional {"_depot": {cluster_idx: depot_score}} block --
    depot needs a DIFFERENT value per vehicle (unlike every real customer,
    which belongs to exactly one cluster), so it can't live in the per-cluster
    {node: score} blocks that _parse_cluster_cache flattens across clusters;
    see parse_all_clusters_confidence's docstring. Returns {cluster_idx (int):
    depot_score (float)} or {} if absent (older caches, or misfit-era caches
    generated before depot scoring existed)."""
    raw = data.get('_depot', {})
    if not isinstance(raw, dict):
        return {}
    return {int(k): float(v) for k, v in raw.items()}


def _dump_cluster_cache(clusters: list, flat_conf: dict) -> dict:
    """Serialize to compact format: {cluster_idx: {node: score}}."""
    return {str(k): {str(n): round(flat_conf.get(n, 0.0), 2) for n in nodes}
            for k, nodes in enumerate(clusters)}

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from advantage import hybrid_advantage, eps_schedule, tier_counts

from train_vrptw_llm import (
    VRPTWLLMTrainer,
    VRPTWModel,
    env_params     as _base_env_params,
    model_params,
    optimizer_params,
    trainer_params as _base_trainer_params,
    llm_params     as _base_llm_params,
    _get_rain_nodes_mult_evs,
    _get_accidents,
    _make_rain_tokens,
    _make_acc_tokens,
    _load_model_flexible,
    _inst_tag,
    _set_seed,
)
from vrptw_env import (
    VRPTWEnv, load_solomon, make_batch,
    _extract_routes, _print_solution, _save_solution,
    plot_routes, plot_training_curves,
)
from VRPTWLLMModule import (
    AccidentEvent,
    build_accident_prompt, query_llm, parse_priority,
)
from VRPTWOntology import VRPTWOntology as _VRPTWOntology
import train_vrptw as _base_config

from SoftClusterLLMModule import (get_cluster_confidence, refresh_cluster_confidence,
                                  get_all_clusters_confidence, build_bks_fewshot_block,
                                  build_rl_fewshot_block, MISFIT_BIAS)
from SoftClusterOntology import episode_tracker, base_of, EpisodeLog, PhaseOneEpisodeLogger

try:
    import mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False

_ROOT     = os.path.dirname(_HERE)
DATA_DIR  = os.path.join(_ROOT, 'data', 'Solomon')
RESULT_DIR = os.path.join(_HERE, 'result_soft')

# ── Default params (copy from base, override result_dir) ─────────────────────
env_params     = dict(_base_env_params)
trainer_params = dict(_base_trainer_params)
trainer_params['result_dir'] = RESULT_DIR
# 'rl' (current research direction): clusters from RL-derived routes, missing
# cache is a hard error. 'kim2006': original geographic-clustering pipeline,
# live clustering + LLM call. Set explicitly per script -- never mixed silently.
trainer_params['cluster_mode'] = 'rl'
# 'cluster' (original): deterministic per-cluster starts (_det_starts) + cyclic
# vehicle->cluster assignment that varies by rollout index (roll_shifts) -- this
# couples POMO's rollout-diversity axis to bias difficulty, confounding the mean
# -of-rollouts baseline used for advantage (some rollouts get structurally easier
# /harder cluster-visit schedules than others, for reasons unrelated to policy
# quality). 'node': free/diverse starts identical to pure RL (no _det_starts, no
# cluster/vehicle bookkeeping) + a single flat per-node confidence bias applied
# uniformly to every rollout and every vehicle -- removes the confound entirely,
# at the cost of no longer structurally enforcing "vehicle k serves cluster k"
# during rollout (confidence becomes a pure per-node priority hint). Set
# explicitly per script -- never mixed silently.
trainer_params['bias_mode'] = 'cluster'
# ACC instances always start from their base instance's own (pre-accident)
# confidence and get a live post-accident refresh at the real per-rollout
# trigger time, applied per rollout -- train and test now share the exact same
# information-structure (no separate acc_conf_mode setting needed anymore; see
# _ensure_soft_cluster_cache's is_acc branch and _acc_conf_now).
# 'det_conf' (original, bias_mode='cluster' only): every vehicle's cluster comes
# from _det_starts (deterministic, highest-confidence-per-cluster) + a cyclic
# (veh_idx + rollout_index) % K_clusters schedule -- the rollout-index term is
# an arbitrary per-rollout confound (see design discussion: two rollouts get
# structurally different bias difficulty for reasons unrelated to policy
# quality). 'random': vehicle 1's start is a genuinely random distinct node per
# rollout (pure RL's OWN diversity mechanism, see train_vrptw.py), so vehicle
# 1's cluster is whichever cluster that node belongs to -- no arbitrary index
# involved. Vehicle 2+ get NO fixed schedule either: a fresh vehicle's first
# pick is unbiased (its cluster isn't decided yet), and whatever cluster that
# pick lands in becomes that vehicle's bias for its remaining stops. Set
# explicitly per script -- never mixed silently.
trainer_params['start_mode'] = 'det_conf'
llm_params     = dict(_base_llm_params)
llm_params['soft_clustering'] = True   # flag for this architecture
llm_params['use_misfit'] = False   # Step 1 misfit gate is OFF by default: every customer
# gets a plain 0-1 Step 2 score (including depot), and misfit is defined purely by the
# depot-cutoff comparison (below-depot -> MISFIT_BIAS) -- see --use-misfit to opt back
# into the separate LLM-flagged Step 1 gate as an ablation.

# Test instances default (C1 benchmark)
TEST_INSTANCES = ["c101", "c101_rain_A", "c101_rain_B", "c101_acc_A", "c101_acc_B"]


# ── Soft-Cluster Trainer ──────────────────────────────────────────────────────

class SoftClusterTrainer(VRPTWLLMTrainer):
    """
    Extends VRPTWLLMTrainer with soft-clustering LLM confidence.

    Key differences:
      - _ensure_llm_cache → _ensure_soft_cluster_cache (cluster + per-cluster LLM)
      - _det_starts        → top-confidence node per cluster
      - Training loops     → per-vehicle per-step cluster confidence bias
    """

    def __init__(self, env_p, model_p, opt_p, trainer_p, llm_p):
        if trainer_p.get('guidance_mode') == 'selective_advance':
            if (trainer_p.get('bias_mode', 'cluster') != 'cluster'
                    or trainer_p.get('start_mode') != 'random'
                    or trainer_p.get('cluster_mode') != 'rl'
                    or trainer_p.get('experience_refresh_epoch', -1) != -1
                    or trainer_p.get('rl_recluster_epoch') is not None):
                raise ValueError('Selective advance requires fixed RL clusters, random starts, and no automatic refresh')
        # Cluster caches must exist before super().__init__ calls _ensure_llm_cache
        self._cluster_assign_cache:  dict[str, list[list[int]]] = {}
        self._cluster_conf_cache:    dict[str, torch.Tensor]    = {}
        self._original_cluster_conf: dict[str, torch.Tensor]    = {}
        # RL-mode only: {name: {cluster_idx: {late_count, total_late, travel_time}}}
        # -- each vehicle's ACTUAL realized performance from extract_rl_routes.py,
        # shown to the LLM alongside static TW/geometry (see _parse_cluster_stats).
        self._cluster_stats_cache:   dict[str, dict[int, dict]]  = {}
        # Per-rollout accident confidence staging: name -> list of
        # {'t_start': float, 'tensor': (K, N+1)} in chronological trigger order.
        # cluster_conf_cache[name] always holds the BASE (pre-accident) tensor for
        # an ACC instance -- an accident never overwrites it in place. Instead,
        # once _refresh_clusters_accident computes a wave's post-accident tensor,
        # it's appended here, and bias computation picks the latest stage whose
        # t_start THIS ROLLOUT'S OWN current_time has individually passed (see
        # _acc_conf_now), never a batch-wide min/max swap applied to everyone at
        # once regardless of their own progress.
        self._acc_stage_events: dict[str, list[dict]] = {}
        # Names for which precompute has been ATTEMPTED this process run, whether
        # or not every wave succeeded -- distinct from _acc_stage_events' own
        # length, so a wave that fails (falls back, uncached -- see
        # _refresh_clusters_accident) doesn't get silently re-attempted on every
        # single subsequent batch/epoch that touches this instance (each retry is
        # a real LLM call). Retrying only happens on a genuinely NEW run (fresh
        # process, empty set) or an explicit force_refresh.
        self._acc_precompute_done: set[str] = set()
        self._timing = {'llm_init': 0.0, 'llm_refresh': 0.0, 'rl_train': 0.0}
        # Ontology-upgrade Phase 1 diagnostic logging (design doc Part I) -- off
        # by default; enabled via trainer_p['episode_log_enabled'].
        self._current_epoch  = 0
        self._tier_counts    = {'t0': 0, 't1': 0, 't2': 0, 'total': 0}
        self._phase1_logger  = None
        if trainer_p.get('episode_log_enabled', False):
            _epochs   = trainer_p.get('epochs', 500)
            _out_dir  = trainer_p.get('episode_log_dir') or os.path.join(
                trainer_p.get('result_dir', 'result_soft'), 'episode_log')
            self._phase1_logger = PhaseOneEpisodeLogger(
                out_dir=_out_dir,
                start_epoch=trainer_p.get('episode_log_start_epoch', max(1, _epochs - 100 + 1)),
                end_epoch=trainer_p.get('episode_log_end_epoch', _epochs),
                sample_every=trainer_p.get('episode_log_sample_every', 10),
                full_epochs=trainer_p.get('episode_log_full_epochs', 5),
            )
            print(f"[Phase1Log] enabled: epochs [{self._phase1_logger.start_epoch}, "
                  f"{self._phase1_logger.end_epoch}], sampled every "
                  f"{self._phase1_logger.sample_every}, full rollouts for last "
                  f"{self._phase1_logger.full_epochs} epochs -> {_out_dir}")
        # Test instances use real-time LLM on accident — no pre-generation
        self._test_instance_names = {n.upper() for n in trainer_p.get('test_instances', [])}
        self._acc_prompt_counter: dict[str, int] = {}
        self._experience_refreshed = False
        self._experience_refresh_epoch = trainer_p.get('experience_refresh_epoch', 500)
        self._rl_reclustered = False
        self._rl_recluster_epoch = trainer_p.get('rl_recluster_epoch', None)
        self._pending_cache_retry: list = []
        # Few-shot block: precomputed once, prepended to init LLM calls that have no episode experience.
        # Skipped entirely for cluster_mode='rl' -- that pipeline's confidence calls
        # (_ensure_soft_cluster_cache's 'rl' branches) never pass bks_fewshot through
        # to get_all_clusters_confidence, so building it there was dead work that also
        # printed a misleading "[BKS fewshot] Loaded..." log line implying it was used.
        self._bks_fewshot_block = ""
        if (llm_p.get('enabled', True) and not llm_p.get('no_init_llm', False)
                and trainer_p.get('cluster_mode', 'rl') != 'rl'):
            benchmark    = trainer_p.get('benchmark', 'r1')
            use_rl_fs    = trainer_p.get('rl_fewshot', False)
            _cache_dir   = (trainer_p.get('llm_cache_dir')
                            or os.path.join(trainer_p.get('result_dir', 'result_soft'), 'llm_cache'))
            try:
                if use_rl_fs:
                    self._bks_fewshot_block = build_rl_fewshot_block(_cache_dir, DATA_DIR)
                    if self._bks_fewshot_block:
                        print('[RL fewshot] Loaded RL-derived example (r106)')
                    else:
                        print('[RL fewshot] r106_rl_cluster.json not found — run gen_test_rl_cluster.py first')
                else:
                    self._bks_fewshot_block = build_bks_fewshot_block(DATA_DIR, benchmark)
                    if self._bks_fewshot_block:
                        print(f'[BKS fewshot] Loaded reference examples for benchmark={benchmark}')
            except Exception as _e:
                print(f'[fewshot] Failed to build ({_e}); continuing without.')
        super().__init__(env_p, model_p, opt_p, trainer_p, llm_p)
        self._flush_pending_caches()
        print('[SoftCluster] Init complete: per-cluster LLM confidence loaded.')

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def _cluster_customers(self, inst: dict, K: int) -> list[list[int]]:
        """Kim(2006) capacitated TW-feasible clustering → lists of customer indices (1-indexed).

        Adapts the R_utils._kim2006_clusters() logic to the normalized inst dict format:
        - demands are normalized (capacity Q=1.0)
        - tw/service/tt denormalized by multiplying with inst['T']
        - coords are in real scale (no normalization needed)
        - customer c → coords[c-1], demands[c-1], tw_open[c], tw_close[c], service[c]
        """
        import numpy as np
        import math as _math
        import random as _random

        T        = float(inst['T'])
        tt_raw   = inst['tt'].cpu().numpy() * T                         # (N+1, N+1)
        tw_open  = np.concatenate([inst['depot_tw_open'].numpy() * T,
                                   inst['node_tw_open'].numpy()  * T])  # (N+1,)
        tw_close = np.concatenate([inst['depot_tw_close'].numpy() * T,
                                   inst['node_tw_close'].numpy()  * T]) # (N+1,)
        service  = np.concatenate([inst['depot_service'].numpy() * T,
                                   inst['node_service'].numpy()  * T])  # (N+1,)
        coords   = inst['node_xy'].cpu().numpy()   # (N, 2)  customer 1→coords[0]
        demands  = inst['node_demand'].cpu().numpy()          # (N,) normalized; Q=1.0
        N        = inst['n_customers']
        custs    = list(range(1, N + 1))           # 1-indexed customer IDs
        Q        = 1.0                             # normalized capacity
        depot_close = float(tw_close[0])

        def _d(i: int, j: int) -> float:
            return float(tt_raw[i, j])

        def _cdist(cx: float, cy: float, c: int) -> float:
            return _math.sqrt((float(coords[c - 1][0]) - cx) ** 2 +
                              (float(coords[c - 1][1]) - cy) ** 2)

        def _tw_feasible(cluster: list) -> bool:
            if not cluster:
                return True
            seq = sorted(cluster, key=lambda x: float(tw_close[x]))
            cur_time, cur_node = 0.0, 0
            for c in seq:
                arr = cur_time + _d(cur_node, c)
                if arr > float(tw_close[c]) + 1e-8:
                    return False
                cur_time = max(arr, float(tw_open[c])) + float(service[c])
                cur_node = c
            return cur_time + _d(cur_node, 0) <= depot_close + 1e-8

        def _can_add(c: int, cluster: list) -> bool:
            if sum(float(demands[x - 1]) for x in cluster) + float(demands[c - 1]) > Q + 1e-8:
                return False
            return _tw_feasible(cluster + [c])

        K_limit = min(N, int(inst.get('vehicle_limit') or 25), 25)
        K_try    = max(1, K)
        if K_try > K_limit:
            raise ValueError(f'Kim initial K={K_try} exceeds vehicle limit {K_limit}')
        clusters: list[list[int]] = []

        for _inc in range(30):
            _random.seed(42 + _inc)
            seeds        = _random.sample(custs, min(K_try, len(custs)))
            centroid_pos = [[float(coords[s - 1][0]), float(coords[s - 1][1])] for s in seeds]
            clusters     = [[] for _ in range(K_try)]
            prev_assign  = None

            # Main assignment loop (grand-centroid farthest-first)
            for _ in range(50):
                gc_x = sum(cp[0] for cp in centroid_pos) / K_try
                gc_y = sum(cp[1] for cp in centroid_pos) / K_try

                sorted_custs = sorted(
                    custs,
                    key=lambda c: _math.sqrt((float(coords[c - 1][0]) - gc_x) ** 2 +
                                             (float(coords[c - 1][1]) - gc_y) ** 2),
                    reverse=True,
                )
                new_clusters: list[list[int]] = [[] for _ in range(K_try)]
                for c in sorted_custs:
                    order = sorted(range(K_try),
                                   key=lambda i: _cdist(centroid_pos[i][0], centroid_pos[i][1], c))
                    placed = False
                    for i in order:
                        if _can_add(c, new_clusters[i]):
                            new_clusters[i].append(c)
                            placed = True
                            break
                    if not placed:
                        new_clusters[order[0]].append(c)

                for i in range(K_try):
                    if new_clusters[i]:
                        centroid_pos[i] = [
                            float(np.mean([coords[c - 1][0] for c in new_clusters[i]])),
                            float(np.mean([coords[c - 1][1] for c in new_clusters[i]])),
                        ]

                curr_assign = [tuple(sorted(cl)) for cl in new_clusters]
                clusters    = new_clusters
                if curr_assign == prev_assign:
                    break
                prev_assign = curr_assign

            # Move improvement: relocate node to closer feasible cluster
            for _ in range(50):
                cp = [
                    [float(np.mean([coords[c - 1][0] for c in cl])),
                     float(np.mean([coords[c - 1][1] for c in cl]))]
                    if cl else centroid_pos[i]
                    for i, cl in enumerate(clusters)
                ]
                moved = False
                for i, cl in enumerate(clusters):
                    for c in list(cl):
                        d_own = _cdist(cp[i][0], cp[i][1], c)
                        for j, cj in enumerate(clusters):
                            if j == i:
                                continue
                            if _cdist(cp[j][0], cp[j][1], c) >= d_own:
                                continue
                            if not _can_add(c, cj):
                                continue
                            cl.remove(c)
                            cj.append(c)
                            cp[i] = ([float(np.mean([coords[x - 1][0] for x in cl])),
                                      float(np.mean([coords[x - 1][1] for x in cl]))]
                                     if cl else cp[i])
                            cp[j] = [float(np.mean([coords[x - 1][0] for x in cj])),
                                     float(np.mean([coords[x - 1][1] for x in cj]))]
                            moved = True
                            break
                if not moved:
                    break

            raw = [cl for cl in clusters if cl]
            # Forced assignments above may violate capacity even after relocation.
            # Allow float32 normalization noise, but never return an overloaded group.
            capacity_ok = all(sum(float(demands[c - 1]) for c in cl) <= Q + 1e-6
                              for cl in raw)
            if capacity_ok and (all(_tw_feasible(cl) for cl in raw) or K_try >= K_limit):
                break    # Preserve the existing soft-TW policy at the vehicle limit.
            if K_try >= K_limit:
                raise RuntimeError(
                    f'Kim clustering for {inst["name"]} still exceeds capacity '
                    f'at vehicle limit {K_limit}; no valid clustering returned')
            K_try += 1   # add one more cluster and retry
        else:
            raise RuntimeError(f'Kim clustering for {inst["name"]} exhausted retries')

        result = [cl for cl in clusters if cl]
        print(f'  [Kim2006] {inst["name"]}: {len(result)} clusters '
              f'(K_init={K}, K_final={K_try})', flush=True)
        return result

    def _build_cluster_conf_tensor(
        self,
        clusters: list[list[int]],
        flat_conf: dict,
        K: int,
        N: int,
        depot_conf: dict | None = None,
    ) -> torch.Tensor:
        """Build (K, N+1) confidence tensor. conf[k, n] = confidence of node n for vehicle k.

        depot_conf (optional): {cluster_idx: depot_score} -- goes into conf[k, 0].
        Unlike every real customer (exactly one cluster, so flat_conf[n] is
        unambiguous), depot needs a DIFFERENT value per vehicle, so it's threaded
        in separately rather than through flat_conf. Absent/None -> conf[:, 0]
        stays 0 (old behavior, no depot scoring in this cache).

        Any customer scored BELOW its own vehicle's depot score is a misfit for
        that vehicle by construction (see build_all_clusters_prompt's depot
        instructions: that's exactly what scoring below "0" means) -- so it
        gets the fixed MISFIT_BIAS here too, same as an explicit Step-1 misfit
        flag, replacing its own (lower) score rather than sitting just under
        depot's bias.
        """
        conf = torch.zeros(K, N + 1, device=self.device)
        for k, cluster_nodes in enumerate(clusters[:K]):
            for n in cluster_nodes:
                if 1 <= n <= N:
                    conf[k, n] = float(flat_conf.get(n, 0.0))
        if depot_conf:
            for k, score in depot_conf.items():
                if not (0 <= k < K):
                    continue
                conf[k, 0] = float(score)
                for n in clusters[k]:
                    if 1 <= n <= N and MISFIT_BIAS + 1e-9 < conf[k, n] < float(score):
                        conf[k, n] = MISFIT_BIAS
        return conf

    # ------------------------------------------------------------------
    # Adaptive bias helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_late_unserved(inst: dict, routes: list) -> tuple:
        """Return (unserved_ids, late_ids, late_times, K, total_dist)."""
        import numpy as np
        T        = float(inst['T'])
        tt       = inst['tt'].cpu().numpy() * T
        tw_open  = np.concatenate([inst['depot_tw_open'].numpy() * T,
                                   inst['node_tw_open'].numpy()  * T])
        tw_close = np.concatenate([inst['depot_tw_close'].numpy() * T,
                                   inst['node_tw_close'].numpy()  * T])
        service  = np.concatenate([inst['depot_service'].numpy() * T,
                                   inst['node_service'].numpy()  * T])
        N = inst['n_customers']

        visited:    set[int]        = set()
        late:       set[int]        = set()
        late_times: dict[int, float] = {}
        total_dist  = 0.0
        K           = 0

        for route in routes:
            if not route:
                continue
            K += 1
            cur, cur_time = 0, float(tw_open[0])
            for node in route:
                if node == 0:
                    continue
                visited.add(node)
                total_dist += float(tt[cur, node])
                arr        = cur_time + float(tt[cur, node])
                svc_start  = max(arr, float(tw_open[node]))
                lateness   = max(0.0, svc_start - float(tw_close[node]))
                if lateness > 0.0:
                    late.add(node)
                    late_times[node] = lateness
                cur_time = svc_start + float(service[node])
                cur      = node
            total_dist += float(tt[cur, 0])  # return to depot

        unserved = [n for n in range(1, N + 1) if n not in visited]
        return unserved, list(late), late_times, K, total_dist

    @staticmethod
    def _compute_episode_detail(inst: dict, routes: list) -> dict:
        """Superset of _compute_late_unserved: also returns per-node arrival/wait,
        needed for the ontology-upgrade diagnostics (design doc §1, §6, §7) --
        late-chain analysis needs both to tell whether waiting absorbed a delay."""
        import numpy as np
        T        = float(inst['T'])
        tt       = inst['tt'].cpu().numpy() * T
        tw_open  = np.concatenate([inst['depot_tw_open'].numpy() * T,
                                   inst['node_tw_open'].numpy()  * T])
        tw_close = np.concatenate([inst['depot_tw_close'].numpy() * T,
                                   inst['node_tw_close'].numpy()  * T])
        service  = np.concatenate([inst['depot_service'].numpy() * T,
                                   inst['node_service'].numpy()  * T])
        N = inst['n_customers']

        arrival: dict = {}
        wait:    dict = {}
        late_by: dict = {}
        visited: set  = set()
        total_dist = 0.0
        K = 0

        for route in routes:
            if not route:
                continue
            K += 1
            cur, cur_time = 0, float(tw_open[0])
            for node in route:
                if node == 0:
                    continue
                visited.add(node)
                total_dist += float(tt[cur, node])
                arr       = cur_time + float(tt[cur, node])
                w         = max(0.0, float(tw_open[node]) - arr)
                svc_start = arr + w
                lateness  = max(0.0, svc_start - float(tw_close[node]))
                arrival[node] = round(arr, 4)
                wait[node]    = round(w, 4)
                late_by[node] = round(lateness, 4)
                cur_time = svc_start + float(service[node])
                cur      = node
            total_dist += float(tt[cur, 0])

        served = {n: (n in visited) for n in range(1, N + 1)}
        Lc = sum(1 for v in late_by.values() if v > 0.0)
        Lt = sum(v for v in late_by.values() if v > 0.0)
        return dict(arrival=arrival, wait=wait, late_by=late_by, served=served,
                    K=K, D=round(total_dist, 4), Lc=Lc, Lt=round(Lt, 2))

    # ------------------------------------------------------------------
    # LLM cluster confidence cache (replaces _ensure_llm_cache)
    # ------------------------------------------------------------------

    def _ensure_llm_cache(self, inst: dict, force_refresh: bool = False):
        """Override: redirect to soft-cluster cache."""
        self._ensure_soft_cluster_cache(inst, force_refresh)

    def _build_base_prior_ref(self, name: str, N: int) -> str:
        """For a RAIN2/ACC2 instance (cluster_mode='rl'): a COMPACT reference
        to the BASE scenario's fit+priority judgment for this scenario's LLM call.

        Deliberately compact: an earlier full per-node dump (every customer's
        exact priority rendered as its own table) roughly doubled prompt size
        and drove gpt-oss:20b into runaway "thinking" that exhausted the
        context budget on every retry (see git history / session notes)
        without ever finishing valid JSON. A flat {node: score} dict is far
        smaller than that (no per-row table formatting, just one compact
        mapping) while still preserving both structural facts worth carrying
        over: WHICH customers were flagged as not belonging under base, and
        roughly how each non-misfit customer was prioritized -- so the LLM
        can judge whether the event plausibly changed either one, not just
        the misfit list.

        NOT a delta/adjustment of base's clusters -- this scenario has its own
        independently-extracted routing (clusters can genuinely differ under
        the event, e.g. a different vehicle count), so the LLM still does a
        full fresh fit+priority pass; this is only a hint.
        """
        base_key = name.split('_')[0]
        if base_key == name:
            return ""
        base_tensor   = self._cluster_conf_cache.get(base_key)
        base_clusters = self._cluster_assign_cache.get(base_key)
        if base_tensor is None or base_clusters is None:
            return ""
        misfits = sorted(
            n for k, nodes in enumerate(base_clusters) for n in nodes
            if 1 <= n <= N and float(base_tensor[k, n]) <= -0.5
        )
        base_conf = {
            n: round(float(base_tensor[k, n]), 2)
            for k, nodes in enumerate(base_clusters) for n in nodes
            if 1 <= n <= N and float(base_tensor[k, n]) > -0.5
        }
        label = "accident" if "ACC" in name.upper() else "rain" if "RAIN" in name.upper() else "event"
        conf_line = (
            f'[Base reference] Under the base (no {label}) scenario, non-misfit customers were scored '
            f'(priority 0-1): {base_conf}. Treat this as a mild prior -- if the {label} plausibly '
            f'changes a customer\'s urgency or reachability, it is fine to score them differently here.\n'
        )
        if not misfits:
            return (conf_line +
                    f'No customer was flagged as a cluster misfit under base -- treat that as a mild '
                    f'prior toward fewer misfits here too, but still judge this scenario\'s own '
                    f'clusters on their own merits.\n')
        return (
            conf_line +
            f'Under base, these customers were flagged as cluster misfits: {misfits}. If they appear '
            f'in this scenario too, that is corroborating evidence (not proof) for flagging them again '
            f'here; if the {label} plausibly resolves the issue (e.g. it moved them to a better-fitting '
            f'group), it is fine to disagree.\n'
        )

    def _ensure_soft_cluster_cache(self, inst: dict, force_refresh: bool = False):
        """Cluster customers, call LLM per cluster, cache (K, N+1) confidence tensor.

        cluster_mode (trainer_params, default 'rl') selects between two
        deliberate, mutually exclusive pipelines -- never a silent runtime
        fallback between them:
          'rl'      : clusters come from RL-derived routes. If the cache for an
                      instance doesn't exist, that's a real gap (generate it
                      first) -- raises rather than silently substituting a
                      different clustering method.
          'kim2006' : clusters come from live Kim(2006) geographic clustering
                      + an LLM call, the original first-implemented pipeline.
        Mixing the two silently (e.g. falling back to kim2006 because an RL
        cache happened to be missing) is exactly what let a contaminated RL
        cluster cache go undetected in the past -- see git history.
        """
        if self.trainer_params.get('guidance_mode') == 'selective_advance':
            from SelectiveAdvanceLLMModule import ensure_trainer_cache
            return ensure_trainer_cache(self, inst, force_refresh=force_refresh)
        name   = inst['name']
        is_acc = 'ACC' in name.upper()

        if name in self._cluster_conf_cache and not force_refresh:
            return

        lp = self.llm_params
        N  = inst['n_customers']
        cluster_mode = self.trainer_params.get('cluster_mode', 'rl')
        if cluster_mode not in ('rl', 'kim2006'):
            raise ValueError(f"cluster_mode must be 'rl' or 'kim2006', got {cluster_mode!r}")

        # ACC instances start from the BASE instance's own confidence (pre-
        # accident) -- never their own {name}_rl_cluster.json. That file would
        # encode confidence built from a full post-accident episode's outcome,
        # applied from t=0 -- i.e. before any accident has happened yet in THIS
        # episode. Post-accident confidence is instead computed live, at the
        # real per-rollout trigger time, via _refresh_clusters_accident, and
        # blended in per rollout by _acc_conf_now (see self._acc_stage_events).
        # One real LLM call per accident wave, ever -- cached to
        # {name}_acc_refresh_<t_start>.json and reused by every future
        # epoch/run that hits the same wave.
        if is_acc:
            base_key = name.split('_')[0]
            if cluster_mode == 'rl':
                if base_key not in self._cluster_conf_cache:
                    _rl_base = os.path.join(self._llm_cache_dir, f'{base_key}_rl_cluster.json')
                    if not os.path.isfile(_rl_base):
                        raise RuntimeError(
                            f"ACC {name}: base instance cache {base_key}_rl_cluster.json not "
                            f"found in {self._llm_cache_dir} -- generate the base instance's "
                            f"cache first (it never falls back to kim2006 -- see class docstring)."
                        )
                    with open(_rl_base, encoding='utf-8') as f:
                        bd = json.load(f)
                    clusters, flat_conf = _parse_cluster_cache(bd)
                    base_stats = _parse_cluster_stats(bd)
                    base_depot = _parse_cluster_depot(bd)
                    K_actual   = len(clusters)
                    built = self._build_cluster_conf_tensor(
                        clusters, flat_conf, K_actual, N, depot_conf=base_depot)
                    self._cluster_assign_cache[base_key] = clusters
                    self._cluster_conf_cache[base_key]   = built
                    self._cluster_stats_cache[base_key]  = base_stats
                self._cluster_assign_cache[name] = self._cluster_assign_cache[base_key]
                self._cluster_conf_cache[name]   = self._cluster_conf_cache[base_key]
                self._cluster_stats_cache[name]  = self._cluster_stats_cache.get(base_key, {})

                # Precompute this instance's post-accident stage(s) NOW, before any
                # routing/training starts -- not lazily on first decode-time trigger.
                # One real LLM call per wave (accidents sharing an exact t_start are
                # one wave), done here and cached to {name}_acc_refresh_<t_start>.json;
                # decoding later just picks the right stage per rollout (_acc_conf_now).
                accident_list = _get_accidents(inst)
                n_waves = 0
                if accident_list and (name not in self._acc_precompute_done or force_refresh):
                    self._acc_precompute_done.add(name)
                    self._acc_stage_events[name] = []
                    # Waves MUST be processed in chronological (t_start) order --
                    # accident_list's own order is generation/file order, which is NOT
                    # guaranteed to be time order. Processing out of order used to let
                    # a chronologically-LATER wave's stats/confidence leak into an
                    # EARLIER stage (e.g. building the t=173.5 stage AFTER already
                    # having applied the t=174.5 wave to running_stats).
                    #
                    # The CONFIDENCE tensor is still built cumulatively (each wave's
                    # _refresh_clusters_accident call sees the previous wave's
                    # resulting tensor) -- a later wave resetting every OTHER cluster
                    # back to base would silently discard an earlier wave's still-
                    # ongoing effect. Restoring a cluster once ITS OWN accident ends is
                    # a separate, still-unaddressed concern (matches the existing
                    # travel-time restore, which never had an LLM-side counterpart
                    # either) -- this only fixes "don't erase a still-active wave."
                    #
                    # The STATS shown to the LLM, however, are recomputed FRESH from
                    # the untouched base every wave, using node_mult = the MAX
                    # multiplier among every accident affecting that node so far --
                    # matching the env's own combination rule (_event_multiplier_arc/
                    # _row take max(dim=0) across simultaneously active events, never
                    # a product). Stacking multipliers from the previous stage's
                    # already-adjusted numbers (the old behavior) compounded them
                    # multiplicatively instead -- two overlapping 5.88x accidents on
                    # the same node inflated a travel_time of 10 to 345.7 instead of
                    # the env's actual 58.8.
                    base_stats_orig = self._cluster_stats_cache.get(base_key, {})
                    running_tensor  = self._cluster_conf_cache[base_key]
                    wave_t_starts = sorted(set(fp['t_start'] for _, fp in accident_list))
                    for t_start in wave_t_starts:
                        wave = [(a, f) for a, f in accident_list if f['t_start'] == t_start]
                        # node_mult is recomputed FRESH every wave from ONLY the
                        # accidents ACTIVE at THIS wave's own t_start (f['t_start'] <=
                        # t_start < f['t_end']) -- modeling each wave's LLM call as
                        # "news arriving exactly when this wave's accident starts", so
                        # "now" for node_mult purposes IS this wave's own t_start. An
                        # earlier accident that has ALREADY ended by then is stale news
                        # and must not keep inflating this node's estimate -- the
                        # previous version accumulated the max multiplier EVER seen for
                        # a node with no expiry, so a since-ended severe accident kept
                        # contaminating every later wave's stats/prompt numbers forever.
                        node_mult: dict[int, float] = {}
                        for a, f in accident_list:
                            if f['t_start'] <= t_start < f['t_end']:
                                m = float(f['multiplier'])
                                for n in a.affected_nodes:
                                    node_mult[n] = max(node_mult.get(n, 1.0), m)
                        running_stats = self._apply_accident_to_stats(base_stats_orig, node_mult)
                        # _refresh_clusters_accident raises (does not return a
                        # fallback) if this wave can't be scored, which aborts
                        # this whole cache-generation run -- no partial/
                        # unresolved wave is ever cached (see its own docstring).
                        running_tensor = self._refresh_clusters_accident(
                            inst, running_tensor, [a for a, _ in wave], N, t_start,
                            stats=running_stats, force_refresh=force_refresh)
                        self._acc_stage_events[name].append(
                            {'t_start': t_start, 'tensor': running_tensor.clone()})
                    n_waves = len(self._acc_stage_events[name])
                print(f'  [Cluster cache] {name} (rl_acc): using base {base_key} confidence '
                      f'({len(self._cluster_assign_cache[name])} clusters), {n_waves} accident '
                      f'wave(s) precomputed')
                return
            else:  # cluster_mode == 'kim2006'
                # Kim event caches use BASE membership and cumulative wave tensors.
                if not force_refresh:
                    if base_key not in self._cluster_conf_cache:
                        base_disk = os.path.join(self._llm_cache_dir, f'{base_key}_cluster.json')
                        with open(base_disk, encoding='utf-8') as f:
                            bd = json.load(f)
                        clusters_b, scores_b = _parse_cluster_cache(bd)
                        base_tensor = self._build_cluster_conf_tensor(
                            clusters_b, scores_b, len(clusters_b), N, _parse_cluster_depot(bd))
                        self._cluster_assign_cache[base_key] = clusters_b
                        self._cluster_conf_cache[base_key] = base_tensor
                    clusters_b = self._cluster_assign_cache[base_key]
                    stages = []
                    for t_start in sorted({fp['t_start'] for _, fp in _get_accidents(inst)}):
                        path = os.path.join(self._llm_cache_dir,
                                            f'{name}_acc_refresh_{t_start:.6f}.json')
                        with open(path, encoding='utf-8') as f:
                            wave_cache = json.load(f)
                        wave_clusters, wave_scores = _parse_cluster_cache(wave_cache)
                        if wave_clusters != clusters_b:
                            raise ValueError(f'Kim ACC membership mismatch: {path}')
                        depot = _parse_cluster_depot(wave_cache)
                        if set(wave_scores) != {n for ns in clusters_b for n in ns} or set(depot) != set(range(len(clusters_b))):
                            raise ValueError(f'Incomplete Kim ACC cache: {path}')
                        tensor = self._build_cluster_conf_tensor(
                            clusters_b, wave_scores, len(clusters_b), N, depot)
                        stages.append(dict(t_start=t_start, tensor=tensor))
                    self._acc_stage_events[name] = stages
                    self._cluster_assign_cache[name] = clusters_b
                    self._cluster_conf_cache[name] = self._cluster_conf_cache[base_key].clone()
                    self._original_cluster_conf[name] = self._cluster_conf_cache[name].clone()
                    print(f'  [Kim ACC] {name}: loaded {len(stages)} cached waves')
                    return
                _acc_path = os.path.join(self._llm_cache_dir, f'{name}_cluster.json')
                if os.path.isfile(_acc_path) and not force_refresh:
                    with open(_acc_path, encoding='utf-8') as f:
                        cached = json.load(f)
                    clusters, flat_conf = _parse_cluster_cache(cached)
                    K_actual = len(clusters)
                    built = self._build_cluster_conf_tensor(clusters, flat_conf, K_actual, N)
                    self._cluster_assign_cache[name]  = clusters
                    self._cluster_conf_cache[name]    = built
                    self._original_cluster_conf[name] = built.clone()
                    print(f'  [Cluster cache] loaded {name} (kim2006_acc): {K_actual} clusters')
                    return

                if base_key not in self._cluster_assign_cache:
                    base_disk = os.path.join(self._llm_cache_dir, f'{base_key}_cluster.json')
                    if os.path.isfile(base_disk):
                        with open(base_disk, encoding='utf-8') as f:
                            bd = json.load(f)
                        clusters_b, flat_conf_b = _parse_cluster_cache(bd)
                        K_b = len(clusters_b)
                        built_b = self._build_cluster_conf_tensor(clusters_b, flat_conf_b, K_b, N)
                        self._cluster_assign_cache[base_key]  = clusters_b
                        self._cluster_conf_cache[base_key]    = built_b
                        self._original_cluster_conf[base_key] = built_b.clone()
                if base_key not in self._cluster_assign_cache:
                    raise RuntimeError(
                        f'ACC {name}: no Kim(2006) cluster cache found for base {base_key} '
                        f'(cluster_mode=kim2006). It should have been built when the base '
                        f'instance was first processed.'
                    )

            clusters_b  = self._cluster_assign_cache[base_key]
            conf_tensor = self._cluster_conf_cache[base_key]

            # Pre-generate post-accident cache only for training instances.
            # Test instances call LLM in real-time at accident trigger (deployment behavior).
            is_test = name.upper() in self._test_instance_names
            if not is_test and lp.get('enabled', True):
                flat_conf_b = {n: conf_tensor[k, n].item()
                               for k, nodes in enumerate(clusters_b)
                               for n in nodes if 1 <= n <= N}
                self._gen_acc_cache(inst, clusters_b, flat_conf_b, N, name)

            base_conf = self._cluster_conf_cache[base_key]
            self._cluster_assign_cache[name]  = clusters_b
            self._cluster_conf_cache[name]    = base_conf.clone()
            self._original_cluster_conf[name] = base_conf.clone()
            print(f'  [Cache] {name}: reusing {base_key} clusters+confidence (acc pre-generated)')
            return

        # Number of clusters = theoretical minimum vehicles
        K = math.ceil(float(inst['node_demand'].sum().item()))
        K = max(1, K)

        if cluster_mode == 'rl':
            # RL cluster MEMBERSHIP always comes from disk (extract_rl_routes.py,
            # itself derived from a trained checkpoint, not the LLM) -- loading it
            # is never gated by force_refresh, only the LLM-computed CONFIDENCE
            # (fit + priority) is. Conflating the two used to mean force_refresh
            # broke cluster loading entirely (no LLM-based fallback exists for
            # membership), and omitting it silently reused whatever confidence
            # was already on disk -- including extract_rl_routes.py's own 1.0
            # placeholder, which is not real confidence and was never meant to
            # be used as such. Fresh confidence is generated whenever
            # force_refresh is set OR the cached confidence is still that
            # placeholder (i.e. never actually scored by the LLM yet), then
            # written back so later non-refresh runs reuse the real scores.
            _base_key_for_rl = name.split('_')[0]
            _rl_path_inst = os.path.join(self._llm_cache_dir, f'{name}_rl_cluster.json')
            _rl_path_base = os.path.join(self._llm_cache_dir, f'{_base_key_for_rl}_rl_cluster.json')
            _rl_path = _rl_path_inst if os.path.isfile(_rl_path_inst) else _rl_path_base
            if not os.path.isfile(_rl_path):
                raise RuntimeError(
                    f'{name}: no RL cluster cache found at {_rl_path_inst} or {_rl_path_base} '
                    f'(cluster_mode=rl). Generate it first (extract_rl_routes.py).'
                )
            with open(_rl_path, encoding='utf-8') as f:
                cached = json.load(f)
            clusters, flat_conf = _parse_cluster_cache(cached)
            cluster_stats = _parse_cluster_stats(cached)
            depot_conf    = _parse_cluster_depot(cached)
            K_actual = len(clusters)
            src = 'rl_instance' if os.path.isfile(_rl_path_inst) else 'rl_base'

            is_placeholder = all(v == 1.0 for v in flat_conf.values())
            # src == 'rl_base' means THIS instance (e.g. a RAIN2 variant) has no
            # dedicated {name}_rl_cluster.json of its own yet -- every successful
            # scoring attempt below writes one, so never having one only means
            # "never successfully scored" (including a prior run that failed
            # after retries and correctly avoided writing a partial/fallback
            # file). Skipping this case used to silently accept the borrowed
            # BASE confidence as if it were a real, dedicated score for this
            # instance forever, so a once-failed instance would never be
            # retried on a later plain (non---refresh-cache) run even though
            # nothing about it was ever actually resolved.
            needs_scoring = force_refresh or is_placeholder or src == 'rl_base'
            if needs_scoring:
                reason = ('refreshing' if force_refresh else
                          'placeholder confidence, scoring' if is_placeholder else
                          'no dedicated cache yet, scoring')
                print(f'  [Cluster cache] {name} ({src}): {K_actual} clusters -- {reason} via LLM')
                # No fallback: cache generation must fully succeed before any
                # training/eval can start, so this retries generously and then
                # raises rather than silently continuing with placeholder/prior
                # confidence (which used to let a run "succeed" while quietly
                # missing real LLM judgment for some instances). Always
                # temperature=0.0 (see query_llm) -- purely repeated sampling,
                # no escalation, so a demanding combined prompt (e.g. RC103's
                # 21 clusters) gets more independent chances to land a
                # complete response instead of a different sampling strategy.
                _MAX_ATTEMPTS = 20
                fresh = {}
                fresh_depot = {}
                last_err = None
                for _attempt in range(_MAX_ATTEMPTS):
                    try:
                        fresh_depot = {}
                        fresh = get_all_clusters_confidence(
                            inst, clusters,
                            model=lp['model'], use_cot=lp['use_cot'],
                            use_ontology=lp['use_ontology'],
                            use_misfit=lp.get('use_misfit', False),
                            reward_config=lp.get('reward_config', 'F'),
                            stats=cluster_stats,
                            experience_section=self._build_base_prior_ref(name, N),
                            depot_out=fresh_depot,
                        )
                        break
                    except Exception as e:
                        last_err = e
                        more = _attempt + 1 < _MAX_ATTEMPTS
                        print(f'  [Cluster cache] {name}: attempt {_attempt+1}/{_MAX_ATTEMPTS} failed ({e}), '
                              + ('retrying...' if more else 'all attempts exhausted.'))
                if not fresh:
                    raise RuntimeError(
                        f'{name}: LLM cluster scoring failed after {_MAX_ATTEMPTS} attempts '
                        f'({last_err}) -- cache generation cannot proceed with a partial/'
                        f'placeholder result.'
                    )
                flat_conf  = fresh
                depot_conf = fresh_depot
                cached_out = _dump_cluster_cache(clusters, flat_conf)
                if cluster_stats:
                    cached_out['_stats'] = {str(k): v for k, v in cluster_stats.items()}
                if depot_conf:
                    cached_out['_depot'] = {str(k): v for k, v in depot_conf.items()}
                with open(_rl_path_inst, 'w', encoding='utf-8') as f:
                    json.dump(cached_out, f)

            built = self._build_cluster_conf_tensor(clusters, flat_conf, K_actual, N, depot_conf=depot_conf)
            self._cluster_assign_cache[name]  = clusters
            self._cluster_conf_cache[name]    = built
            self._original_cluster_conf[name] = built.clone()
            self._cluster_stats_cache[name]   = cluster_stats
            print(f'  [Cluster cache] loaded {name} ({src}): {K_actual} clusters')
            return

        # cluster_mode == 'kim2006': disk cache (fewshot or plain), else a fresh
        # Kim(2006) clustering + LLM call -- both intentional parts of this mode.
        fewshot_path = os.path.join(self._llm_cache_dir, f'{name}_fewshot_cluster.json')
        cache_path   = os.path.join(self._llm_cache_dir, f'{name}_cluster.json')
        use_fewshot  = lp.get('use_fewshot_cache', False)
        load_path    = fewshot_path if (use_fewshot and os.path.isfile(fewshot_path)) else cache_path
        if os.path.isfile(load_path) and not force_refresh:
            with open(load_path, encoding='utf-8') as f:
                cached = json.load(f)
            clusters, flat_conf = _parse_cluster_cache(cached)
            K_actual = len(clusters)
            built = self._build_cluster_conf_tensor(clusters, flat_conf, K_actual, N,
                                                    _parse_cluster_depot(cached))
            self._cluster_assign_cache[name]  = clusters
            self._cluster_conf_cache[name]    = built
            self._original_cluster_conf[name] = built.clone()
            tag = 'fewshot' if load_path == fewshot_path else 'init'
            print(f'  [Cluster cache] loaded {name} ({tag}): {K_actual} clusters')
            return

        # Fresh clustering + LLM call
        clusters = self._cluster_customers(inst, K)
        K_actual = len(clusters)
        print(f'  [Cluster] {name}: {K_actual} clusters from Kim(2006) (K_init={K})')
        print(f'  [LLM] {name}: single call for all {K_actual} clusters')
        _t_llm = time.time()
        exp_path = os.path.join(self._llm_cache_dir, f'{name}_experience.txt')
        experience_section = ""
        if os.path.isfile(exp_path):
            with open(exp_path, encoding='utf-8') as _f:
                experience_section = _f.read()
            print(f'  [ExperiencePrompt] {name}: loaded episode-based few-shot examples')

        # For RAIN instances: pass base confidence as prior reference (step-by-step design).
        # Base scores are computed first; rain LLM adjusts from that baseline.
        prior_ref = ""
        is_rain = 'RAIN' in name.upper()
        if is_rain:
            base_key = name.split('_')[0]
            base_tensor = self._cluster_conf_cache.get(base_key)
            if base_tensor is None:
                base_disk = os.path.join(self._llm_cache_dir, f'{base_key}_cluster.json')
                if os.path.isfile(base_disk):
                    with open(base_disk, encoding='utf-8') as f:
                        _bd = json.load(f)
                    _bc, _bf = _parse_cluster_cache(_bd)
                    base_tensor = self._build_cluster_conf_tensor(_bc, _bf, len(_bc), N)
            if base_tensor is not None:
                base_clusters = self._cluster_assign_cache.get(base_key, clusters)
                prior_lines = []
                for k, cluster_nodes in enumerate(base_clusters):
                    row = sorted(
                        [n for n in cluster_nodes if 1 <= n <= N],
                        key=lambda n: -float(base_tensor[k, n]),
                    )
                    if row:
                        scores_str = ', '.join(f'node {n}: {float(base_tensor[k, n]):.2f}' for n in row)
                        prior_lines.append(f'  Vehicle {k + 1}: {scores_str}')
                if prior_lines:
                    prior_ref = (
                        '[Base Confidence (clear weather, before rain adjustment)]\n'
                        + '\n'.join(prior_lines) + '\n'
                    )
                    print(f'  [RainPrior] {name}: base confidence reference ready ({len(prior_lines)} vehicles)')

        combined_experience = "\n\n".join(p for p in [prior_ref, experience_section] if p)
        # BKS few-shot: passed separately so it appears before the actual instance data.
        # Only used when no episode experience is available yet.
        bks_fewshot = self._bks_fewshot_block if not experience_section else ""

        _succeeded = False
        for _attempt in range(2):
            try:
                flat_conf: dict[int, float] = get_all_clusters_confidence(
                    inst, clusters,
                    model=lp['model'], use_cot=lp['use_cot'],
                    use_ontology=lp['use_ontology'],
                    use_misfit=lp.get('use_misfit', False),
                    experience_section=combined_experience,
                    reward_config=lp.get('reward_config', 'F'),
                    bks_fewshot=bks_fewshot,
                    cluster_source='kim',
                )
                _succeeded = True
                break
            except Exception as e:
                print(f'  [LLM] {name}: attempt {_attempt+1}/2 failed ({e}), '
                      + ('retrying...' if _attempt == 0 else 'queuing for retry after all instances.'))
                if _attempt == 0:
                    time.sleep(10)
        if not _succeeded:
            self._pending_cache_retry.append((inst, clusters, K_actual, combined_experience, bks_fewshot))
            self._timing['llm_init'] += time.time() - _t_llm
            return
        self._timing['llm_init'] += time.time() - _t_llm

        self._cluster_assign_cache[name] = clusters
        built = self._build_cluster_conf_tensor(clusters, flat_conf, K_actual, N)
        self._cluster_conf_cache[name]    = built
        self._original_cluster_conf[name] = built.clone()

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(_dump_cluster_cache(clusters, flat_conf), f)
        print(f'  [Cluster cache] saved {name}: {K_actual} clusters')

    def _flush_pending_caches(self):
        """Retry instances that failed during initial LLM cache generation."""
        if not self._pending_cache_retry:
            return
        pending = list(self._pending_cache_retry)
        self._pending_cache_retry.clear()
        print(f'[SoftCluster] Retrying {len(pending)} deferred LLM cache(s)...')
        for inst, clusters, K_actual, combined_experience, bks_fewshot in pending:
            name = inst['name']
            lp = self.llm_params
            _t_llm = time.time()
            try:
                flat_conf: dict[int, float] = get_all_clusters_confidence(
                    inst, clusters,
                    model=lp['model'], use_cot=lp['use_cot'],
                    use_ontology=lp['use_ontology'],
                    use_misfit=lp.get('use_misfit', False),
                    experience_section=combined_experience,
                    reward_config=lp.get('reward_config', 'F'),
                    bks_fewshot=bks_fewshot,
                    cluster_source='kim',
                )
            except Exception as e:
                print(f'  [LLM] {name}: retry also failed ({e}), skipping.')
                self._timing['llm_init'] += time.time() - _t_llm
                continue
            self._timing['llm_init'] += time.time() - _t_llm
            N = inst['dimension'] - 1
            cache_path = os.path.join(
                self.trainer_params.get('result_dir', '.'),
                'llm_cache', f'{name}_cluster.json',
            )
            self._cluster_assign_cache[name] = clusters
            built = self._build_cluster_conf_tensor(clusters, flat_conf, K_actual, N)
            self._cluster_conf_cache[name]    = built
            self._original_cluster_conf[name] = built.clone()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(_dump_cluster_cache(clusters, flat_conf), f)
            print(f'  [Cluster cache] retry saved {name}: {K_actual} clusters')

    def _gen_acc_cache(self, inst, clusters, base_flat, N, name):
        """Pre-generate post-accident LLM scores at accident trigger time (init-time, warm LLM).

        Self-contained: computes the trigger time and cache path itself and skips if
        that exact wave was already cached, so distinct accident waves (different
        trigger times) never collide on (overwrite) each other's cache file.
        """
        accident_list = _get_accidents(inst)
        if not accident_list:
            return
        lp    = self.llm_params
        K     = len(clusters)
        cur_t = min(fp['t_start'] for _, fp in accident_list)
        cache_path = os.path.join(self._llm_cache_dir, f'{name}_acc_refresh_{cur_t:.6f}.json')
        if os.path.isfile(cache_path):
            return
        acc_set: set[int] = set()
        for acc_ev, _ in accident_list:
            acc_set.update([acc_ev.node_a, acc_ev.node_b] + list(acc_ev.affected_nodes))

        full_flat = dict(base_flat)
        affected_clusters = [nodes for nodes in clusters if any(n in acc_set for n in nodes)]
        if affected_clusters:
            _t = time.time()
            _scores = None
            for _attempt in range(2):
                try:
                    _scores = get_all_clusters_confidence(
                        inst, affected_clusters,
                        model=lp['model'], use_cot=lp['use_cot'], use_ontology=lp['use_ontology'],
                        use_misfit=lp.get('use_misfit', False),
                        cur_time=cur_t,
                        reward_config=lp.get('reward_config', 'F'),
                        cluster_source='kim',
                    )
                    break
                except Exception as e:
                    print(f'  [Cluster cache] ACC {name} attempt {_attempt+1}/2 failed ({e}), '
                          + ('retrying...' if _attempt == 0 else 'skipping ACC cache.'), flush=True)
                    if _attempt == 0:
                        time.sleep(10)
            self._timing['llm_init'] += time.time() - _t
            if _scores is not None:
                for n, s in _scores.items():
                    if 1 <= n <= N:
                        full_flat[n] = float(s)

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(_dump_cluster_cache(clusters, full_flat), f)
        print(f'  [Cluster cache] pre-generated ACC {name}', flush=True)

    # ------------------------------------------------------------------
    # Experience-based prompt refresh (mid-training)
    # ------------------------------------------------------------------

    def _refresh_experience_caches(self, epoch: int):
        """At a curriculum point, rebuild LLM caches with episode-derived few-shot examples.

        Two-phase:
        1. Build experience text per instance (cross-reference init confidence × empirical late_rate).
        2. Force-rebuild cluster JSON for non-ACC instances with the experience text included
           in the LLM prompt, so future confidence biases reflect accumulated episode knowledge.
        """
        from SoftClusterLLMModule import build_experience_examples
        lp = self.llm_params
        print(f'\n[ExperienceRefresh] Epoch {epoch}: generating episode-based few-shot examples...')

        tracker_path = os.path.join(self._llm_cache_dir, 'episode_tracker.json')
        episode_tracker.save_to_disk(tracker_path)
        print(f'[ExperienceRefresh] episode_tracker saved → {tracker_path}')

        refreshed = 0
        for inst in self.instance_pool:
            name     = inst['name']
            clusters = self._cluster_assign_cache.get(name)
            conf_t   = self._original_cluster_conf.get(name)
            if clusters is None or conf_t is None:
                continue

            exp_text = build_experience_examples(inst, clusters, conf_t, name)
            if not exp_text:
                print(f'  [{name}] no episode data — skipping')
                continue

            exp_path = os.path.join(self._llm_cache_dir, f'{name}_experience.txt')
            with open(exp_path, 'w', encoding='utf-8') as f:
                f.write(exp_text)
            print(f'  [{name}] experience saved')
            refreshed += 1

        print(f'[ExperienceRefresh] Done — {refreshed} experience text(s) saved'
              f' (cluster JSONs unchanged; run build_fewshot_cache.py to generate fewshot caches)')

    # ------------------------------------------------------------------
    # RL-guided re-clustering
    # ------------------------------------------------------------------

    def _refresh_clusters_from_rl_routes(self, epoch: int):
        """Replace cluster assignments with RL best-episode routes and re-run LLM.

        Iterative improvement loop:
          LLM init cluster → RL training → RL discovers better groupings
          → LLM re-evaluates those groupings (TW/event reasoning)
          → RL trains again with refined clusters

        Routes from episode_tracker (already stored as list[list[int]]) map directly
        to the clusters format used by get_all_clusters_confidence.
        """
        from SoftClusterLLMModule import get_all_clusters_confidence
        from SoftClusterOntology import episode_tracker

        lp = self.llm_params
        print(f'\n[RLRecluster] Epoch {epoch}: re-clustering from RL best-episode routes...')

        refreshed = 0
        for inst in self.instance_pool:
            name = inst['name']
            N    = inst['n_customers']

            # ACC instances reuse base clusters — skip
            if 'ACC' in name.upper():
                continue

            best_eps = episode_tracker.best_episodes(name, k=1)
            if not best_eps or not best_eps[0].get('routes'):
                print(f'  [{name}] no route data yet — skipping')
                continue

            best_ep = best_eps[0]
            # Convert routes to cluster format (filter depot node 0 and empty routes)
            new_clusters = [
                [n for n in route if n != 0]
                for route in best_ep['routes']
                if any(n != 0 for n in route)
            ]
            if not new_clusters:
                print(f'  [{name}] empty routes — skipping')
                continue

            K_new = len(new_clusters)
            print(f'  [{name}] {K_new} vehicles from RL ep#{best_ep["ep"]}'
                  f' (reward={best_ep["reward"]:.3f}, Lc={best_ep["Lc"]})')

            # Load experience text if available (used as additional context)
            exp_path = os.path.join(self._llm_cache_dir, f'{name}_experience.txt')
            experience_section = ""
            if os.path.isfile(exp_path):
                with open(exp_path, encoding='utf-8') as f:
                    experience_section = f.read()

            _t = time.time()
            flat_conf = None
            for _attempt in range(2):
                try:
                    flat_conf = get_all_clusters_confidence(
                        inst, new_clusters,
                        model=lp['model'], use_cot=lp['use_cot'],
                        use_ontology=lp['use_ontology'],
                        use_misfit=lp.get('use_misfit', False),
                        experience_section=experience_section,
                        reward_config=lp.get('reward_config', 'F'),
                    )
                    break
                except Exception as e:
                    print(f'  [{name}] LLM call attempt {_attempt+1}/2 failed ({e}), '
                          + ('retrying...' if _attempt == 0 else 'skipping refresh.'))
                    if _attempt == 0:
                        time.sleep(10)
            self._timing['llm_refresh'] += time.time() - _t
            if flat_conf is None:
                continue

            built = self._build_cluster_conf_tensor(new_clusters, flat_conf, K_new, N)
            self._cluster_assign_cache[name]  = new_clusters
            self._cluster_conf_cache[name]    = built
            self._original_cluster_conf[name] = built.clone()

            # Save as separate file to keep original init clusters for reference
            rl_cache_path = os.path.join(self._llm_cache_dir, f'{name}_rl_cluster.json')
            with open(rl_cache_path, 'w', encoding='utf-8') as f:
                json.dump(_dump_cluster_cache(new_clusters, flat_conf), f)
            print(f'  [{name}] saved → {name}_rl_cluster.json')
            refreshed += 1

        print(f'[RLRecluster] Done — {refreshed} instance(s) re-clustered from RL routes')

    # ------------------------------------------------------------------
    # Deterministic starts — top-confidence node per cluster
    # ------------------------------------------------------------------

    def _det_starts(self, inst: dict, pomo_size: int) -> torch.Tensor:
        """Return (1, pomo_size): rollout p starts inside cluster (p % K), using the
        (p // K)-th best-confidence node of that cluster.

        This chooses confidence-ranked starting customers across clusters.
        Bias assignment is independently anchored to the actual first customer
        by _active_route_cluster, including when short clusters require padding.
        """
        name     = inst['name']
        clusters = self._cluster_assign_cache.get(name, [])
        conf_t   = self._cluster_conf_cache.get(name)   # (K, N+1) or None
        K = len(clusters)

        if K == 0:
            N   = inst['n_customers']
            pad = list(range(1, N + 1))
            random.shuffle(pad)
            return torch.tensor(pad[:pomo_size], dtype=torch.long,
                                device=self.device).unsqueeze(0)

        ranked: list[list[int]] = []
        for k, nodes in enumerate(clusters):
            if conf_t is not None:
                ranked.append(sorted(nodes, key=lambda n: -conf_t[k, n].item()))
            else:
                ranked.append(list(nodes))

        starts = []
        used   = set()
        for p in range(pomo_size):
            k, rank = p % K, p // K
            pool = ranked[k]
            pick = pool[rank] if rank < len(pool) and pool[rank] not in used else None
            if pick is None:
                pick = next((n for n in pool if n not in used), None)
            if pick is not None:
                starts.append(pick)
                used.add(pick)

        # Pad with random unassigned nodes if still short (e.g. tiny clusters)
        if len(starts) < pomo_size:
            N   = inst['n_customers']
            pad = [n for n in range(1, N + 1) if n not in used]
            random.shuffle(pad)
            starts.extend(pad[:pomo_size - len(starts)])

        return torch.tensor(starts, dtype=torch.long, device=self.device).unsqueeze(0)

    # ------------------------------------------------------------------
    # Shared per-rollout confidence-bias computation (used by _train_one_batch,
    # _eval_test, _best_solution -- factored out so the accident per-rollout
    # blend below only has to be written once).
    # ------------------------------------------------------------------
    def _conf_now(self, cluster_conf, bias_mode, K_clusters, roll_shifts,
                   start_mode, current_cluster, veh_idx, fresh_start, bias_strength):
        """conf_now for ONE specific (K, N+1) confidence tensor -- the same
        bias_mode/start_mode branching every call site already had inline."""
        if bias_mode == 'node':
            node_conf = cluster_conf.sum(dim=0)
            mask      = (veh_idx < K_clusters).float().unsqueeze(-1)
            return node_conf[None, None, :] * mask * bias_strength
        elif current_cluster is not None:
            mask = (~fresh_start).float().unsqueeze(-1)
            return cluster_conf[current_cluster] * mask * bias_strength
        else:
            cluster_idx = (veh_idx + roll_shifts[None, :]) % K_clusters
            mask        = (veh_idx < K_clusters).float().unsqueeze(-1)
            return cluster_conf[cluster_idx] * mask * bias_strength

    def _active_route_cluster(self, name):
        """Anchor bias to the first customer since the most recent depot visit.

        Use actual route history so forced starts, free starts and every eval
        path have identical semantics. Keep the anchor through depot return:
        visited customers are masked by the environment, but depot retains
        its confidence even after every customer in the group was visited.
        """
        history = self.env.selected_node_list
        membership = torch.full(
            (self.env.problem_size + 1,), -1, dtype=torch.long, device=history.device)
        for k, nodes in enumerate(self._cluster_assign_cache.get(name, [])):
            membership[nodes] = k
        positions = torch.arange(history.size(-1), device=history.device)
        last_depot = torch.where(history == 0, positions, -1).amax(dim=-1)
        first_pos = (last_depot + 1).clamp(max=history.size(-1) - 1)
        first_node = history.gather(-1, first_pos.unsqueeze(-1)).squeeze(-1)
        active = membership[first_node]
        enabled = (active >= 0) & (history[:, :, -1] != 0)
        return active.clamp(min=0), enabled

    def _acc_conf_now(self, name, base_conf, bias_mode, K_clusters, roll_shifts,
                       start_mode, current_cluster, veh_idx, fresh_start, bias_strength,
                       current_time):
        """conf_now blended per rollout across this instance's accident stages
        (self._acc_stage_events), using each rollout's OWN current_time -- never
        a single batch-wide min/max swap applied to every rollout at once
        regardless of its own progress (that was the bug: a fast rollout stayed
        "unaware" until the slowest one caught up, and vice versa for restore).
        Falls back to plain base-tensor conf_now when this instance has no
        accident or none has triggered for ANY rollout yet (the common case
        pays no extra cost beyond the one dict lookup)."""
        if bias_mode == 'cluster':
            current_cluster, enabled = self._active_route_cluster(name)
            # Keep depot guidance until return; never reanchor on an absorbed stop.
            fresh_start = ~enabled
        conf_now = self._conf_now(base_conf, bias_mode, K_clusters, roll_shifts,
                                   start_mode, current_cluster, veh_idx, fresh_start, bias_strength)
        stages = self._acc_stage_events.get(name)
        if not stages:
            return conf_now
        for stage in stages:   # chronological trigger order -- later stages may
            hit = current_time >= stage['t_start']            # (B, P)   override earlier ones for rollouts that reached both
            if not hit.any():
                continue
            stage_now = self._conf_now(stage['tensor'], bias_mode, K_clusters, roll_shifts,
                                        start_mode, current_cluster, veh_idx, fresh_start, bias_strength)
            conf_now = torch.where(hit.unsqueeze(-1), stage_now, conf_now)
        return conf_now

    # ------------------------------------------------------------------
    # Cluster confidence re-scoring after accident
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_accident_to_stats(base_stats: dict, node_mult: dict) -> dict:
        """Copy of base_stats with each node in node_mult's OWN (per-stop)
        travel_time scaled by its multiplier and slack reduced by that same
        delta -- applied FRESH from base_stats every call, never stacked onto
        a previously-adjusted copy. node_mult must hold, for every node
        touched by ANY accident so far (not just the current wave), the MAX
        multiplier among all of them -- this matches the env's own combination
        rule (_event_multiplier_arc/_row in vrptw_env.py combine simultaneously
        active events via max(dim=0), never a product). Applying multipliers
        sequentially onto an already-adjusted copy compounds them
        multiplicatively instead: two overlapping 5.88x accidents on the same
        node would inflate a travel_time of 10 to 345.7, when the env's actual
        rule produces 58.8.

        This is called ahead of any routing (see is_acc precompute in
        _ensure_soft_cluster_cache), so which nodes end up actually visited
        near the accident, and in what order, isn't known yet -- this only
        tightens the ONE number computable without guessing a route (an
        affected node's own leg), it does NOT propagate the delay downstream
        to later stops. The disruption-section prompt text already tells the
        LLM shown slack may still be optimistic for nearby stops for this
        reason."""
        import copy
        adjusted = copy.deepcopy(base_stats)
        for cstat in adjusted.values():
            node_detail = cstat.get('node_detail', {})
            tt_delta_sum = 0.0
            for n_str, det in node_detail.items():
                n = int(n_str)
                if n in node_mult:
                    old_tt = det['travel_time']
                    new_tt = old_tt * node_mult[n]
                    det['travel_time'] = new_tt
                    det['slack']       = det['slack'] - (new_tt - old_tt)
                    tt_delta_sum      += (new_tt - old_tt)
            if tt_delta_sum:
                # Keep the vehicle-level aggregate consistent with the per-node
                # adjustment above -- previously only node_detail changed, so a
                # customer could show negative slack while the SAME vehicle's
                # late_count/travel_time in "[Actual performance]" still read as
                # if nothing were wrong. late_count/total_late are recomputed
                # from the (now-adjusted) node_detail rather than incrementally,
                # since a node whose slack was already negative before this
                # adjustment must not be double-counted.
                cstat['travel_time'] = cstat.get('travel_time', 0.0) + tt_delta_sum
                cstat['late_count']  = sum(1 for d in node_detail.values() if d['slack'] < 0)
                cstat['total_late']  = sum(-d['slack'] for d in node_detail.values() if d['slack'] < 0)
        return adjusted

    def _refresh_clusters_accident(
        self,
        inst: dict,
        cluster_conf: torch.Tensor,
        accidents,          # AccidentEvent or list[AccidentEvent]
        N: int,
        wave_t_start: float,
        stats: dict | None = None,
        force_refresh: bool = False,
    ) -> torch.Tensor:
        """Re-score clusters containing accident-affected nodes; return the
        resulting post-accident tensor, which the caller appends as a new
        stage in self._acc_stage_events -- this function never mutates the
        shared BASE tensor callers pass as cluster_conf, it only clones from
        it. Raises (does not return a partial/fallback result) if the LLM
        call is needed but never succeeds after retries -- see the no-fallback
        note above the retry loop below.

        Accepts a single AccidentEvent or a list (for simultaneous-trigger waves).
        All affected nodes are merged into one set so one LLM call covers the wave.

        wave_t_start: the wave's own declared trigger time (from accident_list's
        fp['t_start'], not the batch's current simulated time) -- used as the
        refresh-cache filename key so the SAME wave hits the SAME cache file
        across different runs/epochs regardless of which rollouts happen to be
        in the batch (current_time.mean() varied run-to-run, making the cache
        miss almost every time and defeating the "one real LLM call per
        accident, ever" cost-saving this cache exists for).

        stats: optional {cluster_idx: {...}} block to show the LLM as this
        wave's '[Actual performance]' reference (see _apply_accident_to_stats).
        Omitted for the live in-episode call path (no route-agnostic stats to
        adjust there beyond what the disruption-section text already covers).
        """
        lp       = self.llm_params
        clusters = self._cluster_assign_cache.get(inst['name'], [])
        K        = len(clusters)
        if not isinstance(accidents, list):
            accidents = [accidents]
        acc_set = set()
        for a in accidents:
            acc_set.update([a.node_a, a.node_b] + list(a.affected_nodes))
        # NOTE: previously self._get_visited_set(N), which hardcodes POMO index 0 --
        # fine when all rollouts follow a similar-paced schedule (start_mode='det_conf'),
        # but with start_mode='random' rollout 0 is just an arbitrary one of many
        # uncorrelated starts and may already be fully done while the group mean is
        # only just crossing the trigger time, making every affected node look
        # "visited" and silently short-circuiting the whole refresh (no LLM call, no
        # error, no log line). Same fix as the staged-cache design: since progress is
        # genuinely rollout-dependent here, don't try to pick a single "representative"
        # rollout at all -- score the full cluster membership; the model's own
        # ninf_mask already makes bias for any given rollout's already-visited nodes
        # moot regardless of what this function assumes (see VRPTWLLMModule score
        # order: ninf_mask applied before llm_bias is added).
        visited = set()
        # The wave's own declared trigger time IS "what time is it when this
        # accident starts" -- more meaningful for the prompt's context than a
        # rollout-batch mean, and (unlike self.env.current_time) available even
        # when called ahead of time during cache-gen, before any env state exists.
        cur_t   = wave_t_start
        updated = cluster_conf.clone()

        # Filename keyed on the wave's own declared trigger time (deterministic,
        # same every run) so distinct accident waves within the same instance
        # never share a refresh cache, AND the same wave reuses the same cache
        # across runs instead of recomputing every time (see wave_t_start note above).
        cache_path = os.path.join(
            self._llm_cache_dir, f'{inst["name"]}_acc_refresh_{wave_t_start:.6f}.json')

        if os.path.isfile(cache_path) and not force_refresh:
            # Post-accident scores pre-cached — load and apply visited mask
            with open(cache_path, encoding='utf-8') as f:
                cached_acc = json.load(f)
            _, full_flat = _parse_cluster_cache(cached_acc)
            depot_flat   = _parse_cluster_depot(cached_acc)
            for k, cluster_nodes in enumerate(clusters):
                for n in cluster_nodes:
                    if 1 <= n <= N:
                        s = full_flat.get(n, 0.0)
                        updated[k, n] = 0.0 if n in visited else float(s)
                if k in depot_flat:
                    updated[k, 0] = depot_flat[k]
            return updated

        # First time: call LLM once for all affected clusters, save full scores.
        # depot_flat is seeded from the INPUT cluster_conf for every cluster, not
        # just left empty -- cluster_conf here is the CUMULATIVE running tensor
        # (see is_acc precompute), so an earlier wave's depot for clusters THIS
        # wave doesn't touch would otherwise be silently absent from the saved
        # cache's "_depot" (unlike full_flat, which was already fully seeded
        # this way) -- reloading that file later would then read those clusters'
        # depot back as the tensor's un-set default (0.0), not the earlier
        # wave's real value.
        full_flat: dict[int, float] = {}
        depot_flat: dict[int, float] = {}
        for k, cluster_nodes in enumerate(clusters):
            for n in cluster_nodes:
                if 1 <= n <= N:
                    full_flat[n] = float(cluster_conf[k, n])
            depot_flat[k] = float(cluster_conf[k, 0])

        # Keep the ORIGINAL cluster index alongside each affected cluster's node
        # list -- get_all_clusters_confidence/build_all_clusters_prompt number
        # vehicles 1..len(clusters passed in), so if we passed just the affected
        # subset with its ORIGINAL indices stripped, "vehicle 1" in the prompt
        # would silently mean whichever cluster happens to be first in acc_set
        # order, while any `stats` we pass alongside would still be keyed by the
        # ORIGINAL full-instance index -- e.g. only original cluster 3 affected
        # would show as "vehicle 1" with cluster 0's stats attached. Remap stats
        # (and the prior-confidence reference's vehicle numbers) to match.
        affected_indices  = [k for k, nodes in enumerate(clusters) if any(n in acc_set for n in nodes)]
        affected_clusters = [clusters[k] for k in affected_indices]
        remapped_stats = (
            {new_k: stats[old_k] for new_k, old_k in enumerate(affected_indices) if old_k in stats}
            if stats else None
        )
        if affected_clusters:
            name = inst['name']
            cnt  = self._acc_prompt_counter.get(name, 0)
            self._acc_prompt_counter[name] = cnt + 1
            prompt_path = os.path.join(self._llm_cache_dir, f'{name}_acc_prompt_{cnt}.txt')

            # Prior confidence reference for remaining nodes (same spirit as rain
            # baseline) -- numbered to match affected_clusters' remapped "vehicle
            # 1..len(affected_clusters)", not the original full-instance index.
            prior_lines = []
            for new_k, old_k in enumerate(affected_indices):
                cluster_nodes = clusters[old_k]
                remaining = sorted(
                    [n for n in cluster_nodes if n not in visited and 1 <= n <= N],
                    key=lambda n: -float(cluster_conf[old_k, n]),
                )
                if remaining:
                    scores_str = ', '.join(f'node {n}: {float(cluster_conf[old_k, n]):.2f}' for n in remaining)
                    prior_lines.append(f'  Vehicle {new_k + 1}: {scores_str}')
            prior_ref = (
                '[Prior Confidence Before Accident (remaining stops only)]\n'
                + '\n'.join(prior_lines) + '\n'
            ) if prior_lines else ''
            # This stats block (if any) is an ESTIMATE, not a replayed route: it
            # only adjusts affected nodes' OWN travel_time/slack against the BASE
            # plan (see _apply_accident_to_stats) and does not know which nodes
            # any given rollout will actually have reached by wave_t_start.
            if remapped_stats:
                prior_ref = (
                    "[Note] The '[Actual performance]' line below is an ESTIMATE: "
                    "BASE-route travel_time/slack with directly affected stops' own "
                    "leg adjusted for this accident, NOT a re-simulated route -- "
                    "downstream stops after an affected leg are not adjusted.\n"
                    + prior_ref
                )

            # No fallback: a wave that can't be scored must stop the whole
            # cache-generation run (see query_llm's own no-fallback change and
            # the base 'rl_instance' branch above) rather than silently keep
            # the pre-wave confidence and let a later run trust an incomplete
            # cache as if it were final.
            expected_nodes = {n for nodes in affected_clusters for n in nodes if 1 <= n <= N}
            _MAX_ATTEMPTS = 5
            scores, depot_by_cluster, resp_ok, last_err = {}, {}, False, None
            for _attempt in range(_MAX_ATTEMPTS):
                try:
                    depot_by_cluster = {}
                    scores = get_all_clusters_confidence(
                        inst, affected_clusters,
                        model=lp['model'], use_cot=lp['use_cot'], use_ontology=lp['use_ontology'],
                        use_misfit=lp.get('use_misfit', False),
                        cur_time=cur_t,
                        prompt_save_path=prompt_path,
                        reward_config=lp.get('reward_config', 'F'),
                        visited=visited,
                        experience_section=prior_ref,
                        stats=remapped_stats,
                        depot_out=depot_by_cluster,
                        events_visible_until=wave_t_start,
                        stats_is_estimate=bool(remapped_stats),
                    )
                    # Completeness check: a cluster with only SOME of its members
                    # scored (or a missing depot score) previously looked
                    # identical to "the LLM scored it near zero" -- both silently
                    # became a 0.0 default. Treat incomplete as a failure and retry.
                    missing_nodes = expected_nodes - scores.keys()
                    missing_depot = set(range(len(affected_clusters))) - depot_by_cluster.keys()
                    if missing_nodes or missing_depot:
                        raise RuntimeError(
                            f"incomplete response: {len(missing_nodes)} customer(s) and "
                            f"{len(missing_depot)} depot score(s) missing")
                    resp_ok = True
                    break
                except Exception as e:
                    last_err = e
                    more = _attempt + 1 < _MAX_ATTEMPTS
                    print(f'  [AccRefresh] {name}: attempt {_attempt+1}/{_MAX_ATTEMPTS} failed ({e}), '
                          + ('retrying...' if more else 'all attempts exhausted.'))
            if not resp_ok:
                raise RuntimeError(
                    f'{name}: accident wave t={wave_t_start:.4f} LLM scoring failed after '
                    f'{_MAX_ATTEMPTS} attempts ({last_err}) -- cache generation cannot '
                    f'proceed with a partial/prior-state result.'
                )

            for n, s in scores.items():
                if 1 <= n <= N:
                    full_flat[n] = float(s)
            for new_k, old_k in enumerate(affected_indices):
                for n in clusters[old_k]:
                    if 1 <= n <= N and n in scores:
                        updated[old_k, n] = 0.0 if n in visited else float(scores[n])
                if new_k in depot_by_cluster:
                    depot_score = depot_by_cluster[new_k]
                    depot_flat[old_k]  = depot_score
                    updated[old_k, 0]  = depot_score
                    # Same below-depot -> MISFIT_BIAS cutoff _build_cluster_conf_tensor
                    # applies for the base tensor -- the prompt tells the LLM "scored
                    # below '0' is treated exactly like a misfit", so the accident
                    # refresh path has to actually enforce that too, not just record
                    # depot's own number.
                    for n in clusters[old_k]:
                        if 1 <= n <= N and MISFIT_BIAS + 1e-9 < updated[old_k, n] < depot_score:
                            updated[old_k, n] = MISFIT_BIAS
                            full_flat[n] = MISFIT_BIAS
            print(f"  [LLM:acc] {name} re-scored in {len(affected_clusters)} cluster(s), "
                  f"prompt→{prompt_path}")

        # affected_clusters empty is a legitimate no-op (this wave doesn't touch
        # this instance); any real scoring attempt above either succeeded or
        # raised, so reaching here always means a genuine, complete result.
        cached_out = _dump_cluster_cache(clusters, full_flat)
        if depot_flat:
            cached_out['_depot'] = {str(k): v for k, v in depot_flat.items()}
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cached_out, f)
        print(f"  [LLM:acc] {inst['name']} post-accident cache saved")
        return updated

    def _get_visited_set(self, N: int) -> set:
        """Visited customer indices from representative rollout (b=0, p=0)."""
        visited = (self.env.visited_ninf_flag[0, 0, 1:] == float('-inf')).cpu()
        return {c for c in range(1, N + 1) if visited[c - 1].item()}

    # ------------------------------------------------------------------
    # Training — per-vehicle per-step cluster confidence bias
    # ------------------------------------------------------------------

    def _train_one_batch(self, inst: dict, batch_size: int):
        self.model.train()
        lp     = self.llm_params
        llm_on = lp['enabled']

        _, _, rain_evs = _get_rain_nodes_mult_evs(inst)
        no_init_llm = lp.get('no_init_llm', False)
        if llm_on and not no_init_llm:
            self._ensure_soft_cluster_cache(inst)
        accident_list = _get_accidents(inst) if llm_on else []
        _t_rl = time.time()
        N = self.env.problem_size

        # pomo_size is forced identical across ALL methods (--pomo, mandatory) so
        # rollout-count never confounds the LLM-bias comparison. K_clusters (the
        # modulus for cyclic vehicle->cluster assignment) is independent of it --
        # when pomo_size > K_clusters, multiple rollouts share a cluster and are
        # given different start nodes within it (see _det_starts).
        pomo_size_eff = self.trainer_params['pomo_size']
        self.env.pomo_size = pomo_size_eff
        if llm_on:
            clusters   = self._cluster_assign_cache.get(inst['name'], [])
            K_clusters = len(clusters) or 1
        else:
            K_clusters = pomo_size_eff

        batch = make_batch(inst, batch_size, self.device)
        self.env.load_problems(batch)

        reset_state, _, _ = self.env.reset()
        reset_state.rain_tokens = _make_rain_tokens(inst, rain_evs, batch_size, self.device)
        reset_state.acc_tokens  = _make_acc_tokens([], N, batch_size, self.device)
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(batch_size, pomo_size_eff, 0, device=self.device)
        state, reward, done = self.env.pre_step()

        # Step 0: depot (always)
        selected, prob = self.model(state)
        state, reward, done = self.env.step(selected)
        prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Cluster confidence tensor (K, N+1); None if LLM off
        cluster_conf  = self._cluster_conf_cache.get(inst['name']) if llm_on else None
        bias_strength = lp.get('bias_strength', 5.0)
        # Cyclic roll: rollout i's vehicle j uses cluster (i+j) % K_clusters
        # so rollout i starts at cluster (i % K_clusters) and follows that cyclic
        # assignment. When pomo_size > K_clusters, multiple rollouts share a
        # cluster (see _det_starts for how their start nodes are differentiated).
        roll_shifts = torch.arange(pomo_size_eff, device=self.device)   # (P,)

        max_steps        = self.trainer_params.get('max_steps', 600)

        # bias_mode='cluster' (default): deterministic per-cluster starts + cyclic
        #   vehicle->cluster assignment that varies by rollout index (roll_shifts) --
        #   this is what created the POMO-baseline confound (see design discussion:
        #   different rollouts get structurally different, unequal-difficulty bias
        #   schedules, so their reward differences aren't purely policy-attributable).
        # bias_mode='node' : free/diverse starts exactly like pure RL (no _det_starts,
        #   no cluster/vehicle bookkeeping at all) + a FLAT per-node confidence bias
        #   (same value for every rollout, every vehicle, every step) built once from
        #   cluster_conf by collapsing the (K, N+1) tensor to (N+1,) -- each node
        #   belongs to exactly one cluster so this is a lossless flatten. Removes the
        #   confound entirely; confidence becomes a pure per-node priority hint.
        bias_mode = self.trainer_params.get('bias_mode', 'cluster')

        # start_mode='random' (bias_mode='cluster' only): replace the deterministic
        # confidence-driven _det_starts with pure-RL's OWN random-distinct-start
        # mechanism (see train_vrptw.py _train_one_batch: random.sample over all N
        # customers) -- vehicle 1's cluster becomes "whichever cluster contains this
        # rollout's own randomly-sampled start node", so the per-rollout offset into
        # the cluster cycle is now DERIVED FROM the same fair diversity source pure
        # RL's own advantage comparison already relies on, instead of an arbitrary
        # rollout-index shift (roll_shifts) unrelated to what the rollout actually
        # visits. Vehicle 2+ still cycle (offset + j) % K_clusters exactly as before
        # -- only the anchor changes, the gating/lookup logic is untouched.
        start_mode = self.trainer_params.get('start_mode', 'det_conf')
        if start_mode not in ('det_conf', 'random', 'confidence_topk'):
            raise ValueError(f"start_mode must be 'det_conf', 'random', or 'confidence_topk', got {start_mode!r}")
        # start_mode='random': no fixed vehicle->cluster schedule at all, not even
        # anchored-and-cycled. Each vehicle's cluster is decided ORGANICALLY by
        # whichever node the (unbiased) policy happens to pick as ITS OWN first
        # stop -- current_cluster is updated dynamically per rollout every time a
        # new vehicle begins. node_to_cluster_tensor/current_cluster/prev_veh_idx
        # are set up right after Step 1 below (need clusters + the actual Step-1
        # picks first); left as None here for the 'det_conf'/'node' paths.
        node_to_cluster_tensor = None
        current_cluster        = None
        prev_veh_idx           = None

        # Step 1: deterministic cluster starts (LLM mode, bias_mode='cluster' only) —
        # skipped when free_starts=True, bias_mode='node', or no clusters yet
        free_starts = self.trainer_params.get('free_starts', False)
        if llm_on and not free_starts and cluster_conf is not None and bias_mode == 'cluster':
            if start_mode in ('random', 'confidence_topk'):
                if start_mode == 'random':
                    rand_starts = random.sample(range(1, N + 1), pomo_size_eff)
                    start_t     = torch.tensor(rand_starts, dtype=torch.long, device=self.device)
                else:
                    # confidence_topk: n_starts = each cluster's own top-confidence node
                    # (the node the BKS/LKH-derived order visits FIRST in that cluster) --
                    # avoids the POMO n_starts symmetry-break diagnosed for VRPTW (a node
                    # with a late ready-time forced as the very first stop wastes most of
                    # the horizon regardless of policy quality, see n_starts audit doc).
                    # Confidence already encodes "good to visit early" for BKS-derived
                    # caches, so this reuses existing signal instead of a fresh wait0
                    # filter. K clusters -> K distinct starts, tiled+shuffled to fill
                    # pomo_size_eff (K is usually << pomo_size_eff).
                    top_nodes = cluster_conf.argmax(dim=1)                      # (K,)
                    K_top     = top_nodes.shape[0]
                    reps      = (pomo_size_eff + K_top - 1) // K_top
                    tiled     = top_nodes.repeat(reps)[:pomo_size_eff]
                    perm      = torch.randperm(pomo_size_eff, device=self.device)
                    start_t   = tiled[perm]
                selected    = start_t[None, :].expand(batch_size, pomo_size_eff)
                node_to_cluster_map = {n: k for k, nodes in enumerate(clusters) for n in nodes}
                node_to_cluster_tensor = torch.zeros(N + 1, dtype=torch.long, device=self.device)
                for n, k in node_to_cluster_map.items():
                    node_to_cluster_tensor[n] = k
                # vehicle 1's cluster = cluster of THIS rollout's own (fairly, randomly
                # sampled) start node -- known immediately, no schedule needed.
                current_cluster = node_to_cluster_tensor[start_t][None, :].expand(batch_size, pomo_size_eff).clone()
                prev_veh_idx    = torch.zeros(batch_size, pomo_size_eff, dtype=torch.long, device=self.device)
            else:
                selected = self._det_starts(inst, pomo_size_eff).expand(batch_size, -1)
            prob     = torch.ones(batch_size, pomo_size_eff, device=self.device)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            step = 2
        else:
            step = 1

        while not done and step < max_steps:
            # ACC encoder observations are selected per vehicle inside the model.

            # ── Confidence bias ────────────────────────────────────────────
            if llm_on and cluster_conf is not None:
                B_sz, P_sz = self.env.current_node.shape
                N1         = self.env.problem_size + 1
                dev        = self.env.current_node.device
                b_idx      = torch.arange(B_sz, device=dev)[:, None].expand(B_sz, P_sz)
                tt_to_j    = self.env.tt[b_idx.reshape(-1), self.env.current_node.reshape(-1)] \
                                 .reshape(B_sz, P_sz, N1)                    # (B, P, N+1)
                arrival_j  = self.env.current_time.unsqueeze(-1) + tt_to_j  # (B, P, N+1)
                tw_close_j = self.env.depot_node_tw_close.unsqueeze(1)      # (B, 1, N+1)
                tw_open    = (arrival_j <= tw_close_j).float()              # (B, P, N+1)

                # veh_idx/fresh_start are needed both by _acc_conf_now (to compute
                # conf_now for whichever bias_mode/start_mode is active) and by the
                # current_cluster update just below the decode call.
                veh_idx     = self.env.depot_visit_count.long()             # (B, P)
                fresh_start = (veh_idx != prev_veh_idx)                     # (B, P) bool, only meaningful for start_mode in ('random','confidence_topk')
                conf_now = self._acc_conf_now(
                    inst['name'], cluster_conf, bias_mode, K_clusters, roll_shifts,
                    start_mode, current_cluster, veh_idx, fresh_start, bias_strength,
                    self.env.current_time,
                )
                selected, prob = self.model(state, llm_bias=conf_now)
            else:
                selected, prob = self.model(state)

            if start_mode in ('random', 'confidence_topk') and current_cluster is not None:
                # Resolve any fresh vehicle's cluster from what it just picked, for
                # the NEXT step's bias; carry over veh_idx for the next comparison.
                picked_cluster  = node_to_cluster_tensor[selected.clamp(min=0, max=N)]
                current_cluster = torch.where(fresh_start, picked_cluster, current_cluster)
                prev_veh_idx    = veh_idx

            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            step += 1

        if reward is None:
            reward = self._fallback_reward()

        # ── Episode stats update + adaptive bias ──────────────────────
        best_idx     = int(reward[0].argmax().item())
        best_reward  = float(reward[0, best_idx].item())
        node_list    = self.env.selected_node_list[0, best_idx].cpu().tolist()
        routes       = _extract_routes(node_list)
        unserved, late, late_times, K, total_dist = self._compute_late_unserved(inst, routes)
        episode_tracker.update(inst['name'], N, unserved, late,
                               late_times=late_times, K=K, D=total_dist, reward=best_reward,
                               routes=routes)

        # ── Ontology-upgrade Phase 1 diagnostic logging (design doc Part I) ──
        phase1_logger = self._phase1_logger
        if phase1_logger is not None and phase1_logger.active(self._current_epoch):
            _epoch = self._current_epoch
            if phase1_logger.wants_full(_epoch):
                # all P rollouts -- needed for §5 cluster-consensus co-assignment
                for p in range(pomo_size_eff):
                    nl_p     = self.env.selected_node_list[0, p].cpu().tolist()
                    routes_p = _extract_routes(nl_p)
                    detail_p = self._compute_episode_detail(inst, routes_p)
                    phase1_logger.log(EpisodeLog(
                        instance=inst['name'], epoch=_epoch, rollout=p,
                        routes=routes_p, arrival=detail_p['arrival'], wait=detail_p['wait'],
                        late_by=detail_p['late_by'], served=detail_p['served'],
                        reward=float(reward[0, p].item()),
                        K=detail_p['K'], D=detail_p['D'], Lc=detail_p['Lc'], Lt=detail_p['Lt'],
                    ))
            elif phase1_logger.wants_sampled(_epoch):
                # single representative (best-reward) rollout -- enough for §6/§7/§8
                detail = self._compute_episode_detail(inst, routes)
                phase1_logger.log(EpisodeLog(
                    instance=inst['name'], epoch=_epoch, rollout=best_idx,
                    routes=routes, arrival=detail['arrival'], wait=detail['wait'],
                    late_by=detail['late_by'], served=detail['served'],
                    reward=best_reward, K=detail['K'], D=detail['D'],
                    Lc=detail['Lc'], Lt=detail['Lt'],
                ))

        if self.trainer_params.get('advantage_mode', 'reward') == 'hybrid_lex':
            unserved = (self.env.visited_ninf_flag[:, :, 1:] == 0.0).sum(dim=2).float()
            cur_eps  = eps_schedule(self._current_epoch - self.start_epoch, self.trainer_params)
            advantage = hybrid_advantage(
                self.env.total_late, self.env.depot_visit_count, self.env._get_travel_distance(),
                eps=cur_eps,
                A=self.trainer_params.get('advantage_A', 7.0),
                clip=self.trainer_params.get('advantage_clip', 3.0),
                running_std=self._adv_running_std,
                unserved=unserved,
            )
            tc = tier_counts(self.env.total_late, self.env.depot_visit_count,
                              self.env._get_travel_distance(), cur_eps, unserved=unserved)
            for k in ('t0', 't1', 't2', 'total'):
                self._tier_counts[k] += tc[k]
            self._last_eps = cur_eps
        else:
            advantage = reward - reward.float().mean(dim=1, keepdim=True)
            advantage = advantage / (advantage.std(dim=1, keepdim=True).clamp(min=1e-6))
        log_prob  = prob_list.log().sum(dim=2)
        loss      = -(advantage * log_prob).mean()

        self.model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        max_pomo_reward, _ = reward.max(dim=1)
        self._timing['rl_train'] += time.time() - _t_rl
        return max_pomo_reward.float().mean().item(), loss.item()

    # ------------------------------------------------------------------
    # run() override — adds timing summary on top of base class
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int):
        self._current_epoch = epoch
        lp = self.llm_params
        if (lp.get('enabled') and not self._experience_refreshed
                and epoch == self._experience_refresh_epoch):
            self._refresh_experience_caches(epoch)
            self._experience_refreshed = True
        if (lp.get('enabled') and self._rl_recluster_epoch is not None
                and not self._rl_reclustered
                and epoch == self._rl_recluster_epoch):
            self._refresh_clusters_from_rl_routes(epoch)
            self._rl_reclustered = True
        return super()._train_one_epoch(epoch)

    def run(self):
        super().run()
        t = self._timing
        total = t['llm_init'] + t['llm_refresh'] + t['rl_train']
        print(
            f'\n[Timing] LLM init={t["llm_init"]/60:.1f}min  '
            f'LLM refresh={t["llm_refresh"]/60:.1f}min  '
            f'RL train={t["rl_train"]/60:.1f}min  '
            f'(tracked total={total/60:.1f}min)'
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _eval_test(self):
        self.model.eval()
        lp        = self.llm_params
        llm_on    = lp['enabled']
        rewards, per_inst = [], []
        orig_pomo = self.env.pomo_size
        max_steps = self.trainer_params.get('max_steps', 600)
        bs        = self.trainer_params.get('test_batch_size', 1)

        for inst in self.test_pool:
            accident_list = _get_accidents(inst) if llm_on else []
            pomo_size_eff = self.trainer_params['pomo_size']
            if llm_on:
                self._ensure_soft_cluster_cache(inst)
                clusters     = self._cluster_assign_cache.get(inst['name'], [])
                K_clusters   = len(clusters) or 1
                cluster_conf = self._cluster_conf_cache.get(inst['name'])
            else:
                K_clusters   = pomo_size_eff
                cluster_conf = None

            bias_strength = lp.get('bias_strength', 5.0)
            roll_shifts   = torch.arange(pomo_size_eff, device=self.device)
            self.env.pomo_size = pomo_size_eff
            bias_mode  = self.trainer_params.get('bias_mode', 'cluster')
            start_mode = self.trainer_params.get('start_mode', 'det_conf')
            node_conf  = cluster_conf.sum(dim=0) if (bias_mode == 'node' and cluster_conf is not None) else None
            node_to_cluster_tensor = None
            current_cluster        = None
            prev_veh_idx           = None

            _, _, rain_evs = _get_rain_nodes_mult_evs(inst)
            N = self.env.problem_size

            batch = make_batch(inst, bs, self.device)
            self.env.load_problems(batch)

            reset_state, _, _ = self.env.reset()
            reset_state.rain_tokens = _make_rain_tokens(inst, rain_evs, bs, self.device)
            reset_state.acc_tokens  = _make_acc_tokens([], N, bs, self.device)
            self.model.pre_forward(reset_state)
            state, reward, done = self.env.pre_step()

            # Step 0: depot
            sel, _ = self.model(state)
            state, reward, done = self.env.step(sel)

            # Same start-mode handling as _best_solution: 'det_conf' assumes vehicle
            # k always visits cluster k (never re-verified); 'random'/'confidence_topk'
            # track the cluster dynamically from whichever node the rollout actually
            # starts on, so bias stays correctly aligned even once the policy drifts
            # from the initial clustering. Previously this always used the 'det_conf'
            # path regardless of --start-mode, silently misaligning eval-time bias.
            free_starts = self.trainer_params.get('free_starts', False)
            if llm_on and not free_starts and bias_mode == 'cluster':
                if start_mode in ('random', 'confidence_topk'):
                    if start_mode == 'random':
                        rand_starts = random.sample(range(1, N + 1), pomo_size_eff)
                        start_t     = torch.tensor(rand_starts, dtype=torch.long, device=self.device)
                    else:
                        top_nodes = cluster_conf.argmax(dim=1)
                        K_top     = top_nodes.shape[0]
                        reps      = (pomo_size_eff + K_top - 1) // K_top
                        tiled     = top_nodes.repeat(reps)[:pomo_size_eff]
                        perm      = torch.randperm(pomo_size_eff, device=self.device)
                        start_t   = tiled[perm]
                    sel         = start_t[None, :].expand(bs, pomo_size_eff)
                    node_to_cluster_map = {n: k for k, nodes in enumerate(clusters) for n in nodes}
                    node_to_cluster_tensor = torch.zeros(N + 1, dtype=torch.long, device=self.device)
                    for n, k in node_to_cluster_map.items():
                        node_to_cluster_tensor[n] = k
                    current_cluster = node_to_cluster_tensor[start_t][None, :].expand(bs, pomo_size_eff).clone()
                    prev_veh_idx    = torch.zeros(bs, pomo_size_eff, dtype=torch.long, device=self.device)
                else:
                    sel = self._det_starts(inst, pomo_size_eff).expand(bs, -1)
                state, reward, done = self.env.step(sel)
                step = 2
            else:
                step = 1


            while not done and step < max_steps:

                # (LLM confidence for every wave is precomputed ahead of routing
                # and blended in per rollout by _acc_conf_now below -- see the
                # note in _train_one_batch's decode loop for why no live-
                # triggered refresh happens here anymore.)
                # ACC encoder observations are selected per vehicle inside the model.

                if llm_on and cluster_conf is not None:
                    veh_idx     = self.env.depot_visit_count.long()
                    fresh_start = (veh_idx != prev_veh_idx)
                    conf_now = self._acc_conf_now(
                        inst['name'], cluster_conf, bias_mode, K_clusters, roll_shifts,
                        start_mode, current_cluster, veh_idx, fresh_start, bias_strength,
                        self.env.current_time,
                    )
                    sel, _ = self.model(state, llm_bias=conf_now)
                else:
                    sel, _ = self.model(state)

                if start_mode in ('random', 'confidence_topk') and current_cluster is not None:
                    picked_cluster  = node_to_cluster_tensor[sel.clamp(min=0, max=N)]
                    current_cluster = torch.where(fresh_start, picked_cluster, current_cluster)
                    prev_veh_idx    = veh_idx

                state, reward, done = self.env.step(sel)
                step += 1

            if reward is None:
                reward = self._fallback_reward()

            max_r, _ = reward.max(dim=1)
            rewards.append(max_r.mean().item())
            per_inst.append((inst['name'], max_r.mean().item()))

        self.env.pomo_size = orig_pomo
        return sum(rewards) / len(rewards) if rewards else 0.0, per_inst

    # ------------------------------------------------------------------
    # Best solution extraction (for final eval / test-only)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _best_solution(self, inst: dict):
        self.model.eval()
        lp     = self.llm_params
        llm_on = lp['enabled']
        N      = self.env.problem_size

        _, _, rain_evs = _get_rain_nodes_mult_evs(inst)
        accident_list = _get_accidents(inst) if llm_on else []

        pomo_size_eff = self.trainer_params['pomo_size']
        if llm_on:
            self._ensure_soft_cluster_cache(inst)
            clusters     = self._cluster_assign_cache.get(inst['name'], [])
            K_clusters   = len(clusters) or 1
            cluster_conf = self._cluster_conf_cache.get(inst['name'])
        else:
            K_clusters   = pomo_size_eff
            cluster_conf = None

        bias_strength = lp.get('bias_strength', 5.0)
        roll_shifts   = torch.arange(pomo_size_eff, device=self.device)
        self.env.pomo_size = pomo_size_eff
        bias_mode  = self.trainer_params.get('bias_mode', 'cluster')
        start_mode = self.trainer_params.get('start_mode', 'det_conf')
        node_to_cluster_tensor = None
        current_cluster        = None
        prev_veh_idx           = None

        batch = make_batch(inst, 1, self.device)
        self.env.load_problems(batch)

        reset_state, _, _ = self.env.reset()
        reset_state.rain_tokens = _make_rain_tokens(inst, rain_evs, 1, self.device)
        reset_state.acc_tokens  = _make_acc_tokens([], N, 1, self.device)
        self.model.pre_forward(reset_state)
        state, reward, done = self.env.pre_step()

        # Step 0: depot
        sel, _ = self.model(state)
        state, reward, done = self.env.step(sel)

        max_steps        = self.trainer_params.get('max_steps', 600)

        free_starts = self.trainer_params.get('free_starts', False)
        if llm_on and not free_starts and bias_mode == 'cluster':
            if start_mode in ('random', 'confidence_topk'):
                if start_mode == 'random':
                    rand_starts = random.sample(range(1, N + 1), pomo_size_eff)
                    start_t     = torch.tensor(rand_starts, dtype=torch.long, device=self.device)
                else:
                    top_nodes = cluster_conf.argmax(dim=1)
                    K_top     = top_nodes.shape[0]
                    reps      = (pomo_size_eff + K_top - 1) // K_top
                    tiled     = top_nodes.repeat(reps)[:pomo_size_eff]
                    perm      = torch.randperm(pomo_size_eff, device=self.device)
                    start_t   = tiled[perm]
                sel         = start_t[None, :].expand(1, pomo_size_eff)
                node_to_cluster_map = {n: k for k, nodes in enumerate(clusters) for n in nodes}
                node_to_cluster_tensor = torch.zeros(N + 1, dtype=torch.long, device=self.device)
                for n, k in node_to_cluster_map.items():
                    node_to_cluster_tensor[n] = k
                current_cluster = node_to_cluster_tensor[start_t][None, :].expand(1, pomo_size_eff).clone()
                prev_veh_idx    = torch.zeros(1, pomo_size_eff, dtype=torch.long, device=self.device)
            else:
                sel = self._det_starts(inst, pomo_size_eff).expand(1, -1)
            state, reward, done = self.env.step(sel)
            step = 2
        else:
            step = 1

        while not done and step < max_steps:

            # (LLM confidence for every wave is precomputed ahead of routing
            # and blended in per rollout by _acc_conf_now below -- see the
            # note in _train_one_batch's decode loop for why no live-
            # triggered refresh happens here anymore.)
            # ACC encoder observations are selected per vehicle inside the model.

            if llm_on and cluster_conf is not None:
                veh_idx     = self.env.depot_visit_count.long()
                fresh_start = (veh_idx != prev_veh_idx)
                conf_now = self._acc_conf_now(
                    inst['name'], cluster_conf, bias_mode, K_clusters, roll_shifts,
                    start_mode, current_cluster, veh_idx, fresh_start, bias_strength,
                    self.env.current_time,
                )
                sel, _ = self.model(state, llm_bias=conf_now)
            else:
                sel, _ = self.model(state)

            if start_mode in ('random', 'confidence_topk') and current_cluster is not None:
                picked_cluster  = node_to_cluster_tensor[sel.clamp(min=0, max=N)]
                current_cluster = torch.where(fresh_start, picked_cluster, current_cluster)
                prev_veh_idx    = veh_idx

            state, reward, done = self.env.step(sel)
            step += 1

        if reward is None:
            reward = self._fallback_reward()

        best_idx  = int(reward[0].argmax().item())
        node_list = self.env.selected_node_list[0, best_idx].cpu().tolist()
        return _extract_routes(node_list), float(reward[0, best_idx].item())

    # ── 8-fold augmentation helpers ────────────────────────────────────────────

    @staticmethod
    def _aug_instance(inst: dict, aug_idx: int) -> dict:
        """Return inst with depot_xy/node_xy transformed by one of 8 D4 isometries.
        tt (travel-time matrix) is unchanged — Euclidean distances are invariant."""
        aug = dict(inst)
        def _tf(xy):      # xy: (K, 2) cpu tensor
            x, y = xy[:, 0].clone(), xy[:, 1].clone()
            ops = [
                lambda x, y: torch.stack([x,   y  ], 1),  # 0: identity
                lambda x, y: torch.stack([1-x, y  ], 1),  # 1: flip x
                lambda x, y: torch.stack([x,   1-y], 1),  # 2: flip y
                lambda x, y: torch.stack([1-x, 1-y], 1),  # 3: rotate 180
                lambda x, y: torch.stack([y,   x  ], 1),  # 4: transpose
                lambda x, y: torch.stack([1-y, x  ], 1),  # 5
                lambda x, y: torch.stack([y,   1-x], 1),  # 6
                lambda x, y: torch.stack([1-y, 1-x], 1),  # 7
            ]
            return ops[aug_idx % 8](x, y)
        aug['depot_xy'] = _tf(inst['depot_xy'])
        aug['node_xy']  = _tf(inst['node_xy'])
        return aug

    def _best_solution_aug(self, inst: dict, n_aug: int = 8):
        """Run _best_solution on n_aug augmented instances, return best across all."""
        best_routes, best_reward = None, -float('inf')
        for aug_idx in range(n_aug):
            aug_inst = self._aug_instance(inst, aug_idx) if aug_idx > 0 else inst
            routes, reward = self._best_solution(aug_inst)
            if reward > best_reward:
                best_reward = reward
                best_routes = routes
        return best_routes, best_reward

    def _best_solution_mc(self, inst: dict, n_samples: int = 8):
        """MC sampling: run _best_solution n_samples times with stochastic decoding.
        Each run uses the same K cluster starts but samples from the probability
        distribution at every step → different trajectories each time.
        Compatible with LLM confidence bias (bias shifts the distribution; sampling
        adds diversity on top)."""
        mp = self.model.mp                   # model params dict (train_vrptw_llm.py uses self.mp)
        orig_eval_type = mp.get('eval_type', 'argmax')
        mp['eval_type'] = 'softmax'          # switch to multinomial sampling
        try:
            best_routes, best_reward = None, -float('inf')
            for _ in range(n_samples):
                routes, reward = self._best_solution(inst)
                if reward > best_reward:
                    best_reward = reward
                    best_routes = routes
        finally:
            mp['eval_type'] = orig_eval_type  # always restore
        return best_routes, best_reward


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Soft-Clustering POMO VRPTW')
    parser.add_argument('--test-only',      action='store_true')
    parser.add_argument('--no-llm',         action='store_true')
    parser.add_argument('--guidance-mode', choices=['cluster_confidence', 'selective_advance'],
                        default='cluster_confidence')
    parser.add_argument('--no-ontology',    action='store_true')
    parser.add_argument('--resume',         default=None)
    parser.add_argument('--epochs',         type=int,   default=trainer_params['epochs'])
    parser.add_argument('--cluster-mode', choices=['rl', 'kim2006'], default='rl')
    parser.add_argument('--model',          default=llm_params['model'])
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--config',         default='E',
                        choices=list(_base_config._REWARD_CONFIGS))
    parser.add_argument('--refresh-cache',  action='store_true',
                        help='Ignore disk cache and call LLM fresh')
    parser.add_argument('--no-cot',         action='store_true')
    parser.add_argument('--use-misfit',     action='store_true',
                        help='Opt INTO the separate LLM-flagged Step 1 misfit gate (ablation) -- '
                             'default is OFF: every customer gets a plain Step 2 score and misfit '
                             'is defined purely by the depot-cutoff comparison')
    parser.add_argument('--cache-only',     action='store_true',
                        help='Generate LLM caches only (no training)')
    parser.add_argument('--bias-strength',  type=float, default=llm_params['bias_strength'])
    parser.add_argument('--experience-refresh-epoch', type=int, default=500,
                        help='Epoch at which to rebuild LLM caches with episode-derived few-shot examples')
    parser.add_argument('--rl-recluster-epoch', type=int, default=None,
                        help='Epoch at which to re-cluster using RL best-episode routes and re-run LLM')
    parser.add_argument('--no-init-llm', action='store_true',
                        help='Skip initial LLM clustering; Phase 1 runs as pure RL until --rl-recluster-epoch')
    parser.add_argument('--benchmark',      default='c1',
                        choices=['c1', 'rc1', 'r1', 'mixed', 'c2'])
    parser.add_argument('--acc-ratio',      type=float, default=0.1,
                        help='ACC instance training ratio (default 0.1 → base:0.7 rain:0.2 acc:0.1)')
    parser.add_argument('--no-curriculum',  action='store_true')
    parser.add_argument('--result-dir',     default=None,
                        help='Override base result directory (default: result_soft)')
    parser.add_argument('--llm-cache-dir',  default=None,
                        help='Override llm_cache directory (default: {result_dir}/llm_cache)')
    parser.add_argument('--fewshot-cache',  action='store_true',
                        help='Load {name}_fewshot_cluster.json instead of {name}_cluster.json')
    parser.add_argument('--rl-fewshot',     action='store_true',
                        help='Use RL-derived cluster (r106_rl_cluster.json) as few-shot example instead of BKS')
    parser.add_argument('--free-starts',   action='store_true',
                        help='Skip deterministic Step 1; let model freely choose first node with LLM bias (cyclic vehicle assignment still applies)')
    parser.add_argument('--start-mode',    default='det_conf',
                        choices=['det_conf', 'random', 'confidence_topk'],
                        help="'det_conf' (default): vehicle k assumed to visit cluster (k+p)%%K, "
                             "never re-verified -- breaks once the policy drifts from the initial "
                             "clustering. 'confidence_topk'/'random': each vehicle's cluster is "
                             "locked dynamically from whichever node it actually starts on (no bias "
                             "on that first node, but every stop after is correctly aligned).")
    parser.add_argument('--batch-size',     type=int, default=None,
                        help='Override train_batch_size (default: inherited from train_vrptw.py, '
                             'currently 64) -- pass this to match whatever --batch-size Phase 1 '
                             'used (e.g. 32), since this file previously had no way to override it '
                             'and silently trained at a different batch size than Phase 1.')
    parser.add_argument('--pomo',           type=int, default=None, required=True,
                        help='POMO rollout count, forced identical across LLM-on and '
                             '--no-llm so rollout count never confounds the bias '
                             'comparison (no default -- must be set explicitly).')
    args = parser.parse_args()

    if args.seed is not None:
        _set_seed(args.seed)
        print(f'[Seed] {args.seed}')

    # Apply reward config — use benchmark-calibrated table when available
    _BENCH_CFG = {
        'r1':  _base_config._REWARD_CONFIGS_R1,
        'rc1': _base_config._REWARD_CONFIGS_RC1,
    }
    _cfg_table = _BENCH_CFG.get(args.benchmark, _base_config._REWARD_CONFIGS)
    cfg = _cfg_table.get(args.config, _base_config._REWARD_CONFIGS[args.config])
    env_params.update(cfg)
    print(f'[Config {args.config}] benchmark={args.benchmark}  {cfg}')

    trainer_params['epochs']           = args.epochs
    trainer_params['cluster_mode']     = args.cluster_mode
    trainer_params['guidance_mode']    = args.guidance_mode
    llm_params['model']                = args.model
    llm_params['reward_config']        = args.config
    llm_params['use_fewshot_cache']    = args.fewshot_cache
    llm_params['use_misfit']           = args.use_misfit
    trainer_params['free_starts']      = args.free_starts
    trainer_params['start_mode']       = args.start_mode

    # Benchmark selection
    # rc1/r1 use the single _acc2/_rain2 variant per base instance (one each,
    # not the older two-variant _rain_A/_rain_B/_acc_A/_acc_B scheme) --
    # matching the same validated event files used all session for pure-RL
    # training on these two families (see final_*_weightedsum_aug8.py): same
    # base pool, same held-out test instance, same acceptance-criteria event
    # generation. c1/mixed/c2 are untouched (not in active use this session).
    if args.benchmark == 'rc1':
        _B = [f"rc{i:03d}" for i in range(102, 109)]
        _E = [f"{n}_rain2" for n in _B]
        trainer_params['train_instances'] = _B + _E
        trainer_params['test_instances']  = ["rc101", "rc101_acc2", "rc101_rain2"]
        print('[benchmark=rc1] Train: rc102-rc108 + rain2  |  Test: rc101 (base/acc2/rain2)')
    elif args.benchmark == 'r1':
        _B = [f"r{i:03d}" for i in range(102, 113)]
        _E = [f"{n}_rain2" for n in _B]
        trainer_params['train_instances'] = _B + _E
        trainer_params['test_instances']  = ["r101", "r101_acc2", "r101_rain2"]
        print('[benchmark=r1] Train: r102-r112 + rain2  |  Test: r101 (base/acc2/rain2)')
    elif args.benchmark == 'mixed':
        _C  = [f"c{i:03d}"  for i in range(102, 110)]
        _RC = [f"rc{i:03d}" for i in range(102, 109)]
        _R  = [f"r{i:03d}"  for i in range(102, 113)]
        _ALL = _C + _RC + _R
        _E   = [f"{n}_rain_A" for n in _ALL] + [f"{n}_rain_B" for n in _ALL]
        trainer_params['train_instances'] = _ALL + _E
        trainer_params['test_instances']  = [
            "c101", "rc101", "r101",
            "c101_rain_A", "rc101_rain_A", "r101_rain_A",
            "c101_acc_A",  "rc101_acc_A",  "r101_acc_A",
        ]
        print('[benchmark=mixed] Train: C1+RC1+R1 + rain  |  Test: all scenarios')
    elif args.benchmark == 'c2':
        _B = [f"c{i:03d}" for i in range(202, 209)]
        _E = [f"{n}_rain_A" for n in _B] + [f"{n}_rain_B" for n in _B]
        trainer_params['train_instances'] = _B + _E
        trainer_params['test_instances']  = [
            "c201", "c201_rain_A", "c201_rain_B", "c201_acc_A", "c201_acc_B"
        ]
        print('[benchmark=c2] Train: c202-c208 + rain  |  Test: c201 (all scenarios)')
    else:  # c1 default -- single _acc2/_rain2 variant, matching rc1/r1 above
        _B = [f"c{i:03d}" for i in range(102, 110)]
        _E = [f"{n}_rain2" for n in _B]
        trainer_params['train_instances'] = _B + _E
        trainer_params['test_instances']  = ["c101", "c101_acc2", "c101_rain2"]
        print('[benchmark=c1] Train: c102-c109 + rain2  |  Test: c101 (base/acc2/rain2)')

    # ACC instances in training
    if args.acc_ratio > 0:
        _cur = trainer_params['train_instances']
        if args.benchmark in ('rc1', 'r1', 'c1'):
            # single _acc2 variant per base instance, matching _rain2 above
            _base_names = [n for n in _cur if '_rain2' not in n and '_acc2' not in n]
            _acc_names  = [f"{n}_acc2" for n in _base_names]
        else:
            _base_names = [n for n in _cur if '_rain_' not in n and '_acc_' not in n]
            _acc_names  = ([f"{n}_acc_A" for n in _base_names]
                           + [f"{n}_acc_B" for n in _base_names])
        trainer_params['train_instances'] = _cur + _acc_names
        trainer_params['type_ratio'] = {'base': 1/3, 'rain': 1/3, 'acc': 1/3}
        trainer_params['curriculum_epoch'] = 0
        print(f'[acc-ratio={args.acc_ratio}] ACC instances added. Ratio=1/3 each, no curriculum.')

    if args.no_curriculum:
        trainer_params['curriculum_epoch'] = 0

    if args.result_dir:
        trainer_params['result_dir'] = args.result_dir
    if args.llm_cache_dir:
        trainer_params['llm_cache_dir'] = os.path.abspath(args.llm_cache_dir)
    if args.no_llm:
        llm_params['enabled'] = False
    if args.no_ontology:
        llm_params['use_ontology'] = False
    if args.resume:
        import re as _re
        _ckpt = args.resume
        _m = _re.search(r'checkpoint-(\d+|last)\.pt$', _ckpt)
        if _m:
            trainer_params['model_load']['enable'] = True
            trainer_params['model_load']['path']   = os.path.dirname(_ckpt)
            trainer_params['model_load']['epoch']  = (int(_m.group(1))
                                                      if _m.group(1).isdigit()
                                                      else _m.group(1))
        else:
            trainer_params['model_load']['enable'] = True
            trainer_params['model_load']['path']   = _ckpt
    if args.refresh_cache:
        trainer_params['force_llm_refresh'] = True
        print('[LLM] Cache refresh: calling LLM fresh')
    if args.no_cot:
        llm_params['use_cot'] = False
    llm_params['bias_strength'] = args.bias_strength
    trainer_params['experience_refresh_epoch'] = args.experience_refresh_epoch
    trainer_params['benchmark'] = args.benchmark
    if args.rl_recluster_epoch is not None:
        trainer_params['rl_recluster_epoch'] = args.rl_recluster_epoch
    if args.no_init_llm:
        llm_params['no_init_llm'] = True
    if args.rl_fewshot:
        trainer_params['rl_fewshot'] = True
    trainer_params['pomo_size'] = args.pomo
    print(f'[pomo_size] {args.pomo}  (forced identical for LLM-on and --no-llm)')
    if args.batch_size is not None:
        trainer_params['train_batch_size'] = args.batch_size
        print(f'[batch] train_batch_size overridden to {args.batch_size}')

    tag = ("Soft-Cluster+Ontology+LLM+RL" if llm_params['enabled'] and llm_params['use_ontology']
           else "Soft-Cluster+LLM+RL"       if llm_params['enabled']
           else "POMO (no LLM)")
    print(f"[{tag}]  model={llm_params['model']}  bias={llm_params['bias_strength']}")

    trainer_params['test_only'] = args.test_only
    trainer = SoftClusterTrainer(
        env_params, model_params, optimizer_params, trainer_params, llm_params,
    )

    if args.cache_only:
        print('[cache-only] All LLM caches generated. Exiting.')
        sys.exit(0)

    if args.test_only:
        ckpt_path = args.resume if args.resume else os.path.join(trainer.result_dir, 'checkpoint-last.pt')
        if not os.path.isfile(ckpt_path):
            print(f'[Test] No checkpoint at {ckpt_path}')
            return
        ckpt = torch.load(ckpt_path, map_location=trainer.device)
        _load_model_flexible(trainer.model, ckpt['model_state_dict'])
        trainer.model.eval()
        plots_dir = os.path.join(trainer.result_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        for inst in trainer.test_pool:
            # Match the end-of-training eval's search budget (_best_solution_aug,
            # aug x8) -- calling plain _best_solution here bypassed that fix for
            # this separate --test-only entry point.
            routes, reward = trainer._best_solution_aug(inst)
            print(f"\n[Test] {inst['name']}  reward={reward:.2f}")
            _print_solution(inst, routes, reward)
            sol_path = os.path.join(plots_dir, f"{inst['name']}_solution.txt")
            _save_solution(inst, routes, reward, sol_path)
            plot_routes(inst, routes, reward,
                        os.path.join(plots_dir, f"{inst['name']}_routes.png"))
        return

    trainer.run()


if __name__ == '__main__':
    main()
