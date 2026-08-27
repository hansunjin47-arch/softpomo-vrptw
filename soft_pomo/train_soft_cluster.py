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
    New: {"0": {"node": score, ...}, "1": {...}, ...}  (compact, written by _dump_cluster_cache)
    Returns (clusters: list[list[int]], flat_conf: dict[int, float])
    """
    if 'clusters' in data:
        clusters  = data['clusters']
        flat_conf = {int(k): v for k, v in data['confidence'].items()}
    else:
        keys      = sorted(data.keys(), key=lambda x: int(x))
        clusters  = [[int(n) for n in data[k]] for k in keys]
        flat_conf = {int(n): s for k in keys for n, s in data[k].items()}
    return clusters, flat_conf


def _dump_cluster_cache(clusters: list, flat_conf: dict) -> dict:
    """Serialize to compact format: {cluster_idx: {node: score}}."""
    return {str(k): {str(n): round(flat_conf.get(n, 0.0), 2) for n in nodes}
            for k, nodes in enumerate(clusters)}

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

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
                                  build_rl_fewshot_block)
from SoftClusterOntology import episode_tracker

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
llm_params     = dict(_base_llm_params)
llm_params['soft_clustering'] = True   # flag for this architecture

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
        # Cluster caches must exist before super().__init__ calls _ensure_llm_cache
        self._cluster_assign_cache:  dict[str, list[list[int]]] = {}
        self._cluster_conf_cache:    dict[str, torch.Tensor]    = {}
        self._original_cluster_conf: dict[str, torch.Tensor]    = {}
        self._timing = {'llm_init': 0.0, 'llm_refresh': 0.0, 'rl_train': 0.0}
        # Test instances use real-time LLM on accident — no pre-generation
        self._test_instance_names = {n.upper() for n in trainer_p.get('test_instances', [])}
        self._acc_prompt_counter: dict[str, int] = {}
        self._experience_refreshed = False
        self._experience_refresh_epoch = trainer_p.get('experience_refresh_epoch', 500)
        self._rl_reclustered = False
        self._rl_recluster_epoch = trainer_p.get('rl_recluster_epoch', None)
        self._pending_cache_retry: list = []
        # Few-shot block: precomputed once, prepended to init LLM calls that have no episode experience
        self._bks_fewshot_block = ""
        if llm_p.get('enabled', True) and not llm_p.get('no_init_llm', False):
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

        K_try    = max(1, K)
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
            if all(_tw_feasible(cl) for cl in raw):
                break    # all clusters TW-feasible → done
            K_try += 1   # add one more cluster and retry

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
    ) -> torch.Tensor:
        """Build (K, N+1) confidence tensor. conf[k, n] = confidence of node n for vehicle k."""
        conf = torch.zeros(K, N + 1, device=self.device)
        for k, cluster_nodes in enumerate(clusters[:K]):
            for n in cluster_nodes:
                if 1 <= n <= N:
                    conf[k, n] = float(flat_conf.get(n, 0.0))
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
                if lateness > 1e-6:
                    late.add(node)
                    late_times[node] = lateness
                cur_time = svc_start + float(service[node])
                cur      = node
            total_dist += float(tt[cur, 0])  # return to depot

        unserved = [n for n in range(1, N + 1) if n not in visited]
        return unserved, list(late), late_times, K, total_dist

    # ------------------------------------------------------------------
    # LLM cluster confidence cache (replaces _ensure_llm_cache)
    # ------------------------------------------------------------------

    def _ensure_llm_cache(self, inst: dict, force_refresh: bool = False):
        """Override: redirect to soft-cluster cache."""
        self._ensure_soft_cluster_cache(inst, force_refresh)

    def _ensure_soft_cluster_cache(self, inst: dict, force_refresh: bool = False):
        """Cluster customers, call LLM per cluster, cache (K, N+1) confidence tensor."""
        name   = inst['name']
        is_acc = 'ACC' in name.upper()

        if name in self._cluster_conf_cache and not force_refresh:
            return

        lp = self.llm_params
        N  = inst['n_customers']

        # ACC instances: reuse base clusters + confidence at episode start.
        # Post-accident scores are pre-generated here (like BASE/RAIN) and cached to disk.
        # _refresh_clusters_accident loads the cache and applies visited mask per episode.
        if is_acc:
            # ACC-specific rl_cluster takes highest priority
            _rl_acc_path = os.path.join(self._llm_cache_dir, f'{name}_rl_cluster.json')
            if os.path.isfile(_rl_acc_path) and not force_refresh:
                with open(_rl_acc_path, encoding='utf-8') as f:
                    cached = json.load(f)
                clusters, flat_conf = _parse_cluster_cache(cached)
                K_actual = len(clusters)
                built = self._build_cluster_conf_tensor(clusters, flat_conf, K_actual, N)
                self._cluster_assign_cache[name]  = clusters
                self._cluster_conf_cache[name]    = built
                self._original_cluster_conf[name] = built.clone()
                print(f'  [Cluster cache] loaded {name} (rl_acc): {K_actual} clusters')
                return

            base_key = name.split('_')[0]
            if base_key not in self._cluster_assign_cache:
                # Base not yet in memory — load its disk cache first (RL cache takes priority)
                _rl_base = os.path.join(self._llm_cache_dir, f'{base_key}_rl_cluster.json')
                base_cache = _rl_base if os.path.isfile(_rl_base) else os.path.join(self._llm_cache_dir, f'{base_key}_cluster.json')
                if os.path.isfile(base_cache):
                    with open(base_cache, encoding='utf-8') as f:
                        bd = json.load(f)
                    clusters_b, flat_conf_b = _parse_cluster_cache(bd)
                    K_b = len(clusters_b)
                    built_b = self._build_cluster_conf_tensor(clusters_b, flat_conf_b, K_b, N)
                    self._cluster_assign_cache[base_key]  = clusters_b
                    self._cluster_conf_cache[base_key]    = built_b
                    self._original_cluster_conf[base_key] = built_b.clone()
            if base_key in self._cluster_assign_cache:
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
            raise RuntimeError(f'ACC {name}: base {base_key} cache not found')

        # Number of clusters = theoretical minimum vehicles
        K = math.ceil(float(inst['node_demand'].sum().item()))
        K = max(1, K)

        # RL cluster cache takes highest priority (instance-specific first, then base key)
        _base_key_for_rl = name.split('_')[0]
        _rl_path_inst = os.path.join(self._llm_cache_dir, f'{name}_rl_cluster.json')
        _rl_path_base = os.path.join(self._llm_cache_dir, f'{_base_key_for_rl}_rl_cluster.json')
        _rl_path = _rl_path_inst if os.path.isfile(_rl_path_inst) else _rl_path_base
        if os.path.isfile(_rl_path) and not force_refresh:
            with open(_rl_path, encoding='utf-8') as f:
                cached = json.load(f)
            clusters, flat_conf = _parse_cluster_cache(cached)
            K_actual = len(clusters)
            built = self._build_cluster_conf_tensor(clusters, flat_conf, K_actual, N)
            self._cluster_assign_cache[name]  = clusters
            self._cluster_conf_cache[name]    = built
            self._original_cluster_conf[name] = built.clone()
            src = 'rl_instance' if os.path.isfile(_rl_path_inst) else 'rl_base'
            print(f'  [Cluster cache] loaded {name} ({src}): {K_actual} clusters')
            return

        fewshot_path = os.path.join(self._llm_cache_dir, f'{name}_fewshot_cluster.json')
        cache_path   = os.path.join(self._llm_cache_dir, f'{name}_cluster.json')
        use_fewshot  = lp.get('use_fewshot_cache', False)
        load_path    = fewshot_path if (use_fewshot and os.path.isfile(fewshot_path)) else cache_path
        if os.path.isfile(load_path) and not force_refresh:
            with open(load_path, encoding='utf-8') as f:
                cached = json.load(f)
            clusters, flat_conf = _parse_cluster_cache(cached)
            K_actual = len(clusters)
            built = self._build_cluster_conf_tensor(clusters, flat_conf, K_actual, N)
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
                    experience_section=combined_experience,
                    reward_config=lp.get('reward_config', 'F'),
                    bks_fewshot=bks_fewshot,
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
                    experience_section=combined_experience,
                    reward_config=lp.get('reward_config', 'F'),
                    bks_fewshot=bks_fewshot,
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
        cache_path = os.path.join(self._llm_cache_dir, f'{name}_acc_refresh_{cur_t:.1f}.json')
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
                        cur_time=cur_t,
                        reward_config=lp.get('reward_config', 'F'),
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

        This keeps the start node aligned with the per-step cyclic bias assignment
        (cluster_idx = (veh_idx + p) % K_clusters, see _train_one_batch/_eval_test/
        _best_solution) even when pomo_size > K: rollouts p, p+K, p+2K, ... all
        target cluster (p % K) and are given that cluster's 1st, 2nd, 3rd, ...
        best-confidence node respectively, rather than an unrelated random node.
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
    # Cluster confidence re-scoring after accident
    # ------------------------------------------------------------------

    def _refresh_clusters_accident(
        self,
        inst: dict,
        cluster_conf: torch.Tensor,
        accidents,          # AccidentEvent or list[AccidentEvent]
        N: int,
    ) -> torch.Tensor:
        """Re-score clusters containing accident-affected nodes; return updated tensor.

        Accepts a single AccidentEvent or a list (for simultaneous-trigger waves).
        All affected nodes are merged into one set so one LLM call covers the wave.
        """
        lp       = self.llm_params
        clusters = self._cluster_assign_cache.get(inst['name'], [])
        K        = len(clusters)
        if not isinstance(accidents, list):
            accidents = [accidents]
        acc_set = set()
        for a in accidents:
            acc_set.update([a.node_a, a.node_b] + list(a.affected_nodes))
        visited = self._get_visited_set(N)
        cur_t   = float(self.env.current_time[0, 0].item())
        updated = cluster_conf.clone()

        # Filename includes the trigger time so distinct accident waves within the
        # same instance never share (and thus never collide on) a refresh cache.
        cache_path = os.path.join(
            self._llm_cache_dir, f'{inst["name"]}_acc_refresh_{cur_t:.1f}.json')

        if os.path.isfile(cache_path):
            # Post-accident scores pre-cached — load and apply visited mask
            with open(cache_path, encoding='utf-8') as f:
                cached_acc = json.load(f)
            _, full_flat = _parse_cluster_cache(cached_acc)
            for k, cluster_nodes in enumerate(clusters):
                for n in cluster_nodes:
                    if 1 <= n <= N:
                        s = full_flat.get(n, 0.0)
                        updated[k, n] = 0.0 if n in visited else float(s)
            return updated

        # First time: call LLM once for all affected clusters, save full scores
        full_flat: dict[int, float] = {}
        for k, cluster_nodes in enumerate(clusters):
            for n in cluster_nodes:
                if 1 <= n <= N:
                    full_flat[n] = float(cluster_conf[k, n])

        affected_clusters = [nodes for nodes in clusters if any(n in acc_set for n in nodes)]
        if affected_clusters:
            name = inst['name']
            cnt  = self._acc_prompt_counter.get(name, 0)
            self._acc_prompt_counter[name] = cnt + 1
            prompt_path = os.path.join(self._llm_cache_dir, f'{name}_acc_prompt_{cnt}.txt')

            # Prior confidence reference for remaining nodes (same spirit as rain baseline)
            prior_lines = []
            for k, cluster_nodes in enumerate(clusters):
                if not any(n in acc_set for n in cluster_nodes):
                    continue
                remaining = sorted(
                    [n for n in cluster_nodes if n not in visited and 1 <= n <= N],
                    key=lambda n: -float(cluster_conf[k, n]),
                )
                if remaining:
                    scores_str = ', '.join(f'node {n}: {float(cluster_conf[k, n]):.2f}' for n in remaining)
                    prior_lines.append(f'  Vehicle {k + 1}: {scores_str}')
            prior_ref = (
                '[Prior Confidence Before Accident (remaining stops only)]\n'
                + '\n'.join(prior_lines) + '\n'
            ) if prior_lines else ''

            _t_llm = time.time()
            try:
                scores = get_all_clusters_confidence(
                    inst, affected_clusters,
                    model=lp['model'], use_cot=lp['use_cot'], use_ontology=lp['use_ontology'],
                    cur_time=cur_t,
                    prompt_save_path=prompt_path,
                    reward_config=lp.get('reward_config', 'F'),
                    visited=visited,
                    experience_section=prior_ref,
                )
            except Exception as e:
                print(f'  [AccRefresh] {name}: LLM failed ({e}), using base confidence fallback.')
                scores = {}
            _llm_elapsed = time.time() - _t_llm
            self._timing['llm_refresh'] += _llm_elapsed
            for n, s in scores.items():
                if 1 <= n <= N:
                    full_flat[n] = float(s)
            for k, cluster_nodes in enumerate(clusters):
                for n in cluster_nodes:
                    if 1 <= n <= N and n in scores:
                        updated[k, n] = 0.0 if n in visited else float(scores[n])
            print(f"  [LLM:acc] {name} re-scored in {_llm_elapsed:.1f}s ({len(affected_clusters)} clusters, prompt→{prompt_path})")

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(_dump_cluster_cache(clusters, full_flat), f)
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
        acc_tt_applied   = [False] * len(accident_list)
        acc_tt_restored  = [False] * len(accident_list)
        active_accs      = []

        # Step 1: deterministic cluster starts (LLM mode) — skipped when free_starts=True or no clusters yet
        free_starts = self.trainer_params.get('free_starts', False)
        if llm_on and not free_starts and cluster_conf is not None:
            selected = self._det_starts(inst, pomo_size_eff).expand(batch_size, -1)
            prob     = torch.ones(batch_size, pomo_size_eff, device=self.device)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            step = 2
        else:
            step = 1

        while not done and step < max_steps:
            cur_t = state.current_time.mean().item()

            # NOTE: training instances treat accidents as known from episode start
            # (same as rain) -- cluster_conf already reflects the accident via the
            # pre-generated RL-cluster cache, so no mid-episode live LLM refresh here.
            # (Test-instance evaluation still does the real-time reveal; see _eval_test.)

            # ── Accident TT modification + encoder re-encode ──────────────
            acc_changed = False
            for i, (accident, fp) in enumerate(accident_list):
                if not acc_tt_applied[i] and cur_t >= fp['t_start']:
                    active_accs.append(dict(
                        idx=i, nodes=accident.affected_nodes,
                        t_start=fp['t_start'], t_end=fp['t_end'],
                        vehicles_involved=fp.get('vehicles_involved'),
                    ))
                    acc_tt_applied[i]  = True
                    acc_changed        = True
                elif acc_tt_applied[i] and not acc_tt_restored[i] and cur_t >= fp['t_end']:
                    active_accs = [a for a in active_accs if a['idx'] != i]
                    acc_tt_restored[i] = True
                    acc_changed        = True

            if acc_changed:
                self.env.reset_state.acc_tokens = _make_acc_tokens(
                    active_accs, N, batch_size, self.device
                )
                self.model.pre_forward(self.env.reset_state)

            # ── Per-vehicle per-step cluster confidence bias ──────────────
            # rollout i, vehicle j → cluster (i+j) % K
            if llm_on and cluster_conf is not None:
                veh_idx     = self.env.depot_visit_count.long()          # (B, P)
                cluster_idx = (veh_idx + roll_shifts[None, :]) % K_clusters  # (B, P)
                mask        = (veh_idx < K_clusters).float().unsqueeze(-1)   # (B, P, 1)

                # TW-aware suppression: zero confidence for nodes whose TW is already closed
                B_sz, P_sz = cluster_idx.shape
                N1         = self.env.problem_size + 1
                dev        = cluster_idx.device
                b_idx      = torch.arange(B_sz, device=dev)[:, None].expand(B_sz, P_sz)
                tt_to_j    = self.env.tt[b_idx.reshape(-1), self.env.current_node.reshape(-1)] \
                                 .reshape(B_sz, P_sz, N1)                    # (B, P, N+1)
                arrival_j  = self.env.current_time.unsqueeze(-1) + tt_to_j  # (B, P, N+1)
                tw_close_j = self.env.depot_node_tw_close.unsqueeze(1)      # (B, 1, N+1)
                tw_open    = (arrival_j <= tw_close_j).float()              # (B, P, N+1)

                conf_now = cluster_conf[cluster_idx] * mask * tw_open * bias_strength  # (B, P, N+1)
                selected, prob = self.model(state, llm_bias=conf_now)
            else:
                selected, prob = self.model(state)

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

            free_starts = self.trainer_params.get('free_starts', False)
            if llm_on and not free_starts:
                sel   = self._det_starts(inst, pomo_size_eff).expand(bs, -1)
                state, reward, done = self.env.step(sel)
                step = 2
            else:
                step = 1

            acc_tt_applied  = [False] * len(accident_list)
            acc_tt_restored = [False] * len(accident_list)
            active_accs     = []
            pending_accs    = []   # deferred for next-step batched LLM call

            while not done and step < max_steps:
                cur_t = state.current_time.mean().item()

                # ── Deferred LLM refresh ──────────────────────────────────
                if llm_on and pending_accs and cluster_conf is not None:
                    cluster_conf = self._refresh_clusters_accident(
                        inst, cluster_conf, pending_accs, N
                    )
                    pending_accs = []

                acc_changed      = False
                acc_just_started = []
                for i, (accident, fp) in enumerate(accident_list):
                    if not acc_tt_applied[i] and cur_t >= fp['t_start']:
                        active_accs.append(dict(
                            idx=i, nodes=accident.affected_nodes,
                            t_start=fp['t_start'], t_end=fp['t_end'],
                            vehicles_involved=fp.get('vehicles_involved'),
                        ))
                        acc_tt_applied[i] = True
                        acc_changed       = True
                        acc_just_started.append(accident)
                    elif acc_tt_applied[i] and not acc_tt_restored[i] and cur_t >= fp['t_end']:
                        active_accs = [a for a in active_accs if a['idx'] != i]
                        acc_tt_restored[i] = True
                        acc_changed        = True

                if acc_changed:
                    self.env.reset_state.acc_tokens = _make_acc_tokens(
                        active_accs, N, bs, self.device
                    )
                    self.model.pre_forward(self.env.reset_state)

                if llm_on and acc_just_started:
                    pending_accs.extend(acc_just_started)

                if llm_on and cluster_conf is not None:
                    veh_idx     = self.env.depot_visit_count.long()
                    cluster_idx = (veh_idx + roll_shifts[None, :]) % K_clusters
                    mask        = (veh_idx < K_clusters).float().unsqueeze(-1)
                    B_sz, P_sz = cluster_idx.shape
                    N1         = self.env.problem_size + 1
                    dev        = cluster_idx.device
                    b_idx      = torch.arange(B_sz, device=dev)[:, None].expand(B_sz, P_sz)
                    tt_to_j    = self.env.tt[b_idx.reshape(-1), self.env.current_node.reshape(-1)] \
                                     .reshape(B_sz, P_sz, N1)
                    tt_to_j    = tt_to_j * self.env._event_multiplier_row(
                        self.env.current_node, self.env.current_time)
                    arrival_j  = self.env.current_time.unsqueeze(-1) + tt_to_j
                    tw_close_j = self.env.depot_node_tw_close.unsqueeze(1)
                    tw_open    = (arrival_j <= tw_close_j).float()
                    conf_now    = cluster_conf[cluster_idx] * mask * tw_open * bias_strength
                    sel, _ = self.model(state, llm_bias=conf_now)
                else:
                    sel, _ = self.model(state)
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
        acc_tt_applied   = [False] * len(accident_list)
        acc_tt_restored  = [False] * len(accident_list)
        active_accs      = []
        pending_accs     = []   # deferred for next-step batched LLM call

        free_starts = self.trainer_params.get('free_starts', False)
        if llm_on and not free_starts:
            sel  = self._det_starts(inst, pomo_size_eff).expand(1, -1)
            state, reward, done = self.env.step(sel)
            step = 2
        else:
            step = 1

        while not done and step < max_steps:
            cur_t = state.current_time.mean().item()

            # ── Deferred LLM refresh: called at step after accident triggers ──
            if llm_on and pending_accs and cluster_conf is not None:
                cluster_conf = self._refresh_clusters_accident(
                    inst, cluster_conf, pending_accs, N
                )
                pending_accs = []

            acc_changed      = False
            acc_just_started = []
            for i, (accident, fp) in enumerate(accident_list):
                if not acc_tt_applied[i] and cur_t >= fp['t_start']:
                    active_accs.append(dict(
                        idx=i, nodes=accident.affected_nodes,
                        t_start=fp['t_start'], t_end=fp['t_end'],
                        vehicles_involved=fp.get('vehicles_involved'),
                    ))
                    acc_tt_applied[i]  = True
                    acc_changed        = True
                    acc_just_started.append(accident)
                elif acc_tt_applied[i] and not acc_tt_restored[i] and cur_t >= fp['t_end']:
                    active_accs = [a for a in active_accs if a['idx'] != i]
                    acc_tt_restored[i] = True
                    acc_changed        = True

            if acc_changed:
                self.env.reset_state.acc_tokens = _make_acc_tokens(active_accs, N, 1, self.device)
                self.model.pre_forward(self.env.reset_state)

            if llm_on and acc_just_started:
                pending_accs.extend(acc_just_started)

            if llm_on and cluster_conf is not None:
                veh_idx     = self.env.depot_visit_count.long()
                cluster_idx = (veh_idx + roll_shifts[None, :]) % K_clusters
                mask        = (veh_idx < K_clusters).float().unsqueeze(-1)
                B_sz, P_sz = cluster_idx.shape
                N1         = self.env.problem_size + 1
                dev        = cluster_idx.device
                b_idx      = torch.arange(B_sz, device=dev)[:, None].expand(B_sz, P_sz)
                tt_to_j    = self.env.tt[b_idx.reshape(-1), self.env.current_node.reshape(-1)] \
                                 .reshape(B_sz, P_sz, N1)
                tt_to_j    = tt_to_j * self.env._event_multiplier_row(
                    self.env.current_node, self.env.current_time)
                arrival_j  = self.env.current_time.unsqueeze(-1) + tt_to_j
                tw_close_j = self.env.depot_node_tw_close.unsqueeze(1)
                tw_open    = (arrival_j <= tw_close_j).float()
                conf_now    = cluster_conf[cluster_idx] * mask * tw_open * bias_strength
                sel, _ = self.model(state, llm_bias=conf_now)
            else:
                sel, _ = self.model(state)
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
    parser.add_argument('--no-ontology',    action='store_true')
    parser.add_argument('--resume',         default=None)
    parser.add_argument('--epochs',         type=int,   default=trainer_params['epochs'])
    parser.add_argument('--model',          default=llm_params['model'])
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--config',         default='E',
                        choices=list(_base_config._REWARD_CONFIGS))
    parser.add_argument('--refresh-cache',  action='store_true',
                        help='Ignore disk cache and call LLM fresh')
    parser.add_argument('--no-cot',         action='store_true')
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
    llm_params['model']                = args.model
    llm_params['reward_config']        = args.config
    llm_params['use_fewshot_cache']    = args.fewshot_cache
    trainer_params['free_starts']      = args.free_starts

    # Benchmark selection
    if args.benchmark == 'rc1':
        _B = [f"rc{i:03d}" for i in range(102, 109)]
        _E = [f"{n}_rain_A" for n in _B] + [f"{n}_rain_B" for n in _B]
        trainer_params['train_instances'] = _B + _E
        trainer_params['test_instances']  = [
            "rc101", "rc101_rain_A", "rc101_rain_B", "rc101_acc_A", "rc101_acc_B"
        ]
        print('[benchmark=rc1] Train: rc102-rc108 + rain  |  Test: rc101 (all scenarios)')
    elif args.benchmark == 'r1':
        _B = [f"r{i:03d}" for i in range(102, 113)]
        _E = [f"{n}_rain_A" for n in _B] + [f"{n}_rain_B" for n in _B]
        trainer_params['train_instances'] = _B + _E
        trainer_params['test_instances']  = [
            "r101", "r101_rain_A", "r101_rain_B", "r101_acc_A", "r101_acc_B"
        ]
        print('[benchmark=r1] Train: r102-r112 + rain  |  Test: r101 (all scenarios)')
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
    else:  # c1 default
        _B = [f"c{i:03d}" for i in range(102, 110)]
        _E = [f"{n}_rain_A" for n in _B] + [f"{n}_rain_B" for n in _B]
        trainer_params['train_instances'] = _B + _E
        trainer_params['test_instances']  = [
            "c101", "c101_rain_A", "c101_rain_B", "c101_acc_A", "c101_acc_B"
        ]
        print('[benchmark=c1] Train: c102-c109 + rain  |  Test: c101 (all scenarios)')

    # ACC instances in training
    if args.acc_ratio > 0:
        _cur        = trainer_params['train_instances']
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
            routes, reward = trainer._best_solution(inst)
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
