"""
SoftClusterOntology.py — TBox + ABox for Soft-Clustering POMO.

Extends the original VRPTW ontology with cluster-aware concepts.

TBox additions (vs original_POMO/VRPTWOntology.py)
----------------------------------------------------
  Stop [multi-label]
    PrimaryClusterStop(k)  : stop is in vehicle k's assigned soft cluster
    CrossClusterStop       : stop is outside the vehicle's assigned cluster
                             (RL can still visit it — action space is fully open)

    TW feasibility (evaluated at current step time, per-vehicle position):
    TW_ClosedStop          : current_time + tt[cur→c] > tw_close[c]
                             → cluster confidence multiplier = 0.0 (suppress)
    TW_OpenStop            : current_time + tt[cur→c] ≤ tw_close[c]
                             → cluster confidence multiplier = 1.0 (no change)

  All original TBox classes are preserved:
    OverdueStop, OvercapacityStop, DepotViolationStop, VisitedStop,
    Rain_<i>, Acc_<i>

ABox additions
--------------
  cluster_<k>            : SoftCluster instance (nodes, centroid)
  vehicle_<k>            : Vehicle assigned to cluster_<k>
  customer_c.primaryOf   : cluster_k (if c ∈ cluster_k.nodes)
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set


# ── Cross-episode statistics tracker ─────────────────────────────────────────
# Identical to VRPTWOntology — kept here so soft_pomo is self-contained.

class EpisodeStatsTracker:
    """Tracks per-(scenario, customer) late/unserved stats and instance-level metrics."""
    WINDOW_SIZE        = 100   # rolling window length
    BEST_K             = 3     # top-k best episodes to keep per scenario
    WORST_K            = 3     # bottom-k worst episodes to keep per scenario

    def __init__(self):
        self._stats: dict[str, dict[int, dict]] = defaultdict(
            lambda: defaultdict(lambda: {'unserved': 0, 'late': 0, 'total': 0})
        )
        # Per-node rolling window: deque of {'unserved': bool, 'late': bool, 'late_time': float}
        self._window: dict[str, dict[int, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=EpisodeStatsTracker.WINDOW_SIZE))
        )
        # Instance-level rolling window: deque of {'K': int, 'D': float, 'Lc': int, 'Lt': float, 'reward': float}
        self._inst_window: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=EpisodeStatsTracker.WINDOW_SIZE)
        )
        # Episode counter per scenario (used as episode ID in best/worst records)
        self._ep_count: dict[str, int] = defaultdict(int)
        # Top-k best and bottom-k worst episode records per scenario
        # Each record: {'ep': int, 'reward': float, 'Lc': int, 'Lt': float,
        #               'routes': list[list[int]], 'late_times': dict}
        self._best_buf:  dict[str, list] = defaultdict(list)
        self._worst_buf: dict[str, list] = defaultdict(list)

    def update(self, scenario: str, n_customers: int,
               unserved_ids: list[int], late_ids: list[int],
               late_times: dict = None,
               K: int = 0, D: float = 0.0, reward: float = 0.0,
               routes: list = None):
        """
        late_times: {node_id: late_amount_in_raw_time_units}, 0 for on-time nodes.
        K: vehicles used, D: total distance, reward: episode reward value.
        routes: list of vehicle routes (each route = list of customer node IDs).
        """
        unserved_set = set(unserved_ids)
        late_set     = set(late_ids)
        lt_map       = late_times or {}
        Lc           = len(late_set)
        Lt           = sum(lt_map.values())

        for c in range(1, n_customers + 1):
            s = self._stats[scenario][c]
            s['total']   += 1
            if c in unserved_set: s['unserved'] += 1
            if c in late_set:     s['late']     += 1
            self._window[scenario][c].append({
                'unserved':  c in unserved_set,
                'late':      c in late_set,
                'late_time': lt_map.get(c, 0.0),
            })

        self._inst_window[scenario].append({
            'K': K, 'D': D, 'Lc': Lc, 'Lt': Lt, 'reward': reward,
        })

        if routes is not None:
            self._ep_count[scenario] += 1
            ep = {
                'ep':        self._ep_count[scenario],
                'reward':    reward,
                'Lc':        Lc,
                'Lt':        round(Lt, 2),
                'routes':    [list(r) for r in routes],
                'late_times': {k: round(v, 2) for k, v in lt_map.items() if v > 0},
            }
            # maintain top-k best (highest reward = least penalty)
            buf = self._best_buf[scenario]
            buf.append(ep)
            buf.sort(key=lambda e: -e['reward'])
            self._best_buf[scenario] = buf[:self.BEST_K]
            # maintain bottom-k worst (lowest reward)
            wbuf = self._worst_buf[scenario]
            wbuf.append(ep)
            wbuf.sort(key=lambda e: e['reward'])
            self._worst_buf[scenario] = wbuf[:self.WORST_K]

    def best_episodes(self, scenario: str, k: int = None) -> list:
        """Return top-k best episode records (highest reward) for scenario."""
        k = k or self.BEST_K
        return self._best_buf.get(scenario, [])[:k]

    def worst_episodes(self, scenario: str, k: int = None) -> list:
        """Return bottom-k worst episode records (lowest reward) for scenario."""
        k = k or self.WORST_K
        return self._worst_buf.get(scenario, [])[:k]

    def rates(self, scenario: str, c: int) -> dict:
        s = self._stats[scenario].get(c)
        if not s or s['total'] == 0:
            return {'unserved_rate': 0.0, 'late_rate': 0.0, 'total': 0}
        t = s['total']
        return {
            'unserved_rate': round(s['unserved'] / t, 3),
            'late_rate':     round(s['late']     / t, 3),
            'total':         t,
        }

    def window_rates(self, scenario: str, c: int) -> Optional[dict]:
        """Recent-window per-node stats. Returns None if no samples."""
        w = self._window[scenario].get(c)
        if not w:
            return None
        n = len(w)
        return {
            'unserved_rate': sum(e['unserved']  for e in w) / n,
            'late_rate':     sum(e['late']       for e in w) / n,
            'avg_late_time': sum(e['late_time']  for e in w) / n,
            'total':         n,
        }

    def instance_window_stats(self, scenario: str) -> Optional[dict]:
        """Recent-window instance-level averages. Returns None if no samples."""
        w = self._inst_window.get(scenario)
        if not w:
            return None
        n = len(w)
        return {
            'avg_K':      sum(e['K']      for e in w) / n,
            'avg_D':      sum(e['D']      for e in w) / n,
            'avg_Lc':     sum(e['Lc']     for e in w) / n,
            'avg_Lt':     sum(e['Lt']     for e in w) / n,
            'avg_reward': sum(e['reward'] for e in w) / n,
            'total':      n,
        }

    def save_to_disk(self, path: str) -> None:
        """Serialize rolling-window and best/worst episode data to JSON."""
        import json
        data = {
            'window': {
                scenario: {
                    str(node): list(entries)
                    for node, entries in node_map.items()
                }
                for scenario, node_map in self._window.items()
            },
            'inst_window': {
                scenario: list(entries)
                for scenario, entries in self._inst_window.items()
            },
            'best_buf':  dict(self._best_buf),
            'worst_buf': dict(self._worst_buf),
            'ep_count':  dict(self._ep_count),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def load_from_disk(self, path: str) -> None:
        """Restore rolling-window and best/worst episode data previously saved."""
        import json
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        for scenario, node_map in data.get('window', {}).items():
            for node_str, entries in node_map.items():
                dq = deque(entries, maxlen=self.WINDOW_SIZE)
                self._window[scenario][int(node_str)] = dq
        for scenario, entries in data.get('inst_window', {}).items():
            self._inst_window[scenario] = deque(entries, maxlen=self.WINDOW_SIZE)
        for scenario, eps in data.get('best_buf', {}).items():
            self._best_buf[scenario] = eps
        for scenario, eps in data.get('worst_buf', {}).items():
            self._worst_buf[scenario] = eps
        for scenario, cnt in data.get('ep_count', {}).items():
            self._ep_count[scenario] = cnt


episode_tracker = EpisodeStatsTracker()


# ── SoftClusterOntology ───────────────────────────────────────────────────────

class SoftClusterOntology:
    """
    VRPTW ontology extended with soft-cluster membership concepts.

    Usage
    -----
    ont = SoftClusterOntology(inst)
    ont.set_clusters(clusters)          # list[list[int]], one list per vehicle

    # Per-cluster LLM context (only nodes in this cluster)
    ctx = ont.get_cluster_context(cluster_nodes, cluster_idx=k)

    # Per-stop concepts including cluster membership
    labels = ont.get_concepts(c, cluster_idx=k)
    """

    def __init__(self, inst: dict):
        T   = float(inst['T'])
        cap = float(inst['vehicle_capacity'])
        mc  = float(inst['max_coord'])

        self.T             = T
        self.cap           = cap
        self.scenario_name = inst.get('name', '')

        self.tt      = inst['tt'].numpy() * T
        self.tw_open = np.concatenate([inst['depot_tw_open'].numpy() * T,
                                       inst['node_tw_open'].numpy()  * T])
        self.tw_close= np.concatenate([inst['depot_tw_close'].numpy() * T,
                                       inst['node_tw_close'].numpy()  * T])
        self.service = np.concatenate([inst['depot_service'].numpy() * T,
                                       inst['node_service'].numpy()  * T])
        self.demands = np.concatenate([inst['depot_demand'].numpy() * cap,
                                       inst['node_demand'].numpy()  * cap])
        self.coords  = np.vstack([inst['depot_xy'].numpy() * mc,
                                  inst['node_xy'].numpy()  * mc])

        # ABox: RoutingEvent instances
        self.rain_events: list[dict]     = []
        self.accident_events: list[dict] = []
        for ev in inst.get('preset_events', []):
            if ev['type'] == 'RAIN':
                self.rain_events.append(dict(
                    rainfall_mm    = float(ev.get('rainfall_mm', 0.0)),
                    t_start        = float(ev['trigger_time']),
                    t_end          = float(ev['trigger_time']) + float(ev['duration']),
                    affected_nodes = list(ev['nodes']),
                ))
            elif ev['type'] == 'ACCIDENT':
                self.accident_events.append(dict(
                    severity       = ev.get('severity', 'medium'),
                    t_start        = float(ev['trigger_time']),
                    t_end          = float(ev['trigger_time']) + float(ev['duration']),
                    affected_nodes = list(ev['nodes']),
                ))

        # ABox: SoftCluster instances (set via set_clusters)
        self._clusters: list[list[int]] = []        # clusters[k] = [node_ids]
        self._node_to_cluster: dict[int, int] = {}  # node → primary cluster index
        self._avg_cluster_size: float = 0.0

    # ── Cluster ABox ──────────────────────────────────────────────────────────

    def set_clusters(self, clusters: List[List[int]]) -> None:
        """Register K-means cluster assignments as ABox instances."""
        self._clusters = clusters
        self._node_to_cluster = {}
        for k, nodes in enumerate(clusters):
            for n in nodes:
                self._node_to_cluster[n] = k
        sizes = [len(c) for c in clusters if c]
        self._avg_cluster_size = float(np.mean(sizes)) if sizes else 1.0

    def cluster_of(self, c: int) -> Optional[int]:
        """Return the primary cluster index for stop c, or None if unassigned."""
        return self._node_to_cluster.get(c)

    def cluster_density(self, cluster_idx: int) -> str:
        """Return 'sparse' / 'normal' / 'dense' for a cluster."""
        if cluster_idx >= len(self._clusters):
            return 'normal'
        size = len(self._clusters[cluster_idx])
        avg  = max(self._avg_cluster_size, 1.0)
        if size < avg / 2:
            return 'sparse'
        if size > avg * 1.5:
            return 'dense'
        return 'normal'

    # ── TW feasibility ────────────────────────────────────────────────────────

    def tw_slack(self, c: int, cur_node: int = 0, cur_time: float = 0.0) -> float:
        """TW slack = tw_close[c] - (cur_time + tt[cur_node → c]).
        Negative means deadline already passed from current position."""
        arrival = cur_time + float(self.tt[cur_node, c])
        return float(self.tw_close[c]) - arrival

    def tw_feasibility_label(self, c: int, cur_node: int = 0, cur_time: float = 0.0) -> str:
        """Return TW feasibility label: 'TW_ClosedStop' or 'TW_OpenStop'."""
        return 'TW_ClosedStop' if self.tw_slack(c, cur_node, cur_time) < 0 else 'TW_OpenStop'

    def tw_feasibility_multiplier(self, c: int, cur_node: int = 0, cur_time: float = 0.0) -> float:
        """Return confidence multiplier: TW_Closed → 0.0 (suppress), TW_Open → 1.0."""
        return 0.0 if self.tw_slack(c, cur_node, cur_time) < 0 else 1.0

    # ── TBox → ABox: multi-label concept classification ───────────────────────

    def get_concepts(
        self,
        c: int,
        cluster_idx: Optional[int] = None,
        cur_node: int = 0,
        cur_time: float = 0.0,
        visited: Optional[Set[int]] = None,
        remaining_capacity: Optional[float] = None,
    ) -> Set[str]:
        """Return all applicable TBox concepts for stop c (multi-label).

        cluster_idx: the vehicle's assigned cluster (for PrimaryClusterStop /
                     CrossClusterStop classification). Pass None to skip.
        """
        if visited and c in visited:
            return {"VisitedStop"}

        labels: Set[str] = set()

        # Capacity
        if remaining_capacity is not None and self.demands[c] > remaining_capacity + 1e-6:
            labels.add("OvercapacityStop")

        # Depot violation
        arrival_c   = cur_time + float(self.tt[cur_node, c])
        svc_start_c = max(arrival_c, float(self.tw_open[c]))
        return_arr  = svc_start_c + float(self.tt[c, 0])
        if return_arr > float(self.tw_close[0]) + 1e-6:
            labels.add("DepotViolationStop")

        # Overdue
        if svc_start_c > float(self.tw_close[c]) + 1e-6:
            labels.add("OverdueStop")

        # TW feasibility (direct travel from cur_node)
        labels.add(self.tw_feasibility_label(c, cur_node, cur_time))

        # Cluster membership (soft-clustering specific)
        if cluster_idx is not None and self._clusters:
            primary_k = self._node_to_cluster.get(c)
            if primary_k == cluster_idx:
                labels.add(f"PrimaryClusterStop({cluster_idx})")
            else:
                labels.add("CrossClusterStop")

        if not labels:
            labels.add("NormalStop")
        return labels

    # ── ABox: stop → event indices ────────────────────────────────────────────

    def get_stop_events(self, c: int) -> dict:
        return {
            'rain':     [i for i, ev in enumerate(self.rain_events)
                         if c in ev['affected_nodes']],
            'accident': [i for i, ev in enumerate(self.accident_events)
                         if c in ev['affected_nodes']],
        }

    # ── Full ABox context for LLM prompt (cluster-scoped) ────────────────────

    def get_cluster_context(
        self,
        cluster_nodes: List[int],
        cluster_idx: int,
        cur_node: int = 0,
        cur_time: float = 0.0,
        visited: Optional[Set[int]] = None,
        remaining_capacity: Optional[float] = None,
    ) -> dict:
        """Full ABox context scoped to a single vehicle's cluster.

        Only nodes in cluster_nodes are considered; CrossClusterStops
        are excluded from the prompt (they have zero confidence bias for
        this vehicle anyway).
        """
        MASK_ONLY = {"OvercapacityStop", "DepotViolationStop", "VisitedStop"}

        all_concepts = {
            c: self.get_concepts(c, cluster_idx, cur_node, cur_time,
                                 visited, remaining_capacity)
            for c in cluster_nodes
        }
        llm_concepts = {c: lbls - MASK_ONLY for c, lbls in all_concepts.items()}

        # Per-event stop groupings (only within this cluster)
        cluster_set   = set(cluster_nodes)
        stops_by_rain = {}
        for i, ev in enumerate(self.rain_events):
            stops_by_rain[i] = [c for c in cluster_nodes
                                 if c in ev['affected_nodes']]
        stops_by_acc = {}
        for i, ev in enumerate(self.accident_events):
            stops_by_acc[i] = [c for c in cluster_nodes
                                if c in ev['affected_nodes']]

        # Cross-episode stats for this cluster's stops
        cross_ep_stats = {}
        for c in cluster_nodes:
            r = episode_tracker.rates(self.scenario_name, c)
            if r['total'] > 0:
                cross_ep_stats[c] = r

        # TW feasibility labels per node (at time of LLM call: cur_time from depot)
        tw_feasibility_map: dict[int, str] = {}
        for c in cluster_nodes:
            tw_feasibility_map[c] = self.tw_feasibility_label(c, cur_node, cur_time)

        def _with(label):
            return sorted(c for c, lbls in llm_concepts.items()
                          if any(label in lbl for lbl in lbls))

        tw_closed = sorted(c for c, lbl in tw_feasibility_map.items() if lbl == 'TW_ClosedStop')

        return {
            "concepts":           llm_concepts,
            "overdue":            _with("OverdueStop"),
            "stops_by_rain":      stops_by_rain,
            "stops_by_acc":       stops_by_acc,
            "cross_ep_stats":     cross_ep_stats,
            "rain_events":        self.rain_events,
            "accident_events":    self.accident_events,
            # Cluster-specific metadata
            "cluster_idx":        cluster_idx,
            "cluster_size":       len(cluster_nodes),
            "avg_cluster_size":   self._avg_cluster_size,
            # TW feasibility (binary: closed vs open)
            "tw_feasibility_map": tw_feasibility_map,
            "tw_closed":          tw_closed,
        }

    # ── Backward-compatible get_context (for unscoped use) ───────────────────

    def get_context(
        self,
        unvisited: List[int],
        cur_node: int = 0,
        cur_time: float = 0.0,
        visited: Optional[Set[int]] = None,
        remaining_capacity: Optional[float] = None,
    ) -> dict:
        """Full ABox context without cluster filtering (for non-cluster LLM calls)."""
        MASK_ONLY = {"OvercapacityStop", "DepotViolationStop", "VisitedStop"}

        all_concepts = {
            c: self.get_concepts(c, None, cur_node, cur_time, visited, remaining_capacity)
            for c in unvisited
        }
        llm_concepts = {c: lbls - MASK_ONLY for c, lbls in all_concepts.items()}

        stops_by_rain = {}
        for i, ev in enumerate(self.rain_events):
            stops_by_rain[i] = [c for c in unvisited if c in ev['affected_nodes']]
        stops_by_acc = {}
        for i, ev in enumerate(self.accident_events):
            stops_by_acc[i] = [c for c in unvisited if c in ev['affected_nodes']]

        cross_ep_stats = {}
        for c in unvisited:
            r = episode_tracker.rates(self.scenario_name, c)
            if r['total'] > 0:
                cross_ep_stats[c] = r

        def _with(label):
            return sorted(c for c, lbls in llm_concepts.items() if label in lbls)

        return {
            "concepts":        llm_concepts,
            "overdue":         _with("OverdueStop"),
            "stops_by_rain":   stops_by_rain,
            "stops_by_acc":    stops_by_acc,
            "cross_ep_stats":  cross_ep_stats,
            "rain_events":     self.rain_events,
            "accident_events": self.accident_events,
        }

