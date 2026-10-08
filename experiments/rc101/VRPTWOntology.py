"""
VRPTW Ontology — TBox + ABox for LLM prompt enrichment and RL reward shaping.

TBox (class definitions)
------------------------
  Stop  [multi-label]
    OverdueStop          : arrival_time > tw_end
    OvercapacityStop     : demand > remaining capacity  → action masking
    DepotViolationStop   : cannot return to depot in time  → action masking
    VisitedStop          : already served this episode
    Rain_<i>             : affected by RainEvent i specifically
    Acc_<i>              : affected by AccidentEvent i specifically
    UnservedStop(r%)     : unserved in r% of past episodes
    LatedStop(r%)        : late in r% of past episodes
    NormalStop           : none of the above

  RoutingEvent
    RainEvent      : rainfall_mm, t_start, t_end, affected_nodes
    AccidentEvent  : vehicles_involved, t_start, t_end, affected_nodes

ABox (instances per episode)
----------------------------
  rain_<i> / acc_<i>  : RoutingEvent instances from preset_events
  customer_c          : set of Stop labels + affected_by event links
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict


# ── Cross-episode statistics tracker ─────────────────────────────────────────

class EpisodeStatsTracker:
    """
    Tracks per-(scenario, customer) violation rates across training episodes.
    Updated after every episode; read by VRPTWOntology for UnservedStop/LatedStop.
    """
    UNSERVED_THRESHOLD = 0.3
    LATED_THRESHOLD    = 0.3

    def __init__(self):
        self._stats: dict[str, dict[int, dict]] = defaultdict(
            lambda: defaultdict(lambda: {'unserved': 0, 'late': 0, 'total': 0})
        )

    def update(self, scenario: str, n_customers: int,
               unserved_ids: list[int], late_ids: list[int]):
        unserved_set = set(unserved_ids)
        late_set     = set(late_ids)
        for c in range(1, n_customers + 1):
            s = self._stats[scenario][c]
            s['total']   += 1
            if c in unserved_set: s['unserved'] += 1
            if c in late_set:     s['late']     += 1

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

    def is_unserved(self, scenario: str, c: int) -> bool:
        return self.rates(scenario, c)['unserved_rate'] >= self.UNSERVED_THRESHOLD

    def is_lated(self, scenario: str, c: int) -> bool:
        return self.rates(scenario, c)['late_rate'] >= self.LATED_THRESHOLD

    def violation_weight(self, scenario: str, c: int) -> float:
        """Ontology-derived late penalty multiplier for RL reward shaping."""
        r = self.rates(scenario, c)
        return 1.0 + 2.0 * r['late_rate'] + 3.0 * r['unserved_rate']


episode_tracker = EpisodeStatsTracker()


# ── Ontology ──────────────────────────────────────────────────────────────────

class VRPTWOntology:

    def __init__(self, inst: dict):
        T   = float(inst['T'])
        cap = float(inst['vehicle_capacity'])
        mc  = float(inst['max_coord'])

        self.T             = T
        self.cap           = cap
        self.scenario_name = inst.get('name', '')

        self.tt      = inst['tt'].numpy() * T
        self.tw_open = np.concatenate([inst['depot_tw_open'].numpy()  * T,
                                        inst['node_tw_open'].numpy()  * T])
        self.tw_close= np.concatenate([inst['depot_tw_close'].numpy() * T,
                                        inst['node_tw_close'].numpy() * T])
        self.service = np.concatenate([inst['depot_service'].numpy()  * T,
                                        inst['node_service'].numpy()  * T])
        self.demands = np.concatenate([inst['depot_demand'].numpy()   * cap,
                                        inst['node_demand'].numpy()   * cap])
        self.coords  = np.vstack([inst['depot_xy'].numpy() * mc,
                                   inst['node_xy'].numpy() * mc])

        # ABox: RoutingEvent instances
        self.rain_events     = []
        self.accident_events = []
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
                    vehicles_involved = ev.get('vehicles_involved'),
                    multiplier        = float(ev['multiplier']),
                    t_start           = float(ev['trigger_time']),
                    t_end             = float(ev['trigger_time']) + float(ev['duration']),
                    affected_nodes    = list(ev['nodes']),
                ))

    # ── TBox → ABox: multi-label concept classification ───────────────────────

    def get_concepts(
        self,
        c: int,
        cur_node: int = 0,
        cur_time: float = 0.0,
        visited: set | None = None,
        remaining_capacity: float | None = None,
    ) -> set[str]:
        """Return all applicable TBox concepts for stop c (multi-label)."""
        if visited and c in visited:
            return {"VisitedStop"}

        labels = set()

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

        # Note: Rain_i / Acc_i are ABox property assertions (event membership),
        # not Stop concepts — they appear only in get_stop_events() / get_context()

        # Cross-episode
        r = episode_tracker.rates(self.scenario_name, c)
        if r['total'] > 0:
            if r['unserved_rate'] >= EpisodeStatsTracker.UNSERVED_THRESHOLD:
                labels.add(f"UnservedStop({int(r['unserved_rate']*100)}%)")
            if r['late_rate'] >= EpisodeStatsTracker.LATED_THRESHOLD:
                labels.add(f"LatedStop({int(r['late_rate']*100)}%)")

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

    # ── Full ABox context for LLM prompt ──────────────────────────────────────

    def get_context(
        self,
        unvisited: list,
        cur_node: int = 0,
        cur_time: float = 0.0,
        visited: set | None = None,
        remaining_capacity: float | None = None,
    ) -> dict:
        """Full ABox context for LLM. Masking-only concepts excluded."""
        MASK_ONLY = {"OvercapacityStop", "DepotViolationStop", "VisitedStop"}

        all_concepts = {
            c: self.get_concepts(c, cur_node, cur_time, visited, remaining_capacity)
            for c in unvisited
        }
        llm_concepts = {c: lbls - MASK_ONLY for c, lbls in all_concepts.items()}

        # Per-event stop groupings
        stops_by_rain = {}
        for i in range(len(self.rain_events)):
            stops_by_rain[i] = [c for c in unvisited
                                 if c in self.rain_events[i]['affected_nodes']]
        stops_by_acc = {}
        for i in range(len(self.accident_events)):
            stops_by_acc[i] = [c for c in unvisited
                                if c in self.accident_events[i]['affected_nodes']]

        # Cross-episode stats for unvisited stops
        cross_ep_stats = {}
        for c in unvisited:
            r = episode_tracker.rates(self.scenario_name, c)
            if r['total'] > 0:
                cross_ep_stats[c] = r

        def _with(label):
            return sorted(c for c, lbls in llm_concepts.items() if label in lbls)

        return {
            "concepts":         llm_concepts,
            "overdue":          _with("OverdueStop"),
            "stops_by_rain":    stops_by_rain,       # {event_idx: [stop_ids]}
            "stops_by_acc":     stops_by_acc,        # {event_idx: [stop_ids]}
            "cross_ep_stats":   cross_ep_stats,      # {stop_id: {unserved_rate, late_rate, total}}
            "rain_events":      self.rain_events,
            "accident_events":  self.accident_events,
        }

    # ── Reward shaping ────────────────────────────────────────────────────────

    def late_weight(self, c: int) -> float:
        """Ontology-derived late penalty multiplier for RL reward shaping."""
        return episode_tracker.violation_weight(self.scenario_name, c)
