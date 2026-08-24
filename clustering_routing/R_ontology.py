"""
VRPTW OWL Ontology — formal knowledge representation for the VRPTW solver.

Architecture:
  - OWL TBox (class hierarchy + property definitions) is built once at startup.
  - ABox (individuals) is re-populated each time the state changes.
  - VRPTWOntologyManager.update_state() creates/updates OWL individuals and
    applies EquivalentClass-style classification rules in Python.
  - get_owl_context_for_llm() serialises the classified individuals so that
    the LLM receives formally typed OWL knowledge, not raw Python dicts.

OWL class hierarchy (TBox):
  VRPTWEntity
  ├── Stop
  │   ├── UrgentStop            (end_slack ≤ URGENT_SLACK_THRESHOLD)
  │   ├── OverdueStop           (end_slack < 0)
  │   ├── HighPriorityStop      (penalty_multiplier > 1.0 AND unserved_rate > threshold)
  │   ├── InfeasibleStop        (depot-return violated → pruned from action set)
  │   ├── RainAffectedStop      (in active RainZone)
  │   ├── AccidentAffectedStop  (in active AccidentZone's affected_nodes)
  │   └── EventAtRiskStop       (UrgentStop/HighPriorityStop AND event-affected; pre-computed intersection)
  ├── Zone
  │   └── CriticalZone     (unserved_rate > CRITICAL_ZONE_THRESHOLD)
  ├── Event
  │   ├── RainZone         (hasRainfallMM, hasRainProbability, hasActiveStart/End)
  │   └── AccidentZone     (hasSeverity, hasActiveStart/End)
  └── Vehicle

OWL properties used:
  Stop:    hasStartSlack, hasEndSlack, hasVisitStatus,
           hasPenaltyMultiplier, hasUnservedRate, hasTravelTime, hasDepotTravelTime
  Zone:    hasZoneUnservedRate
  Vehicle: hasCurrentTime, hasCurrentNode
  Event:   hasActiveStart, hasActiveEnd,
           hasRainfallMM, hasRainProbability (RainZone),
           hasSeverity (AccidentZone)
  Object:  belongsToZone (Stop→Zone), adjacentTo (Zone↔Zone),
           inRainZone (Stop→RainZone), inAccidentZone (Stop→AccidentZone)

Stop time-window properties:
  hasStartSlack = tw_open - arrival_time
    Negative  → TW already open when vehicle arrives (no wait needed)
    Positive  → vehicle must wait this long before service can begin

  hasEndSlack = tw_close - arrival_time - service_time
    Positive  → slack buffer remaining after completing service
    Negative  → stop is overdue (cannot be served within TW)
    Small pos → urgent (use URGENT_SLACK_THRESHOLD to classify as UrgentStop)
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, List, Optional, Tuple

import numpy as np

# ── owlready2 import ──────────────────────────────────────────────────────────
try:
    from owlready2 import (
        AllDisjoint,
        DataProperty,
        FunctionalProperty,
        ObjectProperty,
        SymmetricProperty,
        Thing,
        ThingClass,
        destroy_entity,
        get_ontology,
    )
    _OWL_OK = True
except ImportError:  # pragma: no cover
    _OWL_OK = False
    raise ImportError("owlready2 is required: pip install owlready2")

# ── Default classification thresholds (reference only — actual values passed via constructor) ──
# These define the OWL EquivalentClass restrictions for derived concepts.
# Tunable per experiment via TrainConfig.ontology_urgent_slack_threshold etc.
_DEFAULT_URGENT_SLACK_THRESHOLD: float = 30.0   # Stop.slack ≤ this → UrgentStop
_DEFAULT_CRITICAL_ZONE_THRESHOLD: float = 0.5   # Zone.unserved_rate > this → CriticalZone

# ── Ontology IRI ──────────────────────────────────────────────────────────────
_ONTO_IRI = "http://vrptw.postech.ac.kr/ontology#"


# ============================================================
# Build TBox (class + property definitions) — done once globally
# ============================================================

_onto = get_ontology(_ONTO_IRI)

with _onto:
    # ── Top-level entity ──────────────────────────────────────────────────────
    class VRPTWEntity(Thing):
        """Root class for all VRPTW domain concepts."""

    # ── Stop hierarchy ────────────────────────────────────────────────────────
    class Stop(VRPTWEntity):
        """A customer stop that must be served within a time window."""

    class UrgentStop(Stop):
        """end_slack ≤ URGENT_SLACK_THRESHOLD. Must be visited very soon."""

    class OverdueStop(Stop):
        """end_slack < 0. Cannot be served within TW even if visited immediately."""

    class HighPriorityStop(Stop):
        """hasPenaltyMultiplier > 1.0 AND hasUnservedRate > threshold.
        Stop frequently missed across episodes; penalty is escalated by cross-episode reasoning."""

    class InfeasibleStop(Stop):
        """Depot-return constraint violated. Pruned from action set."""

    class RainAffectedStop(Stop):
        """In an active RainZone. Intra-zone edges slowed (both-endpoint rule)."""

    class AccidentAffectedStop(Stop):
        """In an active AccidentZone's affected_nodes."""

    class EventAtRiskStop(Stop):
        """(UrgentStop OR HighPriorityStop) AND (RainAffectedStop OR AccidentAffectedStop).
        Pre-computed intersection: tight TW combined with active event pressure."""

    # ── Zone hierarchy ────────────────────────────────────────────────────────
    class Zone(VRPTWEntity):
        """A geographic service zone assigned to one vehicle."""

    class CriticalZone(Zone):
        """hasZoneUnservedRate > 0.5. Persistently high miss rate across episodes."""

    # ── Event hierarchy ───────────────────────────────────────────────────────
    class Event(VRPTWEntity):
        """A dynamic event affecting travel times in the current episode."""

    class RainZone(Event):
        """Active rain event zone. Edges slowed only if both endpoints are in the same zone."""

    class AccidentZone(Event):
        """Active accident event. Typically blocks a single segment between two stops."""

    # ── Vehicle ───────────────────────────────────────────────────────────────
    class Vehicle(VRPTWEntity):
        """A delivery vehicle operating from the depot."""

    # ── Data Properties — Stop ────────────────────────────────────────────────
    class hasStartSlack(DataProperty, FunctionalProperty):
        """tw_open - arrival_time. Positive = must wait; negative = TW already open."""
        domain = [Stop]; range = [float]

    class hasEndSlack(DataProperty, FunctionalProperty):
        """tw_close - arrival_time. Negative = overdue (cannot arrive in time); small positive = urgent."""
        domain = [Stop]; range = [float]

    class hasVisitStatus(DataProperty, FunctionalProperty):
        """'served' or 'unserved' in the current episode."""
        domain = [Stop]; range = [str]

    class hasPenaltyMultiplier(DataProperty, FunctionalProperty):
        """Unserved penalty multiplier set by cross-episode reasoning (≥ 1.0)."""
        domain = [Stop]; range = [float]

    class hasUnservedRate(DataProperty, FunctionalProperty):
        """Fraction of past episodes in which this stop was not served."""
        domain = [Stop]; range = [float]

    class hasTravelTime(DataProperty, FunctionalProperty):
        """Travel time from current vehicle position to this stop."""
        domain = [Stop]; range = [float]

    class hasDepotTravelTime(DataProperty, FunctionalProperty):
        """Travel time from this stop back to the depot."""
        domain = [Stop]; range = [float]

    # ── Data Properties — Zone ────────────────────────────────────────────────
    class hasZoneUnservedRate(DataProperty, FunctionalProperty):
        """Average unserved rate across stops in this zone (cross-episode)."""
        domain = [Zone]; range = [float]

    # ── Data Properties — Vehicle ─────────────────────────────────────────────
    class hasCurrentTime(DataProperty, FunctionalProperty):
        """Current time of the vehicle in the episode."""
        domain = [Vehicle]; range = [float]

    class hasCurrentNode(DataProperty, FunctionalProperty):
        """Current location (node index) of the vehicle."""
        domain = [Vehicle]; range = [int]

    # ── Data Properties — Event ───────────────────────────────────────────────
    class hasActiveStart(DataProperty, FunctionalProperty):
        """Time at which this event becomes active (trigger_time)."""
        domain = [Event]; range = [float]

    class hasActiveEnd(DataProperty, FunctionalProperty):
        """Time at which this event ends (trigger_time + duration)."""
        domain = [Event]; range = [float]

    class hasRainfallMM(DataProperty, FunctionalProperty):
        """Rainfall intensity in mm/h."""
        domain = [RainZone]; range = [float]

    class hasRainProbability(DataProperty, FunctionalProperty):
        """Probability (0–100) that rain actually occurs."""
        domain = [RainZone]; range = [float]

    class hasSeverity(DataProperty, FunctionalProperty):
        """Accident severity: 'low', 'medium', or 'high'."""
        domain = [AccidentZone]; range = [str]

    # ── Object Properties ─────────────────────────────────────────────────────
    class belongsToZone(ObjectProperty, FunctionalProperty):
        """Stop → Zone assignment."""
        domain = [Stop]; range = [Zone]

    class adjacentTo(ObjectProperty, SymmetricProperty):
        """Spatial adjacency between zones (convex hull-based)."""
        domain = [Zone]; range = [Zone]

    class inRainZone(ObjectProperty):
        """Stop → RainZone. Both-endpoint rule: edge slowed only if both stops share a zone."""
        domain = [Stop]; range = [RainZone]

    class inAccidentZone(ObjectProperty):
        """Stop → AccidentZone. Stop is part of the blocked segment."""
        domain = [Stop]; range = [AccidentZone]


# ── Helper: OWL class membership list ────────────────────────────────────────
def _owl_classes(ind) -> List[str]:
    """Return sorted list of OWL class names for an individual (excluding Thing)."""
    return sorted(
        cls.name
        for cls in ind.is_a
        if isinstance(cls, ThingClass) and cls is not Thing
    )


# ============================================================
# VRPTWOntologyManager
# ============================================================

class VRPTWOntologyManager:
    """
    Manages the ABox (individual instances) for the VRPTW OWL ontology.

    Usage in training loop:
        mgr = VRPTWOntologyManager(inst, zone_plan)
        ...
        mgr.update_state(env, ontology_engine, candidate_facts)
        owl_context = mgr.get_owl_context_for_llm(feasible_actions)
        # → pass owl_context to LLM prompt
    """

    def __init__(
        self,
        inst,
        zone_plan,
        urgent_slack_threshold: float = 30.0,
        critical_zone_threshold: float = 0.5,
    ):
        self.inst = inst
        self.zone_plan = zone_plan
        self._onto = _onto          # shared TBox

        # OWL classification thresholds (passed from TrainConfig, tunable per experiment)
        self.urgent_slack_threshold = urgent_slack_threshold
        self.critical_zone_threshold = critical_zone_threshold

        # ABox individual caches (cleared on each update)
        self._stop_inds: Dict[int, object] = {}         # stop_id → OWL Stop individual
        self._zone_inds: Dict[int, object] = {}         # zone_id → OWL Zone individual
        self._rain_zone_inds: Dict[int, object] = {}    # event_idx → OWL RainZone individual
        self._accident_zone_inds: Dict[int, object] = {} # event_idx → OWL AccidentZone individual
        self._vehicle_ind = None
        self._last_owl_context: dict = {}   # last result of get_owl_context_for_llm (for event LLM calls)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _clear_abox(self) -> None:
        """Destroy all previously created ABox individuals."""
        for ind in list(self._stop_inds.values()):
            destroy_entity(ind)
        for ind in list(self._zone_inds.values()):
            destroy_entity(ind)
        for ind in list(self._rain_zone_inds.values()):
            destroy_entity(ind)
        for ind in list(self._accident_zone_inds.values()):
            destroy_entity(ind)
        if self._vehicle_ind is not None:
            destroy_entity(self._vehicle_ind)
        self._stop_inds = {}
        self._zone_inds = {}
        self._rain_zone_inds = {}
        self._accident_zone_inds = {}
        self._vehicle_ind = None

    def _make_stop(self, stop_id: int, travel: float, cur_time: float,
                   depot_travel: float,
                   unserved_rate: float, penalty_mult: float,
                   cap_ok: bool, depot_ok: bool,
                   rain_affected: bool = False,
                   accident_affected: bool = False) -> object:
        """
        Create and classify a Stop individual.

        Time-window slacks:
          start_slack = tw_open - arrival_time
            < 0 → TW open on arrival (good); > 0 → must wait
          end_slack   = tw_close - arrival_time  (arrival deadline slack; service extends after)
            < 0 → overdue (OverdueStop); ≤ threshold → UrgentStop

        OWL EquivalentClass restrictions applied manually:
          UrgentStop            ← 0 ≤ end_slack ≤ URGENT_SLACK_THRESHOLD
          OverdueStop           ← end_slack < 0
          HighPriorityStop      ← penalty_multiplier > 1.0
          InfeasibleStop        ← capacity violated OR depot-return violated
          RainAffectedStop      ← stop in active rain zone
          AccidentAffectedStop  ← stop in active accident's affected_nodes
          EventAtRiskStop       ← (Urgent OR HighPriority) AND (Rain OR Accident affected)
        """
        tw_open = float(self.inst.tw_open[stop_id])
        tw_close = float(self.inst.tw_close[stop_id])
        travel = float(travel)
        cur_time = float(cur_time)
        unserved_rate = float(unserved_rate)
        penalty_mult = float(penalty_mult)
        service_time = float(self.inst.service_time[stop_id])

        arrival_time = cur_time + travel
        start_slack = tw_open - arrival_time
        end_slack = tw_close - arrival_time  # arrival deadline slack (service extends after)

        with self._onto:
            ind = Stop(f"stop_{stop_id}")
            ind.hasStartSlack = float(start_slack)
            ind.hasEndSlack = float(end_slack)
            ind.hasUnservedRate = unserved_rate
            ind.hasPenaltyMultiplier = penalty_mult
            ind.hasTravelTime = float(travel)
            ind.hasDepotTravelTime = float(depot_travel)
            ind.hasVisitStatus = "unserved"

            # ── OWL classification ────────────────────────────────────────────
            types = [Stop]
            if not cap_ok or not depot_ok:
                types.append(InfeasibleStop)
            else:
                if end_slack < -1e-6:
                    types.append(OverdueStop)
                elif end_slack <= self.urgent_slack_threshold:
                    types.append(UrgentStop)
                if penalty_mult > 1.0 + 1e-6:
                    types.append(HighPriorityStop)

            is_urgent_class = UrgentStop in types or HighPriorityStop in types
            if rain_affected:
                types.append(RainAffectedStop)
            if accident_affected:
                types.append(AccidentAffectedStop)
            if is_urgent_class and (rain_affected or accident_affected):
                types.append(EventAtRiskStop)

            ind.is_a = types

        return ind

    def _make_zone(self, zone_id: int, zone_unserved_rate: float) -> object:
        """Create and classify a Zone individual."""
        zone_unserved_rate = float(zone_unserved_rate)
        with self._onto:
            ind = Zone(f"zone_{zone_id}")
            ind.hasZoneUnservedRate = zone_unserved_rate

            types = [Zone]
            if zone_unserved_rate > self.critical_zone_threshold:
                types.append(CriticalZone)
            ind.is_a = types

        return ind

    def _make_rain_zone(self, idx: int, rainfall_mm: float, probability: float,
                        active_start: float, active_end: float) -> object:
        """Create a RainZone individual for one active rain event."""
        with self._onto:
            ind = RainZone(f"rain_zone_{idx}")
            ind.hasRainfallMM = float(rainfall_mm)
            ind.hasRainProbability = float(probability)
            ind.hasActiveStart = float(active_start)
            ind.hasActiveEnd = float(active_end)
            ind.is_a = [RainZone]
        return ind

    def _make_accident_zone(self, idx: int, severity: str,
                            active_start: float, active_end: float) -> object:
        """Create an AccidentZone individual for one active accident event."""
        with self._onto:
            ind = AccidentZone(f"accident_zone_{idx}")
            ind.hasSeverity = severity
            ind.hasActiveStart = float(active_start)
            ind.hasActiveEnd = float(active_end)
            ind.is_a = [AccidentZone]
        return ind

    # ── Public API ────────────────────────────────────────────────────────────

    def update_state(
        self,
        env,
        ontology_engine,          # OntologyReasoningEngine (may be None)
        candidate_facts: list,    # List[ActionOntologyFact]
    ) -> None:
        """
        Rebuild the ABox for the current decision step.

        - Creates OWL individuals for zones, the active vehicle, candidates.
        - Applies EquivalentClass-based classification rules.
        - Sets adjacentTo links between zone individuals.
        """
        self._clear_abox()

        # ── Zone individuals ──────────────────────────────────────────────────
        ep_count = ontology_engine.episode_count if ontology_engine else 1

        if self.zone_plan is not None:
            for zi, zone_custs in enumerate(self.zone_plan.zones):
                if ep_count > 0 and ontology_engine is not None:
                    rates = [
                        ontology_engine.unserved_counts[c] / ep_count
                        for c in zone_custs
                    ]
                    zone_rate = float(np.mean(rates)) if rates else 0.0
                else:
                    zone_rate = 0.0
                self._zone_inds[zi] = self._make_zone(zi, zone_rate)

            # Set adjacentTo links
            with self._onto:
                for zi, adj_set in self.zone_plan.adjacent_zones.items():
                    zi_ind = self._zone_inds.get(zi)
                    for zj in adj_set:
                        if zj != zi:
                            zj_ind = self._zone_inds.get(zj)
                            if zi_ind and zj_ind:
                                if zj_ind not in zi_ind.adjacentTo:
                                    zi_ind.adjacentTo.append(zj_ind)

        # ── Active event OWL individuals + stop membership sets ──────────────
        rain_affected_stops: set = set()        # stop_id → rain zone individual
        accident_affected_stops: set = set()
        stop_to_rain_zone: Dict[int, object] = {}      # stop_id → RainZone ind
        stop_to_accident_zone: Dict[int, object] = {}  # stop_id → AccidentZone ind
        rain_ev_idx = 0
        accident_ev_idx = 0
        for event in getattr(env, "active_events", []):
            if not event.active or event.recovered:
                continue
            nodes = set(event.affected_nodes)
            etype = event.event_type.name if hasattr(event.event_type, "name") else str(event.event_type)
            if "RAIN" in etype:
                rz_ind = self._make_rain_zone(
                    idx=rain_ev_idx,
                    rainfall_mm=getattr(event, "rainfall_mm", 0.0),
                    probability=getattr(event, "probability", 100.0),
                    active_start=event.trigger_time,
                    active_end=event.trigger_time + event.duration,
                )
                self._rain_zone_inds[rain_ev_idx] = rz_ind
                rain_ev_idx += 1
                rain_affected_stops |= nodes
                for n in nodes:
                    stop_to_rain_zone[n] = rz_ind
            elif "ACCIDENT" in etype:
                az_ind = self._make_accident_zone(
                    idx=accident_ev_idx,
                    severity=getattr(event, "severity", ""),
                    active_start=event.trigger_time,
                    active_end=event.trigger_time + event.duration,
                )
                self._accident_zone_inds[accident_ev_idx] = az_ind
                accident_ev_idx += 1
                accident_affected_stops |= nodes
                for n in nodes:
                    stop_to_accident_zone[n] = az_ind

        # ── Stop individuals ──────────────────────────────────────────────────
        for fact in candidate_facts:
            a = fact.action
            if a == 0:
                continue

            unserved_rate = 0.0
            penalty_mult = 1.0
            if ontology_engine is not None and ep_count > 0:
                unserved_rate = ontology_engine.unserved_counts[a] / ep_count
                penalty_mult = float(ontology_engine.penalty_multipliers[a])

            cap_ok = fact.capacity.satisfied
            depot_ok = fact.depot_return.satisfied

            stop_ind = self._make_stop(
                stop_id=a,
                travel=fact.time_window.travel_time,
                cur_time=float(env.cur_time),
                depot_travel=float(env.tt[a, 0]),
                unserved_rate=unserved_rate,
                penalty_mult=penalty_mult,
                cap_ok=cap_ok,
                depot_ok=depot_ok,
                rain_affected=(a in rain_affected_stops),
                accident_affected=(a in accident_affected_stops),
            )
            self._stop_inds[a] = stop_ind

            # link stop → routing zone
            if self.zone_plan is not None:
                zi = self.zone_plan.customer_to_zone.get(a)
                if zi is not None and zi in self._zone_inds:
                    with self._onto:
                        stop_ind.belongsToZone = self._zone_inds[zi]

            # link stop → rain/accident zone individuals
            with self._onto:
                if a in stop_to_rain_zone:
                    stop_ind.inRainZone = [stop_to_rain_zone[a]]
                if a in stop_to_accident_zone:
                    stop_ind.inAccidentZone = [stop_to_accident_zone[a]]

    def get_owl_context_for_llm(
        self,
        feasible_actions: List[int],
        active_zone_idx: Optional[int],
        base_penalty: float,
        env=None,
    ) -> dict:
        """
        Serialise the current ABox into a structured dict for LLM consumption.

        Format (슬라이드 구조 + 실제값):
          {
            "vehicle": {
              "current_location": int,
              "current_time": float
            },
            "zones": [
              {
                "zone_id": int,
                "zone_class": str,           # NormalZone / CriticalZone
                "is_active": bool,
                "unserved_rate": float,      # cross-episode average
                "adjacent_zone_ids": [int]
              }
            ],
            "candidates": [
              {
                "stop_id": int,
                "stop_classes": [str],       # UrgentStop / OverdueStop / HighPriorityStop / NormalStop
                "is_feasible": bool,
                "violated_constraints": [str],
                "zone_id": int,
                "zone_class": str,
                "is_same_zone": bool,
                "properties": {              # static stop properties
                  "time_window": {"start": float, "end": float},
                  "penalty_multiplier": float,
                  "unserved_rate": float
                },
                "step_context": {            # dynamic values computed for this step
                  "travel_time": float,
                  "start_slack": float,      # tw_open - arrival_time (neg = TW open)
                  "end_slack": float         # tw_close - arrival_time (neg = overdue)
                }
              }
            ]
          }
        """
        # ── Vehicle ───────────────────────────────────────────────────────────
        veh_info: dict = {
            "current_location": int(env.cur_node) if env is not None else -1,
            "current_time": round(float(env.cur_time), 2) if env is not None else 0.0,
        }

        # ── Zones ─────────────────────────────────────────────────────────────
        zone_infos: List[dict] = []
        for zi, z_ind in sorted(self._zone_inds.items()):
            z_classes = _owl_classes(z_ind)
            specific_zone = next(
                (c for c in z_classes if c not in ("Zone", "VRPTWEntity", "Thing")),
                "NormalZone"
            )
            adj_ids = [
                int(adj.name.split("_")[-1])
                for adj in getattr(z_ind, "adjacentTo", [])
            ]
            zone_infos.append({
                "zone_id": zi,
                "zone_class": specific_zone,
                "is_active": zi == active_zone_idx,
                "unserved_rate": round(z_ind.hasZoneUnservedRate, 3),
                "adjacent_zone_ids": sorted(adj_ids),
            })

        # ── Candidates ────────────────────────────────────────────────────────
        candidate_infos: List[dict] = []

        # Depot action
        if 0 in feasible_actions:
            candidate_infos.append({
                "stop_id": 0,
                "stop_classes": ["DepotReturn"],
                "is_feasible": True,
                "violated_constraints": [],
                "zone_id": None,
                "zone_class": None,
                "is_same_zone": False,
                "properties": {
                    "time_window": {"start": 0, "end": 0},
                    "penalty_multiplier": 1.0,
                    "unserved_rate": 0.0,
                },
                "step_context": {
                    "travel_time": 0.0,
                    "start_slack": 0.0,
                    "end_slack": 0.0,
                },
            })

        for a in feasible_actions:
            if a == 0:
                continue
            stop_ind = self._stop_inds.get(a)
            if stop_ind is None:
                continue

            stop_classes = _owl_classes(stop_ind)
            # Remove base class names; keep only derived subclasses
            derived = [c for c in stop_classes if c not in ("Stop", "VRPTWEntity")]
            if not derived:
                derived = ["NormalStop"]

            is_feasible = "InfeasibleStop" not in stop_classes
            violated = []
            if not is_feasible:
                violated.append("CapacityConstraint or DepotReturnConstraint")

            # Zone context from belongsToZone link
            zone_id = None
            zone_class = "NormalZone"
            z_ind = getattr(stop_ind, "belongsToZone", None)
            if z_ind is not None:
                zone_id = int(z_ind.name.split("_")[-1])
                z_cls = _owl_classes(z_ind)
                zone_class = next(
                    (c for c in z_cls if c not in ("Zone", "VRPTWEntity", "Thing")),
                    "NormalZone"
                )

            pen_mult = getattr(stop_ind, "hasPenaltyMultiplier", 1.0) or 1.0

            candidate_infos.append({
                "stop_id": a,
                "stop_classes": derived,
                "is_feasible": is_feasible,
                "violated_constraints": violated,
                "zone_id": zone_id,
                "zone_class": zone_class,
                "is_same_zone": zone_id == active_zone_idx,
                "properties": {
                    "penalty_multiplier": round(pen_mult, 2),
                    "unserved_rate": round(stop_ind.hasUnservedRate, 3),
                },
                "step_context": {
                    "travel_time": round(stop_ind.hasTravelTime, 2),
                    "depot_travel_time": round(stop_ind.hasDepotTravelTime, 2),
                    "start_slack": round(stop_ind.hasStartSlack, 2),
                    "end_slack": round(stop_ind.hasEndSlack, 2),
                },
            })

        # ── TW context (Phase 1 OWL — no event info) ─────────────────────────
        tw_context: dict = {
            "urgent_stops": [],
            "overdue_stops": [],
            "high_priority_stops": [],
            "infeasible_stops": [],
        }
        for a, stop_ind in self._stop_inds.items():
            cls = _owl_classes(stop_ind)
            if "UrgentStop" in cls:
                tw_context["urgent_stops"].append(a)
            if "OverdueStop" in cls:
                tw_context["overdue_stops"].append(a)
            if "HighPriorityStop" in cls:
                tw_context["high_priority_stops"].append(a)
            if "InfeasibleStop" in cls:
                tw_context["infeasible_stops"].append(a)

        # ── Event context (Phase 2 OWL — rain/accident zone info) ────────────
        rain_zones_info: List[dict] = []
        for rz_ind in self._rain_zone_inds.values():
            stops_in_zone = [
                a for a, s_ind in self._stop_inds.items()
                if getattr(s_ind, "inRainZone", None) and rz_ind in s_ind.inRainZone
            ]
            rain_zones_info.append({
                "zone_id": rz_ind.name,
                "rainfall_mm": round(rz_ind.hasRainfallMM, 1),
                "probability": round(rz_ind.hasRainProbability, 1),
                "active_start": round(rz_ind.hasActiveStart, 1),
                "active_end": round(rz_ind.hasActiveEnd, 1),
                "stops_in_zone": sorted(stops_in_zone),
            })

        accident_zones_info: List[dict] = []
        for az_ind in self._accident_zone_inds.values():
            stops_in_zone = [
                a for a, s_ind in self._stop_inds.items()
                if getattr(s_ind, "inAccidentZone", None) and az_ind in s_ind.inAccidentZone
            ]
            accident_zones_info.append({
                "zone_id": az_ind.name,
                "severity": az_ind.hasSeverity,
                "active_start": round(az_ind.hasActiveStart, 1),
                "active_end": round(az_ind.hasActiveEnd, 1),
                "stops_in_zone": sorted(stops_in_zone),
            })

        event_at_risk = [
            a for a, stop_ind in self._stop_inds.items()
            if "EventAtRiskStop" in _owl_classes(stop_ind)
        ]
        event_context: dict = {
            "rain_zones": rain_zones_info,
            "accident_zones": accident_zones_info,
            "event_at_risk_stops": sorted(event_at_risk),
            "rain_affected_stops": sorted(
                a for a, s in self._stop_inds.items()
                if "RainAffectedStop" in _owl_classes(s)
            ),
            "accident_affected_stops": sorted(
                a for a, s in self._stop_inds.items()
                if "AccidentAffectedStop" in _owl_classes(s)
            ),
        }

        # ── TW Propagation Reasoning (ontology-as-reasoner) ──────────────────
        tw_reasoning = self.compute_tw_propagation_reasoning(feasible_actions, env)

        self._last_owl_context = {
            "vehicle": veh_info,
            "zones": zone_infos,
            "candidates": candidate_infos,
            "tw_context": tw_context,
            "event_context": event_context,
            "tw_reasoning": tw_reasoning,
        }
        return self._last_owl_context

    def compute_tw_propagation_reasoning(
        self,
        feasible_actions: List[int],
        env,
    ) -> Dict[int, dict]:
        """
        For each candidate stop, compute TW cascade effects on remaining stops.

        Returns dict: stop_id → {
            arrival_time, deadline_slack, wait_time,
            infeasible_downstream: [stop_id],       # stops that miss TW if visited after this stop
            tight_downstream: [(stop_id, slack)],   # stops with slack ≤ threshold after this stop
        }

        This is deterministic constraint propagation — the reasoner, not the LLM, does the math.
        LLM receives the pre-computed causal consequences and interprets which trade-off is best.
        """
        if env is None:
            return {}

        cur_time = float(env.cur_time)
        cur_node = int(env.cur_node)
        tt = env.tt
        inst = self.inst
        threshold = self.urgent_slack_threshold

        candidates = [a for a in feasible_actions if a != 0]
        if not candidates:
            return {}

        result: Dict[int, dict] = {}
        for a in candidates:
            travel_a = float(tt[cur_node, a])
            arr_a = cur_time + travel_a
            dep_a = max(arr_a, float(inst.tw_open[a])) + float(inst.service_time[a])

            infeasible_after: List[int] = []
            tight_after: List[Tuple[int, float]] = []

            for b in candidates:
                if b == a:
                    continue
                arr_b = dep_a + float(tt[a, b])
                slack_b = float(inst.tw_close[b]) - arr_b
                if slack_b < 0:
                    infeasible_after.append(b)
                elif slack_b <= threshold:
                    tight_after.append((b, round(slack_b, 1)))

            result[a] = {
                "arrival_time": round(arr_a, 1),
                "deadline_slack": round(float(inst.tw_close[a]) - arr_a, 1),
                "wait_time": round(max(0.0, float(inst.tw_open[a]) - arr_a), 1),
                "infeasible_downstream": sorted(infeasible_after),
                "tight_downstream": sorted(tight_after, key=lambda x: x[1])[:5],
            }

        return result

    def get_zone_state_summary(self) -> List[dict]:
        """Return per-zone OWL classification summary (for logging/debugging)."""
        result = []
        for zi, z_ind in sorted(self._zone_inds.items()):
            result.append({
                "zone_id": zi,
                "owl_classes": _owl_classes(z_ind),
                "zone_unserved_rate": z_ind.hasZoneUnservedRate,
            })
        return result

    def save_owl(self, path: str) -> None:
        """Serialise the current ontology (TBox + ABox) to OWL/XML for Protégé."""
        self._onto.save(file=path, format="rdfxml")
        print(f"[OWL] Saved ontology to {path}")


# ============================================================
# OntologyReasoningEngine
# ============================================================

class OntologyReasoningEngine:
    """
    Cross-episode ontology reasoning engine using OWL classifications.

    Tracks per-stop unserved history and applies:

    Rule 1 — HighUnservedClassification (OWL: HighUnservedStop):
        IF stop.unserved_rate > penalty_threshold
        THEN classify as HighUnservedStop → passed to LLM as context
    """

    def __init__(
        self,
        inst,
        zone_plan,
        base_unserved_penalty: float,
        penalty_threshold: float,
        penalty_scale_factor: float,
        penalty_max_multiplier: float,
        num_episodes: int = 300,
        unserved_window_ratio: float = 0.5,
        min_warmup_ratio: float = 0.1,
    ):
        self.inst = inst
        self.zone_plan = zone_plan
        self.base_penalty = base_unserved_penalty
        self.penalty_threshold = penalty_threshold
        self.penalty_scale_factor = penalty_scale_factor
        self.penalty_max_multiplier = penalty_max_multiplier

        self.chunk_size: int = max(1, int(num_episodes * unserved_window_ratio))
        self.min_warmup_episodes: int = max(1, int(num_episodes * min_warmup_ratio))

        n = inst.n_customers + 1
        self.unserved_counts = np.zeros(n, dtype=np.int32)
        self.episode_count: int = 0
        self.penalty_multipliers = np.ones(n, dtype=np.float32)

        # chunk-based classification state
        self._chunk_unserved_counts = np.zeros(n, dtype=np.int32)
        self._chunk_episode_count: int = 0
        self._high_unserved_cache: Dict[int, List[Tuple[int, float]]] = {}  # zone_idx → [(stop, rate)]
        self._chunk_fired: bool = False  # True when chunk just completed this episode

    def record_episode(self, served: np.ndarray) -> bool:
        """Record which customers were unserved in the just-finished episode.

        Returns True if a chunk boundary was crossed (high_unserved cache updated).
        """
        self.episode_count += 1
        self._chunk_fired = False

        for c in range(1, self.inst.n_customers + 1):
            if served[c] == 0:
                self.unserved_counts[c] += 1
                self._chunk_unserved_counts[c] += 1

        self._chunk_episode_count += 1

        if (self.episode_count >= self.min_warmup_episodes
                and self._chunk_episode_count >= self.chunk_size):
            self._update_high_unserved_cache()
            self._chunk_unserved_counts[:] = 0
            self._chunk_episode_count = 0
            self._chunk_fired = True

        return self._chunk_fired

    def _update_high_unserved_cache(self) -> None:
        """Recompute high_unserved_cache and penalty_multipliers from current chunk counts.

        Uses relative threshold: only stops with rate > zone_mean are flagged.
        When all stops fail equally (e.g. rate=1.0 for all), no stops are flagged
        since there is no discriminative signal to pass to the LLM.
        """
        if self._chunk_episode_count == 0:
            return
        self._high_unserved_cache = {}
        any_flagged = False
        for z, stops in enumerate(self.zone_plan.zones):
            rates = {
                c: self._chunk_unserved_counts[c] / self._chunk_episode_count
                for c in stops
            }
            zone_mean = float(np.mean(list(rates.values()))) if rates else 0.0
            result = []
            for c, rate in rates.items():
                if rate > self.penalty_threshold and rate > zone_mean + 1e-6:
                    result.append((c, rate))
                    mult = min(
                        self.penalty_multipliers[c] + self.penalty_scale_factor * rate,
                        self.penalty_max_multiplier,
                    )
                    self.penalty_multipliers[c] = float(mult)
            self._high_unserved_cache[z] = sorted(result, key=lambda x: -x[1])
            if result:
                any_flagged = True
                stop_info = ", ".join(
                    f"stop{c}(rate={r:.2f} pen={self.penalty_multipliers[c]:.2f}x)"
                    for c, r in self._high_unserved_cache[z]
                )
                print(f"  [OWL:Zone{z}] HighPriorityStops: {stop_info}")
        if not any_flagged:
            print(f"  [OWL:ChunkFired] ep={self.episode_count} → no high-unserved stops")
        else:
            print(f"  [OWL:ChunkFired] ep={self.episode_count} chunk_size={self.chunk_size} "
                  f"→ multipliers updated")

    def get_stop_penalties(self) -> np.ndarray:
        """Returns per-stop unserved penalties scaled by cross-episode multipliers."""
        return self.base_penalty * self.penalty_multipliers

    def get_high_unserved_stops(self, zone_idx: int, zone_plan) -> List[Tuple[int, float]]:
        """
        OWL Rule 1 — HighUnservedStop classification.
        Returns cached result from last chunk boundary. Empty until first chunk fires.
        """
        return self._high_unserved_cache.get(zone_idx, [])


if __name__ == "__main__":
    # TBox-only export for Protégé visualization.
    # ABox (episode instances) is empty — only class hierarchy and properties are saved.
    _out = "vrptw_ontology.owl"
    _onto.save(file=_out, format="rdfxml")
    print(f"[OWL] TBox saved → {_out}")

