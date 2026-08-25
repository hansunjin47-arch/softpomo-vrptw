"""
generate_events.py -- Generate randomised VRPTW event scenario files.

For each (instance, scenario index), generates 4 files:
  {name}_rain_A[_{idx}].txt  -- rain, HIGH IMPACT  (active during routing)
  {name}_rain_B[_{idx}].txt  -- rain, NO IMPACT    (ends before first arrival)
  {name}_acc_A[_{idx}].txt   -- accident, HIGH IMPACT  (early, long, tight-TW pair)
  {name}_acc_B[_{idx}].txt   -- accident, NO IMPACT    (non-route edge OR post-traversal timing)

Rain B guarantee    : trigger_B + duration_B  <  min_arrival_in_nodes - MARGIN
Accident B guarantee: per-event, one of:
  - Location-based: accident on edge not in any baseline route (vehicle never passes)
  - Timing-based  : accident on A's edge, trigger fires after vehicle has already cleared it

Usage
-----
  python generate_events.py                         # c101-c109, 1 scenario each
  python generate_events.py c101 --count 5          # 5 random scenarios for c101
  python generate_events.py --count 10 --seed 42    # reproducible multi-scenario
  python generate_events.py --list                  # dry-run
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DEFAULT_DATA_DIR = os.path.join(_ROOT, "data", "Solomon")

DEFAULT_INSTANCES = [
    "c101", "c102", "c103", "c104", "c105",
    "c106", "c107", "c108", "c109",
]

DEFAULT_R1_INSTANCES = [
    "r101", "r102", "r103", "r104", "r105",
    "r106", "r107", "r108", "r109", "r110", "r111", "r112",
]

# Benchmark configs: test instance + train instances for M_true_hist baseline
BENCHMARK_CONFIGS = {
    "c1":  dict(test="c101",  train=[f"c10{i}" for i in range(2, 10)],
                scenarios=["c101_rain_A", "c101_rain_B", "c101_acc_A", "c101_acc_B"]),
    "rc1": dict(test="rc101", train=[f"rc10{i}" for i in range(2, 9)],
                scenarios=["rc101_rain_A", "rc101_rain_B", "rc101_acc_A", "rc101_acc_B"]),
    "r1":  dict(test="r101",  train=[f"r1{i:02d}" for i in range(2, 13)],
                scenarios=["r101_rain_A", "r101_rain_B", "r101_acc_A", "r101_acc_B"]),
    "c2":  dict(test="c201",  train=[f"c20{i}" for i in range(2, 9)],
                scenarios=["c201_rain_A", "c201_rain_B", "c201_acc_A", "c201_acc_B"]),
}

SECTOR_NAMES  = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
_MARGIN_MIN       = 15.0   # rain_B: rain clears this many minutes before first arrival
_MIN_DURATION     = 5      # floor for any event duration (minutes)
# Late-target as fraction of N so the criterion scales with instance size.
# rain_A: 5–30 % of customers late (wide zone, moderate-to-strong multiplier)
# acc_A : 2–12 % of customers late (narrow but severe single-edge disruption)
RAIN_LATE_MIN_PCT = 0.05
RAIN_LATE_MAX_PCT = 0.30
ACC_LATE_MIN_PCT  = 0.02
ACC_LATE_MAX_PCT  = 0.12
MAX_GEN_RETRIES = 20


# ── Raw loader ─────────────────────────────────────────────────────────────────

def load_raw(path: str) -> dict:
    """Parse Solomon file → raw numpy arrays + original text lines (no events)."""
    with open(path, "r", encoding="utf-8") as f:
        all_lines = f.read().splitlines()

    name = all_lines[0].strip()

    vi = next(i for i, l in enumerate(all_lines) if l.strip().upper().startswith("VEHICLE"))
    vehicle_limit = vehicle_capacity = None
    for off in (1, 2):
        parts = all_lines[vi + off].split()
        try:
            vehicle_limit = int(parts[0]); vehicle_capacity = float(parts[1]); break
        except (ValueError, IndexError):
            continue

    ci = next(i for i, l in enumerate(all_lines) if l.strip().upper().startswith("CUSTOMER"))
    stop = next(
        (i for i, l in enumerate(all_lines) if i > ci and l.strip().upper() == "EVENTS"),
        len(all_lines),
    )
    rows = []
    for ln in all_lines[ci + 1 : stop]:
        parts = ln.split()
        if len(parts) >= 7:
            try:
                rows.append([float(x) for x in parts[:7]])
            except ValueError:
                continue

    data    = np.array(rows, dtype=np.float64)
    T       = float(data[0, 5])
    coords  = data[:, 1:3]
    tw_open = data[:, 4]
    tw_close= data[:, 5]
    service = data[:, 6]

    diff = coords[:, None, :] - coords[None, :, :]
    tt   = np.sqrt((diff ** 2).sum(-1))

    base = list(all_lines[:stop])
    while base and not base[-1].strip():
        base.pop()

    return dict(
        name=name, n_customers=len(data) - 1,
        T=T, coords=coords, tw_open=tw_open, tw_close=tw_close,
        service=service, tt=tt, base_lines=base,
    )


# ── Late-stop simulation ───────────────────────────────────────────────────────

# Internal multiplier table keyed by 1-10 integer severity.
# This mapping is NEVER shown to the LLM or stored in event files.
_SEVERITY_MULT: dict[int, float] = {
    1:  1.5,  2:  2.0,  3:  2.8,  4:  3.5,  5:  5.0,
    6:  6.5,  7:  8.5,  8: 10.5,  9: 13.0, 10: 15.0,
}
# Legacy named-level fallback (for old event files)
_SEVERITY_MULT_NAMED = {"low": 2.0, "medium": 3.5, "high": 5.0}


def _build_event_tt(data: dict, events: list[dict]) -> np.ndarray:
    """Apply event multipliers to tt matrix (full multiplier, no time averaging)."""
    tt = data["tt"].copy()
    for ev in events:
        nodes = ev["nodes"]
        # Resolve multiplier from severity (int 1-10) or legacy formats
        sev = ev.get("severity")
        if isinstance(sev, int):
            mult = _SEVERITY_MULT.get(sev, 5.0)
        elif isinstance(sev, str):
            try:
                sev_int = int(sev)
                mult = _SEVERITY_MULT.get(sev_int, 5.0)
            except ValueError:
                mult = _SEVERITY_MULT_NAMED.get(sev, float(sev) if sev.replace('.','',1).isdigit() else 5.0)
        elif "multiplier" in ev:
            mult = float(ev["multiplier"])
        else:
            mult = 5.0
        if ev["type"] == "RAIN":
            for a in nodes:
                for b in nodes:
                    tt[a, b] = tt[a, b] * mult
                    tt[b, a] = tt[b, a] * mult
        else:  # ACCIDENT
            a, b = nodes[0], nodes[1]
            tt[a, b] = data["tt"][a, b] * mult
            tt[b, a] = data["tt"][b, a] * mult
    return tt


def _event_multiplier(ev: dict) -> float:
    """Resolve an event dict's effective multiplier regardless of storage format
    (plain float, RAIN severity int/str, or ACCIDENT n-car description)."""
    if "multiplier" in ev and isinstance(ev["multiplier"], (int, float)):
        return float(ev["multiplier"])
    sev = ev.get("severity")
    if isinstance(sev, int):
        return _SEVERITY_MULT.get(sev, 5.0)
    if isinstance(sev, str):
        try:
            return _SEVERITY_MULT.get(int(sev), 5.0)
        except ValueError:
            return _SEVERITY_MULT_NAMED.get(sev, 5.0)
    return 5.0


def _arc_in_zone(i: int, j: int, ev: dict, sec: np.ndarray | None) -> bool:
    """Part B-1: either endpoint in the event's affected-node set counts as
    'in zone' (an arc entering OR leaving a disrupted area is slowed -- the old
    both-endpoints-inside rule left entering/leaving traffic at full speed,
    which understates real impact and is not physically motivated)."""
    nodes = ev.get("nodes")
    if nodes is not None:
        node_set = ev.get("_node_set")
        if node_set is None:
            node_set = ev["_node_set"] = set(nodes)
        return i in node_set or j in node_set
    return False   # legacy events without a resolved node list


def travel_time(
    data: dict, i: int, j: int, t_depart: float, events: list[dict],
    mode: str = "live",
) -> float:
    """Part B-9: FIFO-consistent travel time via piecewise-constant speed
    integration, replacing the old 'binary multiplier over the whole arc,
    decided once at departure' rule (which lets a later departure arrive
    earlier than an earlier one whenever the arc straddles an event boundary).

    mode="live"      : events that start after t_depart but before arrival
                        still apply once reached (standard TDVRPTW; Ichoua et
                        al. 2003; Figliozzi 2012; Dabia et al. 2013).
    mode="committed" : only events already active at t_depart apply for the
                        whole hop ("once you commit to an arc you ride it out").
    """
    remaining = float(data["tt"][i, j])
    if remaining <= 1e-9 or not events:
        return remaining

    t = t_depart
    if mode == "committed":
        active = [e for e in events
                  if float(e.get("trigger_time", 0)) <= t_depart
                  < float(e.get("trigger_time", 0)) + float(e.get("duration", 0))]
    boundaries = sorted({float(e.get("trigger_time", 0)) for e in events} |
                        {float(e.get("trigger_time", 0)) + float(e.get("duration", 0))
                         for e in events})

    elapsed = 0.0
    while remaining > 1e-9:
        pool = active if mode == "committed" else events
        mult = 1.0
        for ev in pool:
            trig = float(ev.get("trigger_time", 0))
            end  = trig + float(ev.get("duration", 0))
            if trig <= t < end and _arc_in_zone(i, j, ev, None):
                mult = max(mult, _event_multiplier(ev))
        speed  = 1.0 / mult
        future = [b for b in boundaries if b > t]
        t_next = future[0] if future else float("inf")
        dt     = t_next - t
        can    = speed * dt if dt != float("inf") else float("inf")
        if can >= remaining:
            elapsed += remaining / speed
            return elapsed
        elapsed   += dt
        remaining -= can
        t = t_next
    return elapsed


def simulate_late(data: dict, routes: list[list[int]], events: list[dict],
                  mode: str = "live") -> int:
    """Simulate base routes with FIFO-consistent event travel times (Part B-9);
    return number of late arrivals."""
    tw_open  = data["tw_open"]
    tw_close = data["tw_close"]
    service  = data["service"]
    late     = 0

    for route in routes:
        cur, cur_time = 0, 0.0
        for cust in route:
            travel    = travel_time(data, cur, cust, cur_time, events, mode=mode)
            arr       = cur_time + travel
            svc_start = max(arr, tw_open[cust])
            if arr > tw_close[cust] + 0.5:
                late += 1
            cur_time  = svc_start + service[cust]
            cur       = cust
    return late


# ── Sector analysis ────────────────────────────────────────────────────────────

def sector_info(data: dict, n_sectors: int = 8) -> list[dict]:
    """Split customers into angular sectors; compute slack and first_arrival."""
    depot   = data["coords"][0]
    coords  = data["coords"][1:]
    tt0     = data["tt"][0, 1:]
    tw_open = data["tw_open"][1:]
    tw_close= data["tw_close"][1:]
    service = data["service"][1:]

    dx = coords[:, 0] - depot[0]
    dy = coords[:, 1] - depot[1]
    angles = np.degrees(np.arctan2(dy, dx)) % 360

    step    = 360.0 / n_sectors
    sectors = []
    for s in range(n_sectors):
        mask = (angles >= s * step) & (angles < (s + 1) * step)
        if not mask.any():
            continue
        idxs  = np.where(mask)[0]
        nodes = (idxs + 1).tolist()   # 1-based customer ids
        earliest_start = np.maximum(tt0[idxs], tw_open[idxs])
        slack = tw_close[idxs] - earliest_start - service[idxs]
        sectors.append(dict(
            sector_id   = s,
            name        = SECTOR_NAMES[s % 8],
            nodes       = nodes,
            first_arrival = float(tt0[idxs].min()),
            slack_mean  = float(slack.mean()),
            slack       = slack.tolist(),      # per-node
            tw_width    = (tw_close[idxs] - tw_open[idxs]).tolist(),
        ))

    sectors.sort(key=lambda s: s["slack_mean"])  # tightest first
    return sectors


# ═════════════════════════════════════════════════════════════════════════════
# Part B — sector-based, policy-independent event generation
#
# Replaces gen_rain_pair/gen_acc_pair (both baseline-route-conditioned) with an
# extraction that only ever looks at instance data (coords, tw_open, tw_close,
# service) -- never at any solver's routes. Targeting comes from a hard VRPTW
# guarantee instead: a customer whose time window is fully contained in the
# event's active interval MUST be visited during that interval by ANY feasible
# policy (every customer is visited exactly once, inside its own window).
# ═════════════════════════════════════════════════════════════════════════════

def sector_of(data: dict, n_sectors: int = 8) -> np.ndarray:
    """Return a 0-based int array of length N: sec[i] = sector index of customer i+1.

    Sector 0 is centered on North (matches SECTOR_NAMES ordering) and sectors
    are contiguous equal-angle wedges around the depot -- an instance-fixed
    property, independent of any routing decision.
    """
    depot  = data["coords"][0]
    coords = data["coords"][1:]
    dx = coords[:, 0] - depot[0]
    dy = coords[:, 1] - depot[1]
    # atan2 gives 0=East, CCW positive; rotate by +pi/2 so 0=North, then wrap.
    theta = np.arctan2(dy, dx)
    step  = 2 * np.pi / n_sectors
    idx   = np.floor(((theta + np.pi / 2 + step / 2) % (2 * np.pi)) / step)
    return idx.astype(int) % n_sectors


def avg_gap(data: dict) -> float:
    """Mean inter-customer travel time + mean service time (Part B-7).

    Used as the natural time unit for event duration: a duration expressed as
    a multiple of avg_gap means roughly the same thing (in "how many stops does
    this span") whether the instance's horizon T is 230 (R1) or 1236 (C1).
    """
    tt_cust = data["tt"][1:, 1:]
    n = tt_cust.shape[0]
    off_diag_mean = (tt_cust.sum() - np.trace(tt_cust)) / (n * (n - 1))
    return float(off_diag_mean + data["service"][1:].mean())


# Literature-based default severity levels (Part B-3a / Part E). Each level is a
# dict {name, mu, text, weight, enabled}; draws are weighted by `weight` among
# `enabled` levels. Fully overridable via GenConfig(rain_severity=[...], ...).
DEFAULT_RAIN_SEVERITY = [
    dict(name="light",    mu=1.05, text="약한 비",     weight=1, enabled=True),
    dict(name="moderate", mu=1.11, text="비",          weight=1, enabled=True),
    dict(name="heavy",    mu=1.19, text="호우주의보",  weight=1, enabled=True),
]
DEFAULT_ACC_SEVERITY = [
    dict(name="minor",    mu=1.20, text="갓길 접촉사고",          weight=1, enabled=True),
    dict(name="moderate", mu=2.04, text="1개 차로 통제",          weight=1, enabled=True),
    dict(name="major",    mu=5.88, text="다중 추돌, 2개 차로 통제", weight=1, enabled=True),
]


def _draw_severity(levels: list[dict], rng: np.random.Generator) -> dict:
    active = [lv for lv in levels if lv.get("enabled", True) and lv.get("weight", 1) > 0]
    if not active:
        raise ValueError("No enabled severity levels with weight > 0")
    w = np.array([lv["weight"] for lv in active], dtype=float)
    return active[int(rng.choice(len(active), p=w / w.sum()))]


# Default parameters (Part E). Every field is overridable via GenConfig(...).
class GenConfig:
    def __init__(self, **overrides):
        self.n_sectors_total = 8
        self.rain_n_sectors  = (2, 3)
        self.rain_severity   = DEFAULT_RAIN_SEVERITY
        self.rain_delta_k    = (1.0, 2.0)     # x avg_gap
        self.acc_n_sectors   = (1, 1)
        self.acc_severity    = DEFAULT_ACC_SEVERITY
        self.acc_delta_k     = (0.5, 1.5)     # x avg_gap
        self.acc_n_events    = (2, 3)         # accidents per wave
        self.acc_wave_frac   = 0.5            # x avg_gap
        self.delta_min       = 5.0
        self.delta_max_frac  = 0.30           # x T
        self.max_retries     = 50             # only to avoid degenerate empty sectors
        self.force_pre_clear = False
        for k, v in overrides.items():
            if not hasattr(self, k):
                raise ValueError(f'Unknown GenConfig field: {k}')
            setattr(self, k, v)


def _label_event(data: dict, sec: np.ndarray, sector_set: set, sigma: float,
                  delta: float) -> dict:
    """Post-hoc label + descriptive stats (Part B-4, v2).

    Label is purely a TIME-overlap relationship between the event window and
    the affected zone's delivery activity -- it does NOT depend on whether any
    customer happens to be fully covered. n_covered/n_touched are recorded as
    separate descriptive statistics (used for filtering, not for labeling --
    using them as the label would be circular: they are an outcome of the
    event x instance relationship, not a property of the event itself).
    """
    tw_open, tw_close = data["tw_open"], data["tw_close"]
    tt0 = data["tt"][0]
    N = data["n_customers"]

    inside = [i for i in range(1, N + 1) if int(sec[i - 1]) in sector_set]
    if not inside:
        return dict(label="empty", bound_set=[], n_covered=0, n_touched=0)

    bound = {i for i in inside if tw_open[i] >= sigma and tw_close[i] <= sigma + delta}
    touched = {i for i in inside
              if not (tw_close[i] < sigma or tw_open[i] > sigma + delta)}  # interval overlap

    first_arrival = min(max(tt0[i], tw_open[i]) for i in inside)
    if sigma + delta < first_arrival:
        label = "pre_clear"        # ends before any policy could possibly arrive
    elif sigma > max(tw_close[i] for i in inside):
        label = "post_service"     # starts after every customer's deadline has passed
    else:
        label = "overlap"

    return dict(label=label, bound_set=sorted(bound),
                n_covered=len(bound), n_touched=len(touched))


def generate_sector_event(
    data: dict, kind: str, cfg: GenConfig, rng: np.random.Generator,
    sec: np.ndarray | None = None,
) -> dict | None:
    """Independent extraction (Part B-4, v2): draw (sectors, timing, severity)
    from their own distributions, then attach a post-hoc label + descriptive
    stats. Never looks at any solver's routes.

    Generation is UNCONDITIONAL -- it always returns a labeled event (retrying
    only cfg.max_retries times to skip a degenerate empty-sector draw, e.g. an
    ACCIDENT sector with zero customers). Whether the event is *useful* for a
    given experiment stage is a SEPARATE decision made by select_stage() below,
    so one generation pass can serve every stage just by re-filtering.

    kind: "RAIN" or "ACCIDENT".
    """
    if sec is None:
        sec = sector_of(data, cfg.n_sectors_total)
    T = data["T"]

    is_rain    = (kind.upper() == "RAIN")
    n_sec_rng  = cfg.rain_n_sectors if is_rain else cfg.acc_n_sectors
    severity_lv= cfg.rain_severity  if is_rain else cfg.acc_severity
    k_rng      = cfg.rain_delta_k   if is_rain else cfg.acc_delta_k
    gap        = avg_gap(data)

    for _attempt in range(cfg.max_retries):
        n_sec = int(rng.integers(n_sec_rng[0], n_sec_rng[1] + 1))
        s0    = int(rng.integers(0, cfg.n_sectors_total))
        sector_set = {(s0 + j) % cfg.n_sectors_total for j in range(n_sec)}

        inside = [i for i in range(1, data["n_customers"] + 1)
                  if int(sec[i - 1]) in sector_set]
        if not inside:
            continue

        k     = float(rng.uniform(*k_rng))
        delta = k * gap
        delta = min(max(delta, cfg.delta_min), cfg.delta_max_frac * T)

        if cfg.force_pre_clear:
            first_arrival = min(max(data["tt"][0, i], data["tw_open"][i]) for i in inside)
            delta  = min(delta, max(first_arrival - 10.0, cfg.delta_min))
            sigma  = 0.0
        else:
            sigma = float(rng.uniform(0, max(T - delta, 0.0)))

        stats = _label_event(data, sec, sector_set, sigma, delta)
        if cfg.force_pre_clear and stats["label"] != "pre_clear":
            continue

        sev = _draw_severity(severity_lv, rng)
        mu  = float(sev["mu"])
        nodes = sorted(inside)

        event = dict(
            type          = "RAIN" if is_rain else "ACCIDENT",
            sectors       = sorted(sector_set),
            sector_names  = [SECTOR_NAMES[s] for s in sorted(sector_set)],
            trigger_time  = round(sigma, 1),
            duration      = round(delta, 1),
            multiplier    = mu,
            severity_name = sev["name"],
            severity_text = sev["text"],
            nodes         = nodes,
            reveal_time   = 0.0 if is_rain else round(sigma, 1),
            **stats,   # label, bound_set, n_covered, n_touched
        )
        event["delay_mass"] = delay_mass(data, event, sec=sec)
        if not is_rain:
            event["description"] = sev["text"]
        else:
            event["rainfall_mm"] = round(mu * 10.0 + float(rng.uniform(-5, 10)), 0)
        return event

    return None


def select_stage(events: list[dict], stage: int) -> list[dict]:
    """Part B-5: filter ALREADY-GENERATED events by experiment stage. Kept as a
    separate step from generation so one generation pass can serve any stage.

    stage 1: overlap events with at least one fully-covered customer (the
             policy-independent hit guarantee actually applies).
    stage 2: keep the natural label mix (drop only truly empty-sector draws).
    """
    if stage == 1:
        return [e for e in events if e.get("label") == "overlap" and e.get("n_covered", 0) >= 1]
    if stage == 2:
        return [e for e in events if e.get("label") != "empty"]
    raise ValueError(f"Unknown stage: {stage}")


def draw_until_stage(
    data: dict, kind: str, cfg: GenConfig, rng: np.random.Generator, stage: int,
    sec: np.ndarray | None = None, max_tries: int = 200,
) -> dict | None:
    """Convenience: repeatedly call generate_sector_event + select_stage until
    one draw passes, without baking the stage filter into generation itself."""
    for _ in range(max_tries):
        ev = generate_sector_event(data, kind, cfg, rng, sec=sec)
        if ev is not None and select_stage([ev], stage):
            return ev
    return None


def generate_accident_wave(
    data: dict, cfg: GenConfig, rng: np.random.Generator, sec: np.ndarray | None = None,
) -> list[dict] | None:
    """Part B-8: 2-3 accidents clustered within a short window w = wave_frac * avg_gap.

    Because a vehicle only reacts at its next departure, accidents within one
    inter-stop gap are indistinguishable in effect regardless of their exact
    offsets -- w is estimated from instance statistics only (never a solved
    route), so this does not reintroduce baseline conditioning.
    """
    if sec is None:
        sec = sector_of(data, cfg.n_sectors_total)
    n_events = int(rng.integers(cfg.acc_n_events[0], cfg.acc_n_events[1] + 1))
    w = cfg.acc_wave_frac * avg_gap(data)

    base = draw_until_stage(data, "ACCIDENT", cfg, rng, stage=1, sec=sec)
    if base is None:
        return None
    wave = [base]
    base_t = base["trigger_time"]
    for _ in range(n_events - 1):
        ev = generate_sector_event(data, "ACCIDENT", cfg, rng, sec=sec)
        if ev is None:
            continue
        ev["trigger_time"] = round(base_t + float(rng.uniform(0, w)), 1)
        # Re-label at the wave-adjusted trigger time -- the label/stats computed
        # inside generate_sector_event used the ORIGINAL (pre-override) sigma.
        ev.update(_label_event(data, sec, set(ev["sectors"]),
                               ev["trigger_time"], ev["duration"]))
        wave.append(ev)
    return wave


def delay_mass(data: dict, event: dict, sec: np.ndarray | None = None) -> float:
    """Part D-3 #4: sum over affected arcs of base_travel_time * (mu - 1).

    Used to calibrate RAIN (wide+weak) against ACCIDENT (narrow+strong) so their
    total injected delay is comparable and any performance gap between the two
    scenario types can be attributed to rho_e (information timing) rather than
    to one simply being a "bigger" disruption than the other.
    """
    if sec is None:
        sec = sector_of(data, 8)
    sector_set = set(event["sectors"])
    tt  = data["tt"]
    N   = data["n_customers"]
    mu  = float(event["multiplier"])
    mass = 0.0
    for i in range(N + 1):
        for j in range(N + 1):
            if i == j:
                continue
            i_in = (i != 0 and int(sec[i - 1]) in sector_set)
            j_in = (j != 0 and int(sec[j - 1]) in sector_set)
            if i_in or j_in:
                mass += float(tt[i, j]) * (mu - 1.0)
    return mass


def random_invalidation_rate(
    data: dict, baseline_routes: list[list[int]], n_trials: int = 100,
    seed: int = 0,
) -> float:
    """Part D-3 #1: fraction of uniformly-random node-pair 'accidents' that have
    ZERO effect on a fixed baseline solution -- quantifies why untargeted random
    placement (gen_random_acc) is not a usable diagnostic/control by itself."""
    rng = np.random.default_rng(seed)
    N = data["n_customers"]
    Lc0 = simulate_late(data, baseline_routes, [])
    invalid = 0
    for _ in range(n_trials):
        a, b = rng.choice(range(1, N + 1), size=2, replace=False)
        ev = dict(type="ACCIDENT", trigger_time=0, duration=data["T"] * 0.5,
                  multiplier=8.0, nodes=[int(a), int(b)])
        if simulate_late(data, baseline_routes, [ev]) <= Lc0:
            invalid += 1
    return invalid / n_trials


# ── Legacy: baseline-conditioned generators (superseded by generate_sector_event
#    above; kept only for reference / A-B regression comparison, not called by
#    the default pipeline any more) ─────────────────────────────────────────────

# ── Random event pair generators ───────────────────────────────────────────────

def _sector_weights(sectors: list[dict]) -> np.ndarray:
    """Selection weights: tighter slack → higher probability."""
    slack_vals = np.array([s["slack_mean"] for s in sectors], dtype=float)
    shifted    = slack_vals - slack_vals.min() + 1.0
    w          = 1.0 / shifted
    return w / w.sum()


def _intra_sector_route_edges(sectors: list[dict], baseline_routes: list[list[int]]) -> np.ndarray:
    """
    For each sector, count route edges where BOTH endpoints are inside the sector.
    Rain only slows an edge when both src and dst are in the rain zone, so this is
    the true measure of rain impact potential per sector.
    Returns int array, shape (n_sectors,).
    """
    counts = []
    for s in sectors:
        zone = set(s["nodes"])
        n = sum(
            1
            for route in baseline_routes
            for k in range(len(route) - 1)
            if route[k] in zone and route[k + 1] in zone
        )
        counts.append(n)
    return np.array(counts, dtype=float)


# ── Random event generators (no route targeting) ──────────────────────────────

def gen_random_rain(
    data: dict, rng: np.random.Generator, n_events: int | None = None,
) -> list[dict]:
    """Random RAIN events without route-specific targeting.

    Zone selection: k-nearest cluster around a randomly chosen center node.
    Timing: uniformly drawn from [0, T*0.7] to allow meaningful routing response.
    Duration: [T*0.15, T*0.5] — substantial but not whole-horizon.

    Ref: Pillac et al. (2013) 'A review of dynamic vehicle routing problems',
         European Journal of Operational Research 225(1), pp. 1-11.
    """
    T      = data["T"]
    N      = data["n_customers"]
    coords = data["coords"][1:]   # (N, 2) — customers only

    if n_events is None:
        n_events = int(rng.integers(1, 4))   # 1, 2, or 3

    events: list[dict] = []
    for _ in range(n_events):
        # k-nearest cluster around a random center customer
        k = int(rng.integers(max(2, int(N * 0.05)), max(3, int(N * 0.20)) + 1))
        center_idx = int(rng.integers(0, N))
        dists      = np.sqrt(((coords - coords[center_idx]) ** 2).sum(-1))
        nodes      = sorted((np.argsort(dists)[:k] + 1).tolist())   # 1-based

        trigger    = int(rng.uniform(0, T * 0.7))
        duration   = int(rng.uniform(T * 0.15, T * 0.50))
        multiplier = round(float(rng.uniform(2.0, 5.0)), 1)
        rainfall   = round(float(multiplier * 10.0 + rng.uniform(-5, 10)), 0)

        events.append(dict(
            type="RAIN", trigger_time=trigger, duration=duration,
            multiplier=multiplier, rainfall_mm=rainfall, nodes=nodes,
        ))
    return events


def gen_random_acc(
    data: dict, rng: np.random.Generator, n_events: int | None = None,
) -> list[dict]:
    """Random ACCIDENT events without route-specific targeting.

    Count is intentionally small (1-2): at test time the LLM is invoked once per
    accident event, so a high count makes inference impractically slow.
    Duration is long (T*0.3–0.6) to reflect real-world accident clearance times.

    Ref: Pillac et al. (2013); Gendreau & Potvin (1998) 'Dynamic vehicle routing
         and dispatching', in: Fleet Management and Logistics, Kluwer.
    """
    T = data["T"]
    N = data["n_customers"]

    if n_events is None:
        n_events = int(rng.integers(1, 3))   # 1 or 2 — kept small for LLM budget

    _NCAR_DESC = {3: "3-car-collision", 4: "4-car-pile-up", 5: "5-car-pile-up"}
    _NCAR_MULT = {3: 5.0, 4: 8.5, 5: 13.0}

    events:     list[dict]           = []
    used_pairs: set[frozenset]       = set()
    all_nodes   = list(range(1, N + 1))

    for _ in range(n_events):
        for _ in range(200):
            a = int(rng.choice(all_nodes))
            b = int(rng.choice(all_nodes))
            if a != b and frozenset({a, b}) not in used_pairs:
                used_pairs.add(frozenset({a, b}))
                break

        n_cars   = int(rng.integers(3, 6))
        trigger  = int(rng.uniform(0, T * 0.7))
        duration = int(rng.uniform(T * 0.3, T * 0.6))

        events.append(dict(
            type="ACCIDENT", trigger_time=trigger, duration=duration,
            description=_NCAR_DESC[n_cars], multiplier=_NCAR_MULT[n_cars],
            nodes=[a, b],
        ))
    return events


def generate_random_for_instance(
    inst_name: str,
    data_dir:  str,
    out_dir:   str,
    count:     int,
    seed:      int | None,
    dry_run:   bool,
) -> None:
    """Generate random (non-route-targeted) rain and accident scenarios.

    For each scenario index produces two files:
      {inst_name}_rand_rain[_{idx}].txt  — 1-3 random rain events
      {inst_name}_rand_acc[_{idx}].txt   — 1-2 random accident events
    """
    src_path = os.path.join(data_dir, inst_name + ".txt")
    if not os.path.isfile(src_path):
        print(f"  [SKIP] {src_path} not found")
        return

    data = load_raw(src_path)
    T    = data["T"]
    N    = data["n_customers"]
    print(f"\n[{inst_name.upper()}]  T={T:.0f}  N={N}  random scenarios={count}")

    for i in range(count):
        sc_seed = (seed + i) if seed is not None else None
        rng     = np.random.default_rng(sc_seed)
        suffix  = f"_{i+1:03d}" if count > 1 else ""

        rain_events = gen_random_rain(data, rng)
        acc_events  = gen_random_acc(data, rng)

        for sc_name, events in [
            (f"{inst_name}_rand_rain{suffix}", rain_events),
            (f"{inst_name}_rand_acc{suffix}",  acc_events),
        ]:
            ev_str = "  |  ".join(
                f"{e['type']} t={e['trigger_time']} d={e['duration']} "
                f"mult={e.get('multiplier', '?')} nodes={e['nodes']}"
                for e in events
            )
            print(f"  {'[DRY] ' if dry_run else ''}-> {sc_name}.txt")
            print(f"       {ev_str}")
            if not dry_run:
                out_path = os.path.join(out_dir, sc_name + ".txt")
                write_scenario(data, events, sc_name, out_path)


def gen_rain_pair(
    data: dict, sectors: list[dict], rng: np.random.Generator,
    baseline_routes: list[list[int]] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (events_A, events_B).

    A: sector with the MOST intra-zone route edges  (maximum impact area)
       + timing active during the routing window.
    B: sector with the FEWEST intra-zone route edges (peripheral area)
       + timing guarantee: rain ends before any vehicle reaches that zone.
       Combined: location is already low-traffic AND timing adds a hard guarantee.

    Both A and B share the same multiplier/intensity.
    """
    T    = data["T"]
    tt0  = data["tt"][0, 1:]
    routes   = baseline_routes or []
    slack_w  = _sector_weights(sectors)
    edge_cnt = _intra_sector_route_edges(sectors, routes)

    n_sects = int(rng.integers(2, 4))   # min 2 sectors for meaningful area coverage
    n_sects = min(n_sects, len(sectors))

    # A: sectors with high intra-zone route edge density, weighted by TW tightness
    w_A = (edge_cnt + 0.05) * slack_w
    w_A = w_A / w_A.sum()

    # B: sectors with low intra-zone route edge density (peripherally routed areas)
    inv_edge = 1.0 / (edge_cnt + 0.5)
    w_B = inv_edge / inv_edge.sum()

    idx_A = rng.choice(len(sectors), size=n_sects, replace=False, p=w_A)
    idx_B = rng.choice(len(sectors), size=n_sects, replace=False, p=w_B)

    events_A, events_B = [], []
    for i in range(n_sects):
        sec_A = sectors[int(idx_A[i])]
        sec_B = sectors[int(idx_B[i])]

        # Shared intensity per pair — rain wide area, moderate-to-strong multiplier
        multiplier = round(float(rng.uniform(2.5, 4.0)), 1)
        rainfall   = round(float(multiplier * 10.0 + rng.uniform(-5, 10)), 0)

        # A timing: active throughout the main routing window
        trigger_A  = int(rng.uniform(0, T * 0.2))
        duration_A = int(rng.uniform(T * 0.5, T - trigger_A))
        events_A.append(dict(
            type="RAIN", trigger_time=trigger_A, duration=duration_A,
            multiplier=multiplier, rainfall_mm=rainfall,
            nodes=sorted(sec_A["nodes"]),
        ))

        # B timing: trigger=0 so rain starts at episode start and ends before any
        # vehicle reaches B's zone.  Hard guarantee: trigger_B + duration_B < first_arrival.
        node_idxs_B   = [n - 1 for n in sec_B["nodes"]]
        first_arr_B   = float(tt0[node_idxs_B].min())
        max_dur_B     = max(first_arr_B - _MARGIN_MIN, _MIN_DURATION)
        trigger_B     = 0
        duration_B    = max(int(rng.uniform(max_dur_B * 0.3, max_dur_B)), _MIN_DURATION)
        events_B.append(dict(
            type="RAIN", trigger_time=trigger_B, duration=duration_B,
            multiplier=multiplier, rainfall_mm=rainfall,
            nodes=sorted(sec_B["nodes"]),
        ))

    return events_A, events_B


def _lkh_fresh_routes(src_path: str, time_limit: int = 30) -> list[list[int]] | None:
    """Solve the base instance with LKH-3 and return its routes.

    Uses the exact same inputs as the "no-info" branch of `_print_lkh_table`
    (tt_base, time_limit=30s, runs=1, SEED=42 fixed in `_write_par_file`), so
    the routes returned here are deterministically IDENTICAL to the routes the
    no-info evaluation will execute. Placing/validating A-type events against
    these routes guarantees they actually intersect the no-info execution —
    otherwise an A-type event silently degenerates into a zero-impact B-type
    event (the route never traverses the affected edge/zone).

    Returns None if LKH-3/WSL is unavailable or the solve fails.
    """
    _cr_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "clustering_routing")
    if _cr_dir not in sys.path:
        sys.path.insert(0, _cr_dir)
    try:
        from R_lkh import load_solomon, build_tt_matrix, solve_vrptw_lkh
    except ImportError as exc:
        print(f"  [WARN] R_lkh unavailable ({exc}) for baseline routing")
        return None

    try:
        inst    = load_solomon(src_path)
        tt_base = build_tt_matrix(inst)
        result  = solve_vrptw_lkh(inst, tt_base, time_limit_sec=time_limit, runs=1)
    except Exception as exc:
        print(f"  [WARN] LKH-3 baseline solve failed: {exc}")
        return None

    routes = result.get("routes")
    return routes if routes else None


def _ortools_routes(src_path: str, time_limit: int = 15) -> list[list[int]] | None:
    """Run OR-Tools on the base instance to get actual optimal routes.
    Returns None if OR-Tools is unavailable or no solution is found.
    Falls back gracefully so greedy is used instead.
    """
    try:
        from ortools_vrptw import load_solomon_raw, build_tt, solve
        inst   = load_solomon_raw(src_path)
        tt     = build_tt(inst)
        result = solve(inst, tt, time_limit)
        routes = [r for r in result.get("routes", []) if r]
        return routes if routes else None
    except Exception as exc:
        print(f"  [WARN] OR-Tools unavailable, falling back to greedy: {exc}")
        return None


def _greedy_routes(data: dict) -> list[list[int]]:
    """Greedy nearest-neighbour routes respecting TW and capacity.
    Returns list of routes (each route = list of 1-based customer ids).
    Fallback when OR-Tools is unavailable.
    """
    n       = data["n_customers"]
    tt      = data["tt"]
    tw_open = data["tw_open"]
    tw_close= data["tw_close"]
    service = data["service"]
    Q       = data.get("vehicle_capacity", float("inf"))
    demands = np.zeros(n + 1)  # placeholder if not in raw data

    unvisited = set(range(1, n + 1))
    routes    = []

    while unvisited:
        route     = []
        cur       = 0
        cur_time  = 0.0
        load      = 0.0
        while True:
            best, best_dist = None, float("inf")
            for c in unvisited:
                arr = cur_time + tt[cur, c]
                if arr > tw_close[c] + 1e-6:
                    continue
                if load + demands[c] > Q + 1e-6:
                    continue
                if tt[cur, c] < best_dist:
                    best_dist, best = tt[cur, c], c
            if best is None:
                break
            route.append(best)
            unvisited.remove(best)
            arr      = cur_time + tt[cur, best]
            cur_time = max(arr, tw_open[best]) + service[best]
            load    += demands[best]
            cur      = best
        if route:
            routes.append(route)

    return routes


def _route_edges(routes: list[list[int]]) -> list[tuple[int, int]]:
    """Extract all consecutive (a, b) pairs from routes (depot=0 excluded)."""
    edges = []
    for route in routes:
        for k in range(len(route) - 1):
            edges.append((route[k], route[k + 1]))
    return edges


def _non_route_edges(
    data: dict,
    baseline_routes: list[list[int]],
    n_needed: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Pick n_needed customer pairs that are NOT consecutive edges in baseline routes."""
    route_edge_set: set[frozenset] = set()
    for route in baseline_routes:
        for k in range(len(route) - 1):
            route_edge_set.add(frozenset({route[k], route[k + 1]}))

    all_nodes = list(range(1, data["n_customers"] + 1))
    used: set[frozenset] = set()
    edges: list[tuple[int, int]] = []
    for _ in range(3000):
        a, b = (int(x) for x in rng.choice(all_nodes, size=2, replace=False))
        fp = frozenset({a, b})
        if fp not in route_edge_set and fp not in used:
            edges.append((a, b))
            used.add(fp)
        if len(edges) >= n_needed:
            break
    return edges


def _edge_traversal_times(
    data: dict, routes: list[list[int]]
) -> dict[frozenset, float]:
    """Compute the earliest vehicle departure time for each route edge.

    Returns frozenset{a, b} -> departure time from the source node.
    Used to generate timing-based B events: accident fires after vehicle already passed.
    """
    tt      = data["tt"]
    tw_open = data["tw_open"]
    service = data["service"]

    edge_depart: dict[frozenset, float] = {}
    for route in routes:
        cur, cur_time = 0, 0.0
        for cust in route:
            fp = frozenset({cur, cust})
            if fp not in edge_depart or cur_time < edge_depart[fp]:
                edge_depart[fp] = cur_time
            arr      = cur_time + tt[cur, cust]
            cur_time = max(arr, tw_open[cust]) + service[cust]
            cur      = cust
    return edge_depart


def gen_acc_pair(
    data: dict, sectors: list[dict], rng: np.random.Generator,
    baseline_routes: list[list[int]] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (events_A, events_B).

    A: 2-5 accidents on edges FROM the tightest OR-Tools route — guaranteed to
       be traversed, delays compound along that single route.
    B: zero-impact accidents via one of two mechanisms (chosen randomly per event):
       - Location-based: accident on an edge not in any baseline route.
       - Timing-based  : accident on A's same edge, but trigger fires AFTER the
                         vehicle has already traversed it.
       Both guarantee zero disruption; combining them avoids a purely artificial
       "only pick non-route edges" design.
    """
    T       = data["T"]
    tw_open = data["tw_open"][1:]
    tw_close= data["tw_close"][1:]
    service = data["service"][1:]
    tt0     = data["tt"][0, 1:]

    routes = baseline_routes if baseline_routes else _greedy_routes(data)
    if not routes:
        return [], []

    slack = tw_close - np.maximum(tt0, tw_open) - service  # (N,) 0-based
    routes_sorted = sorted(routes, key=lambda r: float(np.mean([slack[n - 1] for n in r])))

    edge_depart = _edge_traversal_times(data, routes)

    # Number of simultaneous accidents per wave (all share one trigger time).
    # Using a single trigger time reduces LLM refresh calls from N → 1 per episode.
    n_events = int(rng.integers(2, 6))   # 2-5 accidents per wave
    events_A, events_B = [], []
    used_pairs: set[frozenset] = set()

    for target_route in routes_sorted:
        if len(target_route) < 2:
            continue

        # Score edges by downstream-node slack (tightest first)
        scored_edges = sorted(
            [(float(slack[b - 1]), k, a, b)
             for k, (a, b) in enumerate(
                 (target_route[k], target_route[k + 1])
                 for k in range(len(target_route) - 1)
             )],
        )

        # Pick up to n_events distinct edges from this route
        candidates_A = []
        for b_slack, k, a, b in scored_edges:
            fp = frozenset({a, b})
            if fp not in used_pairs:
                candidates_A.append((a, b))
                used_pairs.add(fp)
            if len(candidates_A) >= n_events:
                break

        if len(candidates_A) < 2:
            continue

        # B: location-based only (non-route edges, same timing as A).
        # This guarantees zero impact regardless of timing, making A vs B
        # a clean location comparison (high-traffic vs off-route).
        candidates_B_loc = _non_route_edges(data, routes, len(candidates_A), rng)
        if len(candidates_B_loc) < len(candidates_A):
            continue

        # ── Base trigger time for this accident wave ─────────────────────────
        # Earliest departure across all candidate edges → trigger just before.
        min_dep = min(
            edge_depart.get(frozenset({a, b}), 0.0) for a, b in candidates_A
        )
        duration = int(rng.uniform(T * 0.3, T * 0.6))
        base_trigger = max(0, int(rng.uniform(max(0.0, min_dep - duration + 1), min_dep + 1)))

        # ── N-car description (LLM-visible; RL sees only binary presence) ───
        # 3-car-collision → 5.0x,  4-car-pile-up → 8.5x,  5-car-pile-up → 13.0x
        _NCAR_DESC = {3: "3-car-collision", 4: "4-car-pile-up", 5: "5-car-pile-up"}
        _NCAR_MULT = {3: 5.0, 4: 8.5, 5: 13.0}

        for i, (a_A, b_A) in enumerate(candidates_A):
            n_cars = int(rng.integers(3, 6))   # 3, 4, or 5 cars
            desc   = _NCAR_DESC[n_cars]
            mult   = _NCAR_MULT[n_cars]
            # Per-accident trigger: spread within 20-min window to cluster but not identical
            trigger = base_trigger + int(rng.integers(0, 20))
            events_A.append(dict(
                type="ACCIDENT", trigger_time=trigger, duration=duration,
                description=desc, multiplier=mult, nodes=[a_A, b_A],
            ))
            a_B, b_B = candidates_B_loc[i]
            n_cars_b = int(rng.integers(3, 6))
            desc_b   = _NCAR_DESC[n_cars_b]
            mult_b   = _NCAR_MULT[n_cars_b]
            trigger_b = base_trigger + int(rng.integers(0, 20))
            events_B.append(dict(
                type="ACCIDENT", trigger_time=trigger_b, duration=duration,
                description=desc_b, multiplier=mult_b, nodes=[a_B, b_B],
            ))
        break   # concentrated on one route — done

    return events_A, events_B


# ── OR-Tools evaluation ────────────────────────────────────────────────────────

def _evaluate_scenario(out_path: str, time_limit: int = 15) -> dict | None:
    """Evaluate a generated scenario with OR-Tools.

    Returns dict with:
      r1 : base-routes plan hit by events mid-route (no prior info)
      r2 : event-aware optimal plan (perfect info, None if no events)
      N  : n_customers
    Returns None if OR-Tools is unavailable.
    """
    try:
        from ortools_vrptw import (
            load_solomon_raw, build_tt, build_tt_rain,
            build_tt_accident, solve, resimulate_with_events,
        )
    except ImportError:
        return None

    inst     = load_solomon_raw(out_path)
    has_rain = bool(inst["preset_rain"])
    has_acc  = bool(inst["preset_acc"])
    N        = inst["n_customers"]

    # [1] Solve with base TT, then re-simulate with event effects
    tt_base = build_tt(inst)
    r_base  = solve(inst, tt_base, time_limit)

    if (has_rain or has_acc) and r_base["routes"]:
        ev = resimulate_with_events(r_base["routes"], inst)
        r1 = dict(
            status        = r_base["status"],
            routes        = r_base["routes"],
            vehicles_used = r_base["vehicles_used"],
            served_on_time= ev["served_on_time"],
            served_late   = ev["n_late_stops"],
            n_late_stops  = ev["n_late_stops"],
            unserved      = N - ev["served_on_time"] - ev["n_late_stops"],
            total_distance= ev["total_distance"],
            total_late    = ev["total_late"],
            elapsed       = r_base["elapsed"],
        )
    else:
        r1 = r_base

    # [2] Solve with event TT known upfront (perfect foresight upper bound)
    r2 = None
    if has_rain:
        r2 = solve(inst, build_tt_rain(inst), time_limit)
    elif has_acc:
        r2 = solve(inst, build_tt_accident(inst), time_limit)

    return {"r1": r1, "r2": r2, "N": N, "has_events": has_rain or has_acc}


def _fmt_row(label: str, r: dict, N: int) -> str:
    on       = r["served_on_time"]
    late     = r["n_late_stops"]
    unserved = r.get("unserved", N - on - late)
    dist     = r["total_distance"]
    tlate    = r["total_late"]
    return (f"  {label:<28}  {on:3d}/{N}  late={late:3d}  "
            f"unserved={unserved:3d}  dist={dist:7.1f}  late_mag={tlate:7.1f}")


def _print_ortools_table(
    inst_name: str,
    base_path: str,
    written_paths: dict[str, str],
    out_dir: str,
    time_limit: int = 15,
) -> None:
    """Evaluate all generated scenarios, print comparison table, and save to file."""
    try:
        from ortools_vrptw import (
            load_solomon_raw, build_tt, build_tt_rain,
            build_tt_accident, solve, resimulate_with_events,
        )
    except ImportError:
        print("  [EVAL] OR-Tools not available — skipping evaluation")
        return

    sep = "-" * 88
    lines = []

    def _p(s: str = ""):
        print(s)
        lines.append(s)

    _p(f"\n  {sep}")
    _p(f"  OR-Tools Evaluation  [{inst_name.upper()}]")
    hdr = (f"  {'Scenario':<24}  {'Plan':^8}  {'OnTime':>6}  {'Late':>5}"
           f"  {'Unserved':>8}  {'Veh':>4}  {'Dist':>8}  {'LateMin':>7}")
    _p(f"  {sep}")
    _p(hdr)
    _p(f"  {sep}")

    def _eval_and_print(label: str, path: str, plan_tag: str):
        inst     = load_solomon_raw(path)
        N        = inst["n_customers"]
        has_rain = bool(inst["preset_rain"])
        has_acc  = bool(inst["preset_acc"])

        tt_base = build_tt(inst)
        r_base  = solve(inst, tt_base, time_limit)
        veh     = r_base["vehicles_used"]
        if (has_rain or has_acc) and r_base["routes"]:
            ev = resimulate_with_events(r_base["routes"], inst)
            on, late, dist, tlate = (ev["served_on_time"], ev["n_late_stops"],
                                     ev["total_distance"], ev["total_late"])
        else:
            on, late, dist, tlate = (r_base["served_on_time"], r_base["n_late_stops"],
                                     r_base["total_distance"], r_base["total_late"])
        unserved = N - on - late
        _p(f"  {label:<24}  {'no-info':^7}  {on:3d}/{N}  {late:5d}"
           f"  {unserved:8d}  {veh:4d}  {dist:8.1f}  {tlate:7.1f}")

        if has_rain:
            r2 = solve(inst, build_tt_rain(inst), time_limit)
        elif has_acc:
            r2 = solve(inst, build_tt_accident(inst), time_limit)
        else:
            r2 = None

        if r2 is not None:
            on2, late2, dist2, tlate2 = (r2["served_on_time"], r2["n_late_stops"],
                                          r2["total_distance"], r2["total_late"])
            _p(f"  {'':24}  {'ev-aware':^7}  {on2:3d}/{N}  {late2:5d}"
               f"  {N-on2-late2:8d}  {r2['vehicles_used']:4d}  {dist2:8.1f}  {tlate2:7.1f}")

    _eval_and_print(f"{inst_name} (base)", base_path, "base")
    _p(f"  {'.'*80}")

    sc_order = ["rain_A", "rain_B", "acc_A", "acc_B"]
    for tag in sc_order:
        sc_name = f"{inst_name}_{tag}"
        path    = written_paths.get(sc_name)
        if path and os.path.isfile(path):
            _eval_and_print(sc_name, path, tag)
    _p(f"  {'-'*80}\n")

    # Save to file
    save_path = os.path.join(out_dir, f"ortools_eval_{inst_name}.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [Saved] {save_path}")


def _print_lkh_table(
    inst_name: str,
    base_path: str,
    written_paths: dict[str, str],
    out_dir: str,
    time_limit: int = 30,
    runs: int = 1,
) -> None:
    """Evaluate generated scenarios with LKH-3 and print comparison table.

    Tries to import R_lkh from the clustering_routing sibling directory.
    Falls back to OR-Tools evaluation if LKH-3 is unavailable.
    """
    _cr_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "clustering_routing")
    if _cr_dir not in sys.path:
        sys.path.insert(0, _cr_dir)

    try:
        from R_lkh import (
            load_solomon as load_solomon_instance, build_tt_matrix, build_rain_tt_matrix,
            parse_rain_events, solve_vrptw_lkh, evaluate_routes_with_rain,
        )
    except ImportError as exc:
        print(f"  [EVAL] R_lkh unavailable ({exc}) — falling back to OR-Tools")
        _print_ortools_table(inst_name, base_path, written_paths, out_dir)
        return

    sep = "-" * 90
    lines: list[str] = []
    notes: list[str] = []

    def _p(s: str = ""):
        print(s)
        lines.append(s)

    _p(f"\n  {sep}")
    _p(f"  LKH-3 Evaluation  [{inst_name.upper()}]  (time_limit={time_limit}s/solve)")
    hdr = (f"  {'Scenario':<24}  {'Plan':^8}  {'OnTime':>6}  {'Late':>5}"
           f"  {'Unserved':>8}  {'Veh':>4}  {'Dist':>8}  {'LateMin':>7}")
    _p(f"  {sep}")
    _p(hdr)
    _p(f"  {sep}")

    def _note_status(label: str, phase: str, status: str | None):
        if status == "feasible_extended":
            notes.append(
                f"  [Note] {label} ({phase}): LKH-3 needed an extended search "
                f"budget ({time_limit * 3}s) to certify a zero time-window-"
                f"violation solution (default {time_limit}s search found none)."
            )
        elif status == "best_effort":
            notes.append(
                f"  [Note] {label} ({phase}): LKH-3 could not certify a zero "
                f"time-window-violation solution within {time_limit * 3}s; "
                f"reported result is its best-effort (MTSP_OBJECTIVE=MINSUM) "
                f"solution with residual penalty."
            )

    def _eval_and_print(label: str, path: str):
        try:
            inst = load_solomon_instance(path)
        except Exception as exc:
            _p(f"  {label:<24}  [load error: {exc}]")
            return
        N           = inst["n_customers"]
        rain_events = parse_rain_events(path)
        has_events  = bool(rain_events)

        # ── no-info: solve with base TT, re-simulate with actual events ──────
        tt_base = build_tt_matrix(inst)
        r_base  = solve_vrptw_lkh(inst, tt_base, time_limit_sec=time_limit, runs=runs)
        veh     = r_base.get("vehicles_used", 0)
        _note_status(label, "no-info", r_base.get("status"))

        if has_events and r_base.get("routes"):
            ev   = evaluate_routes_with_rain(r_base["routes"], inst, rain_events)
            on   = ev["served_on_time"]
            late = ev["n_late_stops"]
            dist = ev["total_distance"]
            tlate= ev["total_late"]
        elif r_base.get("routes"):
            # base instance: no events, evaluate route metrics directly
            ev   = evaluate_routes_with_rain(r_base["routes"], inst, [])
            on   = ev["served_on_time"]
            late = ev["n_late_stops"]
            dist = ev["total_distance"]
            tlate= ev["total_late"]
        else:
            on, late, dist, tlate = 0, 0, 0.0, 0.0

        unserved = N - on - late
        _p(f"  {label:<24}  {'no-info':^7}  {on:3d}/{N}  {late:5d}"
           f"  {unserved:8d}  {veh:4d}  {dist:8.1f}  {tlate:7.1f}")

        # ── known: solve with event-aware TT, re-simulate ────────────────────
        if has_events:
            tt_ev   = build_rain_tt_matrix(inst, rain_events)
            r_known = solve_vrptw_lkh(inst, tt_ev, time_limit_sec=time_limit, runs=runs)
            _note_status(label, "known", r_known.get("status"))
            if r_known.get("routes"):
                ev2    = evaluate_routes_with_rain(r_known["routes"], inst, rain_events)
                on2    = ev2["served_on_time"]
                late2  = ev2["n_late_stops"]
                dist2  = ev2["total_distance"]
                tlate2 = ev2["total_late"]
                veh2   = r_known.get("vehicles_used", 0)
                _p(f"  {'':24}  {'known':^7}  {on2:3d}/{N}  {late2:5d}"
                   f"  {N-on2-late2:8d}  {veh2:4d}  {dist2:8.1f}  {tlate2:7.1f}")
            else:
                _p(f"  {'':24}  {'known':^7}  {'N/A':>6}  {'N/A':>5}"
                   f"  {'N/A':>8}  {'N/A':>4}  {'N/A':>8}  {'N/A':>7}")

    _eval_and_print(f"{inst_name} (base)", base_path)
    _p(f"  {'.' * 80}")

    for tag in ("rain_A", "rain_B", "acc_A", "acc_B"):
        sc_name = f"{inst_name}_{tag}"
        path    = written_paths.get(sc_name)
        if path and os.path.isfile(path):
            _eval_and_print(sc_name, path)

    _p(f"  {'-' * 80}")
    if notes:
        for note in notes:
            _p(note)
    _p("")

    save_path = os.path.join(out_dir, f"lkh_eval_{inst_name}.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [Saved] {save_path}")


# ── File writer ────────────────────────────────────────────────────────────────

def write_scenario(data: dict, events: list[dict], scenario_name: str, out_path: str):
    lines      = list(data["base_lines"])
    lines[0]   = scenario_name.upper()
    lines     += ["", "EVENTS", "# TYPE  TRIGGER_TIME  DURATION  MULTIPLIER/SEVERITY  NODES..."]

    for ev in events:
        nodes_str = " ".join(str(n) for n in ev["nodes"])
        if ev["type"] == "RAIN":
            lines.append(
                f"RAIN {ev['trigger_time']} {ev['duration']} "
                f"{ev['multiplier']} {ev['rainfall_mm']} {nodes_str}"
            )
        else:
            # Write the raw multiplier (single token, no whitespace) rather than
            # a natural-language description: the file format is whitespace-
            # delimited and severity_text (e.g. "다중 추돌, 2개 차로 통제") would
            # break column parsing. The existing loader already falls back to
            # float(token) when the token isn't a known severity-name key, so
            # this round-trips exactly without touching any downstream parser.
            lines.append(
                f"ACCIDENT {ev['trigger_time']} {ev['duration']} "
                f"{ev['multiplier']} {nodes_str}"
            )
        lines.append("")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


# ── Per-instance generator ─────────────────────────────────────────────────────

def generate_for_instance(
    inst_name: str,
    data_dir: str,
    out_dir: str,
    count: int,
    seed: int | None,
    dry_run: bool,
):
    src_path = os.path.join(data_dir, inst_name + ".txt")
    if not os.path.isfile(src_path):
        print(f"  [SKIP] {src_path} not found")
        return

    data    = load_raw(src_path)
    sectors = sector_info(data, n_sectors=8)
    T       = data["T"]
    N       = data["n_customers"]

    # Baseline routes: fresh LKH-3 solve (matches no-info eval) → OR-Tools → greedy.
    # Using the no-info eval's own routes ensures A-type events are placed on
    # edges/timing that the no-info evaluation actually traverses (otherwise A
    # silently degenerates into a zero-impact B-type event).
    baseline_routes = None
    if not dry_run:
        baseline_routes = _lkh_fresh_routes(src_path)
        if baseline_routes:
            print(f"  [LKH-3 fresh] {len(baseline_routes)} routes for event placement"
                  f" (= no-info eval baseline)")
        else:
            baseline_routes = _ortools_routes(src_path)
            if baseline_routes:
                print(f"  [OR-Tools]  {len(baseline_routes)} routes for event placement")
    if not baseline_routes:
        baseline_routes = _greedy_routes(data)
        print(f"  [Greedy]    {len(baseline_routes)} routes for event placement (LKH/OR-Tools skipped)")

    # Per-instance late targets scaled by N
    rain_late_min = max(1, int(N * RAIN_LATE_MIN_PCT))
    rain_late_max = max(2, int(N * RAIN_LATE_MAX_PCT))
    acc_late_min  = max(1, int(N * ACC_LATE_MIN_PCT))
    acc_late_max  = max(2, int(N * ACC_LATE_MAX_PCT))

    print(f"\n[{inst_name.upper()}]  T={T:.0f}  N={N}  scenarios={count}"
          f"  late_targets: rain={rain_late_min}-{rain_late_max}"
          f"  acc={acc_late_min}-{acc_late_max}"
          f"  ({RAIN_LATE_MIN_PCT:.0%}-{RAIN_LATE_MAX_PCT:.0%} / "
          f"{ACC_LATE_MIN_PCT:.0%}-{ACC_LATE_MAX_PCT:.0%} of N)")

    sec = sector_of(data, 8)
    cfg = GenConfig()

    def _make_binding_invalid_pair(kind: str) -> tuple[list[dict], list[dict]]:
        """A = policy-independently binding event (Part B-4/B-5 stage 1: label
        "overlap" with n_covered >= 1). B = the SAME sectors/intensity,
        time-shifted past every affected customer's TW_close (Part B, 'B
        시나리오 설계 주의') -- invalid for any policy, since no bound customer
        can still be pending at that time. Neither uses baseline_routes; only
        instance data (coords/tw/service)."""
        ev = draw_until_stage(data, kind, cfg, rng, stage=1, sec=sec)
        if ev is None:
            print(f"  [WARN] {kind}: no stage-1 (overlap, n_covered>=1) event found")
            return [], []
        ev_a = [ev] if kind == "RAIN" else (generate_accident_wave(data, cfg, rng, sec=sec) or [ev])

        sector_set = set(ev["sectors"])
        inside = [i for i in range(1, N + 1) if int(sec[i - 1]) in sector_set]
        margin = 10.0
        sigma_b = max((data["tw_close"][i] for i in inside), default=0.0) + margin
        delta_b = ev["duration"]
        if sigma_b + delta_b > T:
            sigma_b, delta_b = 0.0, min(delta_b, max(data["tt"][0, inside].min() - margin, _MIN_DURATION))
        ev_b = [dict(ev, trigger_time=round(sigma_b, 1), duration=round(delta_b, 1),
                     label="invalidated_by_timeshift", bound_set=[])]
        return ev_a, ev_b

    for i in range(count):
        sc_seed = (seed + i) if seed is not None else None
        rng     = np.random.default_rng(sc_seed)

        # count=1 → plain name (backward compat); count>1 → numbered
        suffix = f"_{i+1:03d}" if count > 1 else ""

        rain_A, rain_B = _make_binding_invalid_pair("RAIN")
        acc_A,  acc_B  = _make_binding_invalid_pair("ACCIDENT")
        for tag, evs in (("rain_A", rain_A), ("acc_A", acc_A)):
            if evs:
                b = evs[0].get("bound_set", [])
                print(f"  [{tag}] sectors={evs[0]['sectors']} mu={evs[0].get('multiplier')} "
                      f"bound={len(b)} customers  (label={evs[0].get('label')})")

        scenarios = [
            (f"{inst_name}_rain_A{suffix}", rain_A, "rain HIGH-IMPACT"),
            (f"{inst_name}_rain_B{suffix}", rain_B, "rain NO-IMPACT"),
            (f"{inst_name}_acc_A{suffix}",  acc_A,  "accident HIGH-IMPACT"),
            (f"{inst_name}_acc_B{suffix}",  acc_B,  "accident NO-IMPACT"),
        ]

        # ── Write files ───────────────────────────────────────────────────────
        written_paths: dict[str, str] = {}   # sc_name → out_path
        for sc_name, events, desc in scenarios:
            if not events:
                print(f"  [WARN] No events generated for {sc_name}")
                continue
            is_high = sc_name.endswith("_A") or sc_name.endswith("_A" + suffix)
            if is_high and baseline_routes and not dry_run:
                late = simulate_late(data, baseline_routes, events)
                late_str = f"  [sim late={late}]"
            else:
                late_str = ""
            ev_str = "  |  ".join(
                f"{e['type']} t={e['trigger_time']} d={e['duration']} "
                f"mult={e.get('multiplier', e.get('severity', '?'))} nodes={e['nodes']}"
                for e in events
            )
            print(f"  {'[DRY] ' if dry_run else ''}-> {sc_name}.txt  [{desc}]{late_str}")
            print(f"       {ev_str}")
            if not dry_run:
                out_path = os.path.join(out_dir, sc_name + ".txt")
                write_scenario(data, events, sc_name, out_path)
                written_paths[sc_name] = out_path

        # ── LKH-3 evaluation ─────────────────────────────────────────────────
        if not dry_run and written_paths:
            _print_lkh_table(inst_name, src_path, written_paths, out_dir)


# ── M_base / M_true_hist LKH-3 baseline evaluation ────────────────────────────

EVENT_SUFFIXES_HIST = ["rain_A", "rain_B", "acc_A", "acc_B"]


def _compute_time_avg_tt(base_tt: np.ndarray, events_raw: list, T: float) -> np.ndarray:
    """Event-conditioned TT for one scenario.
    Uses full multiplier (not time-weighted): LKH-3 is non-time-dependent,
    so it should plan with the actual event cost, not a diluted average.
    """
    n = base_tt.shape[0]
    factor = np.ones((n, n), dtype=np.float64)
    for trigger, duration, mult, nodes in events_raw:
        t_s = max(0.0, float(trigger))
        t_e = min(T, float(trigger) + float(duration))
        if t_e <= t_s:
            continue
        if len(nodes) == 2:
            a, b = int(nodes[0]), int(nodes[1])
            if a < n and b < n:
                factor[a, b] = max(factor[a, b], float(mult))
                factor[b, a] = max(factor[b, a], float(mult))
        else:
            ns = set(int(x) for x in nodes if int(x) < n)
            for i in ns:
                for j in ns:
                    if i != j:
                        factor[i, j] = max(factor[i, j], float(mult))
    return base_tt * factor



def _build_m_true_hist(base_tt: np.ndarray, train_instances: list, T: float, data_dir: str) -> np.ndarray:
    """Average time-weighted TT over base + 4 event files per training instance."""
    _cr_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "clustering_routing")
    if _cr_dir not in sys.path:
        sys.path.insert(0, _cr_dir)
    from R_lkh import parse_rain_events

    matrices = []
    for inst_name in train_instances:
        matrices.append(base_tt.copy())  # base (no-event) per training instance
        for suffix in EVENT_SUFFIXES_HIST:
            ev_path = os.path.join(data_dir, f"{inst_name}_{suffix}.txt")
            if not os.path.isfile(ev_path):
                continue
            evs = parse_rain_events(ev_path)
            if evs:
                matrices.append(_compute_time_avg_tt(base_tt, evs, T))
            else:
                matrices.append(base_tt.copy())
    M = np.mean(matrices, axis=0)
    mask = base_tt > 0
    pct = float(np.mean((M[mask] / base_tt[mask] - 1.0) * 100))
    print(f"  M_true_hist: avg over {len(matrices)} scenarios (+{pct:.2f}% vs M_base)")
    return M


def _eval_lkh_baselines(benchmark: str, data_dir: str, time_limit: int = 60) -> None:
    """Compute M_base and M_true_hist LKH-3 plans; evaluate on all test scenarios."""
    cfg = BENCHMARK_CONFIGS.get(benchmark)
    if cfg is None:
        print(f"  [baseline] Unknown benchmark '{benchmark}', skipping.")
        return

    _cr_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "clustering_routing")
    if _cr_dir not in sys.path:
        sys.path.insert(0, _cr_dir)
    try:
        from R_lkh import load_solomon, solve_vrptw_lkh, parse_rain_events, evaluate_routes_with_rain
    except ImportError as exc:
        print(f"  [baseline] R_lkh unavailable: {exc}")
        return

    test_name   = cfg["test"]
    train_insts = cfg["train"]
    sc_names    = [test_name] + cfg["scenarios"]

    test_path = os.path.join(data_dir, f"{test_name}.txt")
    if not os.path.isfile(test_path):
        print(f"  [baseline] Test file not found: {test_path}")
        return

    inst_test = load_solomon(test_path)
    N = inst_test["n_customers"]
    T = inst_test["tw_close"][0]

    # Build base TT from test instance coords
    coords = np.array(inst_test["coords"])
    diff   = coords[:, None, :] - coords[None, :, :]
    base_tt = np.sqrt((diff ** 2).sum(-1))

    print(f"\n  Building M_true_hist from {train_insts[0]}-{train_insts[-1]} "
          f"({len(train_insts)} train instances)...")
    hist_tt = _build_m_true_hist(base_tt, train_insts, T, data_dir)

    def _to_lkh(M):
        n = M.shape[0]
        return [[int(round(float(M[i, j]))) for j in range(n)] for i in range(n)]

    D_MAX = 10_000.0
    W_D, W_LC, W_LT, W_K = 1.0, 4.0, 4.0, 1.0

    print(f"\n  [LKH-3] Solving M_base plan (time_limit={time_limit}s)...")
    res_b = solve_vrptw_lkh(inst_test, _to_lkh(base_tt), time_limit_sec=time_limit)
    routes_b = res_b["routes"]
    print(f"    K={res_b['vehicles_used']}  dist={res_b['total_distance']:.1f}  ({res_b['elapsed']:.1f}s)")

    print(f"  [LKH-3] Solving M_true_hist plan (time_limit={time_limit}s)...")
    res_h = solve_vrptw_lkh(inst_test, _to_lkh(hist_tt), time_limit_sec=time_limit)
    routes_h = res_h["routes"]
    print(f"    K={res_h['vehicles_used']}  dist={res_h['total_distance']:.1f}  ({res_h['elapsed']:.1f}s)")

    sep = "=" * 92
    hdr = f"{'Scenario':<22} {'Plan':<14} {'K':>4} {'Lc':>5} {'Lt':>9} {'D':>8} {'R_F':>8}"
    print(f"\n  {sep}")
    print(f"  LKH-3 BASELINES [{benchmark.upper()}]  (Config F)")
    print(f"  {sep}")
    print(f"  {hdr}")
    print(f"  {'-'*88}")

    rewards_b, rewards_h = [], []
    for sc_name in sc_names:
        sc_path = os.path.join(data_dir, f"{sc_name}.txt")
        if not os.path.isfile(sc_path):
            print(f"  {sc_name:<22}  [file not found]")
            continue
        inst_sc = load_solomon(sc_path)
        events  = parse_rain_events(sc_path)

        def _sim(routes):
            ev = evaluate_routes_with_rain(routes, inst_sc, events)
            K  = sum(1 for r in routes if r)
            return ev["total_distance"], ev["n_late_stops"], ev["total_late"], K

        def _rf(D, Lc, Lt, K):
            return -(W_D * D / D_MAX + W_LC * Lc / N + W_LT * Lt / (T * N) + W_K * K / N)

        Db, Lb, Ltb, Kb = _sim(routes_b)
        rb = _rf(Db, Lb, Ltb, Kb); rewards_b.append(rb)
        print(f"  {sc_name:<22} {'M_base':<14} {Kb:>4d} {Lb:>5d} {Ltb:>9.1f} {Db:>8.1f} {rb:>8.4f}")

        Dh, Lh, Lth, Kh = _sim(routes_h)
        rh = _rf(Dh, Lh, Lth, Kh); rewards_h.append(rh)
        print(f"  {'':22} {'M_true_hist':<14} {Kh:>4d} {Lh:>5d} {Lth:>9.1f} {Dh:>8.1f} {rh:>8.4f}")
        print()

    print(f"  {'-'*88}")
    print(f"  {'Average':>22} {'M_base':<14} {'':>5}{'':>6}{'':>10}{'':>9} {sum(rewards_b)/len(rewards_b):>8.4f}")
    print(f"  {'':>22} {'M_true_hist':<14} {'':>5}{'':>6}{'':>10}{'':>9} {sum(rewards_h)/len(rewards_h):>8.4f}")
    print(f"  {sep}\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate randomised VRPTW event scenario files for Solomon instances."
    )
    parser.add_argument(
        "instances", nargs="*", default=DEFAULT_INSTANCES,
        help="Instance names without extension (default: c101-c109, c201-c208)",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="Directory containing Solomon .txt files")
    parser.add_argument("--out-dir",  default=DEFAULT_DATA_DIR,
                        help="Output directory (default: same as data-dir)")
    parser.add_argument("--count", type=int, default=1,
                        help="Number of random scenarios per instance (default: 1)")
    parser.add_argument("--seed",  type=int, default=42,
                        help="Base RNG seed for reproducibility; scenario i uses seed+i")
    parser.add_argument("--list",  action="store_true",
                        help="Dry run: print what would be generated without writing")
    parser.add_argument("--r1",    action="store_true",
                        help="Use R1xx instances (r101-r112) with R-type parameters")
    parser.add_argument("--benchmark", default=None, choices=list(BENCHMARK_CONFIGS),
                        help="After generation, also evaluate M_base and M_true_hist LKH-3 baselines "
                             "(c1/rc1/r1/c2)")
    parser.add_argument("--baseline-time-limit", type=int, default=60,
                        help="LKH-3 time limit (s) for baseline evaluation (default: 60)")
    parser.add_argument("--random", action="store_true",
                        help="Generate random (non-route-targeted) rain+acc scenarios instead of A/B pairs. "
                             "Produces {inst}_rand_rain.txt and {inst}_rand_acc.txt per scenario.")
    args = parser.parse_args()

    # R1xx override: use R-type defaults if --r1 and no explicit instances given
    instances = args.instances
    if args.r1 and instances == DEFAULT_INSTANCES:
        instances = DEFAULT_R1_INSTANCES

    print(f"Data dir  : {args.data_dir}")
    print(f"Output    : {args.out_dir}")
    print(f"Instances : {instances}")
    print(f"Count     : {args.count}")
    print(f"Mode      : {'random (no route targeting)' if args.random else 'A/B pair (route-targeted)'}")
    if args.seed is not None:
        print(f"Seed      : {args.seed}")
    if args.list:
        print("(DRY RUN -- no files written)\n")

    for inst in instances:
        if args.random:
            generate_random_for_instance(
                inst, args.data_dir, args.out_dir,
                args.count, args.seed, args.list,
            )
        else:
            generate_for_instance(
                inst, args.data_dir, args.out_dir,
                args.count, args.seed, args.list,
            )

    if args.benchmark and not args.list:
        print(f"\n{'='*60}")
        print(f"  Running M_base / M_true_hist baselines for [{args.benchmark.upper()}]")
        print(f"{'='*60}")
        _eval_lkh_baselines(args.benchmark, args.data_dir, args.baseline_time_limit)

    print("\nDone.")


if __name__ == "__main__":
    main()
