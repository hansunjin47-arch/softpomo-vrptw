"""
compute_historical_avg_tt.py

Computes historical-average TT matrices from C102-C109 event scenarios.
All C1xx instances share the same coordinates → same base TT.

Time-averaged TT for edge (i,j):
  base_TT[i,j] + sum over events of base_TT[i,j] * (mult-1) * overlap_fraction
where overlap_fraction = min(T, t_end) - max(0, t_start) / T

Outputs (saved to <data_dir>/historical_avg/):
  M_base.npy      : base Euclidean TT (raw, time units)
  M_rain_avg.npy  : time-averaged TT averaged over RAIN_A/B x C102-C109
  M_all_avg.npy   : time-averaged TT averaged over all 4 event files x C102-C109

LKH-3 planning strategy:
  - C101 BASE / ACC scenarios → use M_base (no advance knowledge of accidents)
  - C101 RAIN scenarios       → use M_rain_avg (weather forecast available)
"""
import os
import sys
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(DATA_DIR, "historical_avg")
os.makedirs(OUT_DIR, exist_ok=True)

INSTANCES = [f"c10{i}" for i in range(2, 10)]  # c102 .. c109
EVENT_SUFFIXES = ["rain_A", "rain_B", "acc_A", "acc_B"]

_ACCIDENT_SEVERITY = {
    "3-car-collision": 5.0,
    "4-car-pile-up":   8.5,
    "5-car-pile-up":  13.0,
    "low": 1.5, "medium": 2.0, "high": 3.0,
    "1":  1.5, "2":  2.0, "3":  2.8, "4":  3.5, "5":  5.0,
    "6":  6.5, "7":  8.5, "8": 10.5, "9": 13.0, "10": 15.0,
}


def load_base_tt(path: str):
    """Load Solomon file, return (raw_tt [N+1,N+1], T, n_nodes)."""
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    c_idx = next(i for i, l in enumerate(lines) if l.upper().startswith("CUSTOMER"))
    rows = []
    for ln in lines[c_idx + 1:]:
        toks = ln.split()
        if len(toks) < 7:
            continue
        try:
            rows.append([float(x) for x in toks[:7]])
        except ValueError:
            continue

    data = np.array(rows, dtype=np.float64)
    T    = float(data[0, 5])           # depot due date = planning horizon
    coords = data[:, 1:3]
    n    = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    tt   = np.sqrt((diff ** 2).sum(-1))
    return tt, T, n


def load_events(path: str):
    """Parse EVENTS section from a Solomon file. Returns list of event dicts."""
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    events = []
    try:
        ev_start = next(i for i, l in enumerate(lines) if l.upper() == "EVENTS")
    except StopIteration:
        return events

    for ln in lines[ev_start + 1:]:
        toks = ln.split()
        if not toks or toks[0].startswith('#'):
            continue
        kw = toks[0].upper()
        if kw not in ("RAIN", "ACCIDENT"):
            continue
        trigger  = float(toks[1])
        duration = float(toks[2])
        if kw == "RAIN":
            multiplier  = float(toks[3])
            # toks[4] = rainfall_mm, toks[5:] = nodes
            nodes = [int(x) for x in toks[5:]]
        else:
            raw  = toks[3].lower()
            multiplier = _ACCIDENT_SEVERITY.get(raw)
            if multiplier is None:
                try:
                    multiplier = float(raw)
                except ValueError:
                    multiplier = 5.0
            nodes = [int(x) for x in toks[4:]]
        events.append(dict(
            type=kw, trigger=trigger, duration=duration,
            multiplier=multiplier, nodes=nodes,
        ))
    return events


def compute_avg_tt(base_tt: np.ndarray, events: list, T: float) -> np.ndarray:
    """
    Time-averaged TT under given events over horizon [0, T].
    avg_TT[i,j] = base_TT[i,j] * (1 + sum_ev (mult-1) * overlap_fraction)
    """
    n = base_tt.shape[0]
    # Accumulate multiplicative factor for each edge
    factor = np.ones((n, n), dtype=np.float64)

    for ev in events:
        t_s = ev['trigger']
        t_e = t_s + ev['duration']
        t_s_eff = max(0.0, t_s)
        t_e_eff = min(T, t_e)
        if t_e_eff <= t_s_eff:
            continue
        frac = (t_e_eff - t_s_eff) / T
        mult = ev['multiplier']

        if ev['type'] == 'RAIN':
            ns = set(ev['nodes'])
            for i in ns:
                if i >= n:
                    continue
                for j in ns:
                    if j >= n or i == j:
                        continue
                    factor[i, j] += (mult - 1.0) * frac
        else:  # ACCIDENT — bidirectional single edge
            a, b = ev['nodes'][0], ev['nodes'][1]
            if a < n and b < n:
                factor[a, b] += (mult - 1.0) * frac
                factor[b, a] += (mult - 1.0) * frac

    return base_tt * factor


def main():
    # ── Load base TT from c102.txt (same coords for all C1xx) ─────────────────
    base_path = os.path.join(DATA_DIR, "c102.txt")
    base_tt, T, n = load_base_tt(base_path)
    print(f"Base TT: {n}×{n}, T={T}")

    # ── Accumulators ──────────────────────────────────────────────────────────
    # true_hist includes BASE (no-event) + all 4 event files per instance
    # = 5 scenarios × 8 instances = 40 matrices
    rain_matrices  = []     # RAIN_A + RAIN_B for each C102-C109 (16 total)
    all_matrices   = []     # all 4 event files for each C102-C109 (32 total)
    true_hist_mats = []     # base + all 4 events per instance (40 total)

    for inst_name in INSTANCES:
        # Include the base (no-event) scenario for this instance
        true_hist_mats.append(base_tt.copy())

        for suffix in EVENT_SUFFIXES:
            ev_path = os.path.join(DATA_DIR, f"{inst_name}_{suffix}.txt")
            if not os.path.isfile(ev_path):
                print(f"  [WARN] missing: {ev_path}")
                continue
            events = load_events(ev_path)
            avg_tt = compute_avg_tt(base_tt, events, T)

            all_matrices.append(avg_tt)
            true_hist_mats.append(avg_tt)
            if "rain" in suffix:
                rain_matrices.append(avg_tt)

            n_rain = sum(1 for e in events if e['type'] == 'RAIN')
            n_acc  = sum(1 for e in events if e['type'] == 'ACCIDENT')
            _mask = base_tt > 0
            diff  = float(np.mean((avg_tt[_mask] / base_tt[_mask] - 1.0) * 100))
            print(f"  {inst_name}_{suffix}: {n_rain} RAIN, {n_acc} ACC  "
                  f"-> avg TT +{diff:.2f}%")

    # ── Compute averages ───────────────────────────────────────────────────────
    M_base      = base_tt
    M_rain_avg  = np.mean(rain_matrices,  axis=0)
    M_all_avg   = np.mean(all_matrices,   axis=0)
    M_true_hist = np.mean(true_hist_mats, axis=0)  # base + all 4 events per instance

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(os.path.join(OUT_DIR, "M_base.npy"),       M_base)
    np.save(os.path.join(OUT_DIR, "M_rain_avg.npy"),   M_rain_avg)
    np.save(os.path.join(OUT_DIR, "M_all_avg.npy"),    M_all_avg)
    np.save(os.path.join(OUT_DIR, "M_true_hist.npy"),  M_true_hist)

    print()
    print(f"Saved to {OUT_DIR}:")
    print(f"  M_base.npy      : base Euclidean TT (no events)")
    print(f"  M_rain_avg.npy  : avg over {len(rain_matrices)} RAIN scenarios")
    print(f"  M_all_avg.npy   : avg over {len(all_matrices)} event-only scenarios")
    print(f"  M_true_hist.npy : avg over {len(true_hist_mats)} scenarios (base + all events)")
    print()

    # ── Summary stats ─────────────────────────────────────────────────────────
    _mask      = M_base > 0
    diff_rain  = float(np.mean((M_rain_avg[_mask]  / M_base[_mask] - 1.0) * 100))
    diff_all   = float(np.mean((M_all_avg[_mask]   / M_base[_mask] - 1.0) * 100))
    diff_hist  = float(np.mean((M_true_hist[_mask] / M_base[_mask] - 1.0) * 100))
    print(f"M_rain_avg  vs base: avg TT +{diff_rain:.2f}%")
    print(f"M_all_avg   vs base: avg TT +{diff_all:.2f}%")
    print(f"M_true_hist vs base: avg TT +{diff_hist:.2f}%  (base+all events per instance)")


if __name__ == "__main__":
    main()
