"""
generate_c_rc_tw_variants.py — TW-augmented variants of the Solomon C1 / RC1 geometry.

Within each Solomon family, all instances share identical coordinates/depot/
demands/service times -- only the time windows differ. This keeps every fixed
field byte-identical (rewrites the template .txt text, touching only READY
TIME / DUE DATE).

Verified gap (constrained-customer width median + density; computed directly
from the real instances, not the misleading mean -- see generate_r1_tw_variants.py
for the full derivation, which applies identically here):

  RC1: rc101* width~30 density=100%  |  rc102 width~30 density=75%
       rc105 width=39/rc106 width=60 already cover density=100% but at WIDER
       widths. The (width=30, density=100%) cell has no training-pool peer.
  C1:  c101*  width~61 density=100%  |  c102 width~61 density=75%
       c105-c109 already cover density=100% at widths 123-360 (wider).
       The (width=61, density=100%) cell has no training-pool peer.

RC1 uses the same uniform-random-centre procedure as R1 (already matches
Solomon's own R/RC procedure -- see generate_r1_tw_variants.py's docstring).

C1 does NOT: Solomon (1987) sec.2 generates C-type windows by first building a
per-cluster 3-opt tour and centring each customer's window on that tour's
ARRIVAL TIME at the customer, not by drawing an independent random centre.
Drawing an independent centre (what this script did before) breaks the
sequential feasibility that makes a single cluster-by-cluster route optimal,
which is why the naive uniform-centre C1 variants needed K=13-20 vehicles
against the real C1 family's constant K=10. See _cluster_centres().

Usage:
  python generate_c_rc_tw_variants.py                # both families -> data/Solomon
  python generate_c_rc_tw_variants.py --dry-run
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA_DIR = os.path.join(_ROOT, 'data', 'Solomon')

# RC1: width jittered tightly around rc101's own width (30) -- verified gap,
# see module docstring. 5 widths x 3 seeds = 15 instances.
_FAMILIES = {
    'RC1': dict(
        template='RC102.txt',
        prefix_map={'H': 'RCH'},
        tiers={'H': [25, 28, 30, 32, 35]},
        seeds_per_width=3,
    ),
}

# C1: width jittered around c101's own width (61) -- verified gap. Centres come
# from the cluster-tour arrival times (_cluster_centres), NOT a random draw.
C1_TEMPLATE   = 'C102.txt'
C1_WIDTHS     = [50, 55, 61, 65, 70]
C1_SEEDS_PER_WIDTH = 3   # 5 x 3 = 15 instances


def _parse_template(path: str):
    with open(path, encoding='utf-8') as f:
        lines = f.read().splitlines()
    hdr_idx = next(i for i, ln in enumerate(lines)
                   if ln.strip().upper().startswith('CUST NO'))
    header = lines[:hdr_idx + 2]
    rows = []
    for ln in lines[hdr_idx + 2:]:
        parts = ln.split()
        if len(parts) != 7:
            continue
        rows.append([int(float(p)) for p in parts])
    return header, rows


def _write_instance(path: str, name: str, header, depot, custs):
    out = list(header)
    out[0] = name.upper()
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')
        for r in [depot] + custs:
            f.write(f"  {r[0]:4d}  {r[1]:7d}  {r[2]:7d}  {r[3]:9d}"
                    f"  {r[4]:11d}  {r[5]:10d}  {r[6]:9d}\n")


def _make_tw(depot, custs, width: float, T: float, rng) -> tuple[list, int]:
    dx, dy = depot[1], depot[2]
    out = []
    n_unconstrained = 0
    for c in custs:
        cid, x, y, dem, _, _, svc = c
        d = math.hypot(x - dx, y - dy)
        lo = d
        hi = T - d - svc
        if hi <= lo:
            ready, due = 0, int(T)
            n_unconstrained += 1
        else:
            centre = float(rng.uniform(lo, hi))
            ready_f = max(lo, centre - width / 2.0)
            due_f = min(T, ready_f + width)
            ready_f = max(0.0, due_f - width)
            ready, due = int(round(ready_f)), int(round(due_f))
            if due <= ready:
                due = min(int(T), ready + max(1, int(round(width))))
        out.append([cid, x, y, dem, ready, due, svc])
    return out, n_unconstrained


def generate_family(fam_name: str, cfg: dict, seed: int, out_dir: str, dry_run: bool) -> list[str]:
    template_path = os.path.join(DATA_DIR, cfg['template'])
    header, rows = _parse_template(template_path)
    depot, custs = rows[0], rows[1:]
    T = float(depot[5])

    print(f'\n[{fam_name}] template={cfg["template"]}  N={len(custs)}  T={T:.0f}  '
          f'depot=({depot[1]},{depot[2]})')
    print(f'{"name":<8} {"tier":<7} {"width":>6} {"mean_w":>7} {"min_w":>6} {"open":>5}')
    print('-' * 46)

    made = []
    for tier, widths in cfg['tiers'].items():
        prefix = cfg['prefix_map'][tier]
        seeds_per_width = cfg.get('seeds_per_width', 1)
        total = len(widths) * seeds_per_width
        for idx in range(1, total + 1):
            width = widths[(idx - 1) % len(widths)]
            rng = np.random.default_rng(seed * 1000 + hash(fam_name + tier) % 97 * 100 + idx)
            new_custs, n_open = _make_tw(depot, custs, float(width), T, rng)
            name = f'{prefix}{idx:02d}'
            w = np.array([c[5] - c[4] for c in new_custs], dtype=float)
            print(f'{name:<8} {tier:<7} {width:>6} {w.mean():>7.1f} {w.min():>6.0f} {n_open:>5}')
            if not dry_run:
                _write_instance(os.path.join(out_dir, f'{name}.txt'),
                                name, header, depot, new_custs)
            made.append(name)
    return made


# ── C1: cluster-tour arrival-time-centred TW (Solomon's actual C-type procedure) ──

def _kmeans(coords: np.ndarray, K: int, seed: int, n_init: int = 20) -> np.ndarray:
    """Plain Lloyd's-algorithm K-means (no sklearn dependency). Returns the
    best-of-n_init labels array (by inertia)."""
    rng = np.random.default_rng(seed)
    best_labels, best_inertia = None, float('inf')
    n = coords.shape[0]
    for trial in range(n_init):
        centres = coords[rng.choice(n, size=K, replace=False)].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(100):
            d = np.linalg.norm(coords[:, None, :] - centres[None, :, :], axis=2)
            new_labels = d.argmin(axis=1)
            if np.array_equal(new_labels, labels) and _ > 0:
                break
            labels = new_labels
            for k in range(K):
                mask = labels == k
                if mask.any():
                    centres[k] = coords[mask].mean(axis=0)
        d = np.linalg.norm(coords[:, None, :] - centres[None, :, :], axis=2)
        inertia = float(d[np.arange(n), labels].sum())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels.copy()
    return best_labels


def _held_karp_tour(dist: np.ndarray) -> list[int]:
    """Exact TSP over indices [0..n-1] (0 = depot), returning the visiting
    order of 1..n-1 (depot excluded) that minimises the open depot->...->depot
    tour length. O(2^n * n^2); fine for n <= ~14 (per-cluster customer counts)."""
    n = dist.shape[0]
    m = n - 1                      # customers only (indices 1..n-1 -> 0..m-1 here)
    if m == 0:
        return []
    if m == 1:
        return [1]
    FULL = 1 << m
    dp = np.full((FULL, m), np.inf)
    parent = np.full((FULL, m), -1, dtype=int)
    for i in range(m):
        dp[1 << i, i] = dist[0, i + 1]
    for mask in range(FULL):
        for i in range(m):
            if not (mask & (1 << i)) or dp[mask, i] == np.inf:
                continue
            for j in range(m):
                if mask & (1 << j):
                    continue
                nmask = mask | (1 << j)
                cost = dp[mask, i] + dist[i + 1, j + 1]
                if cost < dp[nmask, j]:
                    dp[nmask, j] = cost
                    parent[nmask, j] = i
    full = FULL - 1
    last = int(np.argmin(dp[full, :] + dist[np.arange(1, n), 0]))
    order = []
    mask = full
    while last != -1:
        order.append(last)
        pmask = mask & ~(1 << last)
        prev = parent[mask, last]
        mask = pmask
        last = prev
    order.reverse()
    return [i + 1 for i in order]   # back to 1-based customer indices (within cluster)


def _cluster_centres(depot, custs, seed: int) -> dict[int, float]:
    """Solomon (1987) C-type procedure: cluster customers by capacity, find each
    cluster's optimal (Held-Karp) tour, pick the shorter-length orientation,
    and set each customer's TW centre to its ARRIVAL TIME along that tour.

    Because the tour that defines the centres is always feasible by
    construction (arrival time IS the centre), any width >= 0 keeps that same
    tour feasible -- this is why real C1's K stays at 10 regardless of width.
    """
    coords = np.array([[c[1], c[2]] for c in custs], dtype=float)
    demand = np.array([c[3] for c in custs], dtype=float)
    cap = 200.0   # Solomon C1 vehicle capacity
    K = math.ceil(demand.sum() / cap)

    labels = _kmeans(coords, K, seed)
    # Capacity-respecting fix-up: move overflow customers to the nearest
    # under-capacity cluster (plain K-means has no capacity awareness).
    for _pass in range(20):
        loads = np.array([demand[labels == k].sum() for k in range(K)])
        over = np.where(loads > cap)[0]
        if len(over) == 0:
            break
        k = over[0]
        idxs = np.where(labels == k)[0]
        centre_k = coords[idxs].mean(axis=0)
        far = idxs[np.argmax(np.linalg.norm(coords[idxs] - centre_k, axis=1))]
        other_loads = [(j, demand[labels == j].sum()) for j in range(K) if j != k]
        target = min(other_loads, key=lambda t: t[1])[0]
        if demand[far] + other_loads[[t[0] for t in other_loads].index(target)][1] <= cap:
            labels[far] = target
    loads = np.array([demand[labels == k].sum() for k in range(K)])
    if (loads > cap + 1e-6).any():
        print(f'  [WARN] cluster capacity exceeded after fix-up: loads={loads.tolist()}')

    depot_xy = np.array([depot[1], depot[2]], dtype=float)
    centres: dict[int, float] = {}
    for k in range(K):
        idxs = np.where(labels == k)[0]
        if len(idxs) == 0:
            continue
        cluster_ids = [custs[i][0] for i in idxs]
        sub_xy = np.vstack([depot_xy[None, :], coords[idxs]])
        dist = np.linalg.norm(sub_xy[:, None, :] - sub_xy[None, :, :], axis=2)
        order = _held_karp_tour(dist)               # indices into idxs (1-based)
        fwd_len = sum(dist[0 if p == 0 else order[p - 1], o]
                     for p, o in enumerate(order)) + dist[order[-1], 0]
        rev_order = list(reversed(order))
        rev_len = sum(dist[0 if p == 0 else rev_order[p - 1], o]
                     for p, o in enumerate(rev_order)) + dist[rev_order[-1], 0]
        chosen = order if fwd_len <= rev_len else rev_order

        t, prev = 0.0, 0
        for o in chosen:
            t += dist[prev, o]
            cid = cluster_ids[o - 1]
            centres[cid] = t
            svc = custs[idxs[o - 1]][6]
            t += svc
            prev = o
    return centres


def _make_tw_centred(depot, custs, centres: dict[int, float], width: float,
                     T: float, rng) -> tuple[list, int]:
    """Like _make_tw, but centre[i] = centres[cid] (cluster arrival time)
    instead of an independent random draw. Re-expands the window if clamping
    would push the arrival time itself outside [ready, due] (Part 2.4's
    boundary-clamp safeguard)."""
    dx, dy = depot[1], depot[2]
    out = []
    for c in custs:
        cid, x, y, dem, _, _, svc = c
        d = math.hypot(x - dx, y - dy)
        arrival = centres[cid]
        ready_f = max(0.0, arrival - width / 2.0)
        due_f = min(T, arrival + width / 2.0)
        if due_f <= ready_f:
            due_f = min(T, ready_f + max(1.0, width))
        # Guarantee: the tour's own arrival time must stay inside the window.
        if arrival < ready_f:
            ready_f = arrival
        if arrival > due_f:
            due_f = arrival
        ready, due = int(round(ready_f)), int(round(due_f))
        out.append([cid, x, y, dem, ready, due, svc])
    return out, 0


def generate_c1_cluster_tw(seed: int, out_dir: str, dry_run: bool) -> list[str]:
    template_path = os.path.join(DATA_DIR, C1_TEMPLATE)
    header, rows = _parse_template(template_path)
    depot, custs = rows[0], rows[1:]
    T = float(depot[5])

    print(f'\n[C1] template={C1_TEMPLATE}  N={len(custs)}  T={T:.0f}  '
          f'depot=({depot[1]},{depot[2]})')
    centres = _cluster_centres(depot, custs, seed)
    print(f'  [clusters] arrival-time centres computed for {len(centres)}/{len(custs)} customers')
    print(f'{"name":<8} {"width":>6} {"mean_w":>7} {"min_w":>6}')
    print('-' * 34)

    made = []
    total = len(C1_WIDTHS) * C1_SEEDS_PER_WIDTH
    for idx in range(1, total + 1):
        width = C1_WIDTHS[(idx - 1) % len(C1_WIDTHS)]
        rng = np.random.default_rng(seed * 1000 + hash('C1') % 97 * 100 + idx)
        new_custs, _ = _make_tw_centred(depot, custs, centres, float(width), T, rng)
        name = f'CH{idx:02d}'
        w = np.array([c[5] - c[4] for c in new_custs], dtype=float)
        print(f'{name:<8} {width:>6} {w.mean():>7.1f} {w.min():>6.0f}')
        if not dry_run:
            _write_instance(os.path.join(out_dir, f'{name}.txt'), name, header, depot, new_custs)
        made.append(name)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=DATA_DIR)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    all_made = {}
    all_made['C1'] = generate_c1_cluster_tw(args.seed, args.out_dir, args.dry_run)
    for fam_name, cfg in _FAMILIES.items():
        made = generate_family(fam_name, cfg, args.seed, args.out_dir, args.dry_run)
        all_made[fam_name] = made

    print('\n' + '=' * 50)
    for fam_name, made in all_made.items():
        tag = '[dry-run] would write' if args.dry_run else 'wrote'
        print(f'[{fam_name}] {tag} {len(made)} instances')
        print(f'  train_instances += {made!r}')


if __name__ == '__main__':
    main()
