"""
generate_r1_tw_variants.py — TW-augmented variants of the Solomon R1 geometry.

All Solomon R1 instances (r101-r112) share identical coordinates, depot, demands
and service times; only the time windows differ.  This script keeps every one of
those fixed fields byte-identical (it rewrites the R102.txt text, touching only
the READY TIME / DUE DATE columns).

Verified gap (constrained-customer width median + density = fraction of customers
with width <= 0.5T, computed directly from r101-r112):

    inst   width_con_med   density   K (LKH-3)
    r101*      10             100%      18      <- TEST
    r102       10              75%      18
    r105       30             100%      16
    r109       59             100%      13

r101's own width (10) is NOT below the training range -- r102 has the exact same
constrained width. The real gap is density: r102 shares r101's width but only
75% of customers are constrained (25 are free, width=T), so it's easier. Every
OTHER width level in the family that reaches 100% density (r105 width=30, r109
width=59) is wider than r101. The (narrowest-width, 100%-density) cell is
occupied ONLY by r101 itself and has no training-pool peer -- by original Solomon
design (the "x01" instance in each family is conventionally the held-out test
point), not by accident.

These variants target exactly that cell: width jittered tightly around r101's
own width (8-12), density left at ~100% (every customer gets a real window --
this generator has no unconstrained-customer mechanism, matching r101/r105/r109's
own 100%-density character).

TW placement follows Solomon (1987): centre ~ Uniform(t_depot->i, T - t_i->depot - svc_i).

Usage:
  python generate_r1_tw_variants.py                    # 15 instances -> data/Solomon
  python generate_r1_tw_variants.py --dry-run
"""
from __future__ import annotations

import argparse
import math
import os
import re

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA_DIR = os.path.join(_ROOT, 'data', 'Solomon')
TEMPLATE = os.path.join(DATA_DIR, 'R102.txt')

# Width jittered tightly around r101's own width (10) -- see module docstring
# for why this specific cell (not a wider spread) is the verified gap.
_TIERS: dict[str, list[int]] = {
    'H': [8, 9, 10, 11, 12],
}
_SEEDS_PER_WIDTH = 3   # 5 widths x 3 seeds = 15 instances


def _parse_template(path: str):
    """Return (header_lines, rows, depot_row) from a Solomon .txt.

    rows: list of [cust_no, x, y, demand, ready, due, service] as ints.
    """
    with open(path, encoding='utf-8') as f:
        lines = f.read().splitlines()

    hdr_idx = next(i for i, ln in enumerate(lines)
                   if ln.strip().upper().startswith('CUST NO'))
    header = lines[:hdr_idx + 2]          # through the blank spacer line

    rows = []
    for ln in lines[hdr_idx + 2:]:
        parts = ln.split()
        if len(parts) != 7:
            continue
        rows.append([int(float(p)) for p in parts])
    return header, rows


def _write_instance(path: str, name: str, header, depot, custs):
    """Write Solomon .txt with the template header, name replaced."""
    out = list(header)
    out[0] = name.upper()
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')
        for r in [depot] + custs:
            f.write(f"  {r[0]:4d}  {r[1]:7d}  {r[2]:7d}  {r[3]:9d}"
                    f"  {r[4]:11d}  {r[5]:10d}  {r[6]:9d}\n")


def _make_tw(depot, custs, width: float, T: float, rng) -> list:
    """Return customer rows with READY/DUE replaced by a fixed-width window."""
    dx, dy = depot[1], depot[2]
    out = []
    n_unconstrained = 0
    for c in custs:
        cid, x, y, dem, _, _, svc = c
        d = math.hypot(x - dx, y - dy)          # speed = 1  ->  time == distance
        lo = d
        hi = T - d - svc
        if hi <= lo:
            # Customer cannot be served within the horizon -> leave window open
            ready, due = 0, int(T)
            n_unconstrained += 1
        else:
            centre = float(rng.uniform(lo, hi))
            ready_f = max(lo, centre - width / 2.0)
            due_f = min(T, ready_f + width)
            ready_f = max(0.0, due_f - width)   # re-align if clipped at the top
            ready, due = int(round(ready_f)), int(round(due_f))
            if due <= ready:
                due = min(int(T), ready + max(1, int(round(width))))
        out.append([cid, x, y, dem, ready, due, svc])
    return out, n_unconstrained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds-per-width', type=int, default=_SEEDS_PER_WIDTH,
                    help=f'Independent draws per width level (default: {_SEEDS_PER_WIDTH})')
    ap.add_argument('--template', default=TEMPLATE)
    ap.add_argument('--out-dir', default=DATA_DIR)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    header, rows = _parse_template(args.template)
    depot, custs = rows[0], rows[1:]
    T = float(depot[5])          # depot DUE DATE == time horizon

    print(f'[template] {os.path.basename(args.template)}  '
          f'N={len(custs)}  T={T:.0f}  depot=({depot[1]},{depot[2]})')
    print(f'[fixed]    coords, demand, service, depot, T, capacity, vehicle_limit')
    print(f'[varied]   READY TIME / DUE DATE only\n')

    print(f'{"name":<8} {"tier":<7} {"width":>6} {"mean_w":>7} {"min_w":>6} {"open":>5}')
    print('-' * 46)

    made = []
    for tier, widths in _TIERS.items():
        total = len(widths) * args.seeds_per_width
        for idx in range(1, total + 1):
            width = widths[(idx - 1) % len(widths)]
            rng = np.random.default_rng(args.seed * 1000 + hash(tier) % 97 * 100 + idx)
            new_custs, n_open = _make_tw(depot, custs, float(width), T, rng)

            name = f'R{tier}{idx:02d}'
            w = np.array([c[5] - c[4] for c in new_custs], dtype=float)
            print(f'{name:<8} {tier:<7} {width:>6} {w.mean():>7.1f} {w.min():>6.0f} {n_open:>5}')

            if not args.dry_run:
                _write_instance(os.path.join(args.out_dir, f'{name}.txt'),
                                name, header, depot, new_custs)
            made.append(name)

    print('-' * 46)
    if args.dry_run:
        print(f'[dry-run] would write {len(made)} files to {args.out_dir}')
    else:
        print(f'[done] wrote {len(made)} instances to {args.out_dir}')
    print('\ntrain_instances = ' + repr(made))


if __name__ == '__main__':
    main()
