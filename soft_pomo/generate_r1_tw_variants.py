"""
generate_r1_tw_variants.py — TW-augmented variants of the Solomon R1 geometry.

All Solomon R1 instances (r101-r112) share identical coordinates, depot, demands
and service times; only the time windows differ.  This script keeps every one of
those fixed fields byte-identical (it rewrites the R102.txt text, touching only
the READY TIME / DUE DATE columns) and regenerates the time windows across three
difficulty tiers:

    hard   : TW width  8-15   -- the r101 regime (width 10), absent from r102-r112
    medium : TW width 20-35   -- the r105 regime (width 30)
    easy   : TW width 45-70   -- the r102/r106 regime

Motivation: r102-r112 cover TW widths [30, 148]; r101 sits at 10 and is therefore
outside the training distribution.  These variants fill that gap without using the
test instance itself.

TW placement follows Solomon (1987): centre ~ Uniform(t_depot->i, T - t_i->depot - svc_i),
then a fixed width around it.  r101 and r105 both use a constant width for every
customer, which is what the hard/medium tiers reproduce.

Usage:
  python generate_r1_tw_variants.py                    # 30 instances -> data/Solomon
  python generate_r1_tw_variants.py --seeds 5          # 15 instances
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

# Widths sampled per tier; index cycles with the seed.
_TIERS: dict[str, list[int]] = {
    'H': [8, 10, 12, 15],       # r101 regime
    'M': [20, 25, 30, 35],      # r105 regime
    'E': [45, 55, 70],          # r102 / r106 regime
}


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
    ap.add_argument('--seeds', type=int, default=10,
                    help='Instances per tier (default: 10 -> 30 total)')
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
        for s in range(1, args.seeds + 1):
            width = widths[(s - 1) % len(widths)]
            rng = np.random.default_rng(args.seed * 1000 + hash(tier) % 97 * 100 + s)
            new_custs, n_open = _make_tw(depot, custs, float(width), T, rng)

            name = f'R{tier}{s:02d}'
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
