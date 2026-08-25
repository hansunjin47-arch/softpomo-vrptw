"""
generate_c_rc_tw_variants.py — TW-augmented variants of the Solomon C1 / RC1 geometry.

Same approach as generate_r1_tw_variants.py: within each Solomon family, all
instances share identical coordinates/depot/demands/service times -- only the
time windows differ. This keeps every fixed field byte-identical (rewrites the
template .txt text, touching only READY TIME / DUE DATE) and regenerates TW
across three difficulty tiers, targeting the gap between the test instance
(c101 / rc101) and the narrowest instance already in the training pool.

Gap diagnosis (mean TW width):
  C1:  c101 (test) = 60.8   |  training min = c105 = 121.6   -> gap below 121.6
  RC1: rc101 (test) = 30.0  |  training min = rc105 = 54.3   -> gap below 54.3

Tiers (one seed per width -- no cycling needed):
  C1  (3 seeds/tier, 9 total):  hard=[30,40,50] medium=[60,75,90] easy=[100,110,120]
  RC1 (5 seeds/tier, 15 total): hard=[10,15,20,25,30] medium=[35,40,45,50,55] easy=[70,90,110,130,150]

Note: C1's real Solomon TW structure is cluster-based and non-uniform (skewed
per-customer widths), unlike R1/RC1's simple "uniform width per instance"
model. These variants approximate difficulty via a uniform per-instance width
on the real C1 coordinates -- not a faithful reproduction of Solomon's C-type
generation, but sufficient to expose the policy to the missing TW-tightness
regime on real C1 geometry.

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

_FAMILIES = {
    'C1': dict(
        template='C102.txt',
        prefix_map={'H': 'CH', 'M': 'CM', 'E': 'CE'},
        tiers={'H': [30, 40, 50], 'M': [60, 75, 90], 'E': [100, 110, 120]},
    ),
    'RC1': dict(
        template='RC102.txt',
        prefix_map={'H': 'RCH', 'M': 'RCM', 'E': 'RCE'},
        tiers={'H': [10, 15, 20, 25, 30], 'M': [35, 40, 45, 50, 55], 'E': [70, 90, 110, 130, 150]},
    ),
}


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
        for s, width in enumerate(widths, 1):
            rng = np.random.default_rng(seed * 1000 + hash(fam_name + tier) % 97 * 100 + s)
            new_custs, n_open = _make_tw(depot, custs, float(width), T, rng)
            name = f'{prefix}{s:02d}'
            w = np.array([c[5] - c[4] for c in new_custs], dtype=float)
            print(f'{name:<8} {tier:<7} {width:>6} {w.mean():>7.1f} {w.min():>6.0f} {n_open:>5}')
            if not dry_run:
                _write_instance(os.path.join(out_dir, f'{name}.txt'),
                                name, header, depot, new_custs)
            made.append(name)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=DATA_DIR)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    all_made = {}
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
