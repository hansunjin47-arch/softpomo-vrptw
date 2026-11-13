"""LKH-3 evaluation of a full family's instance pool (base + generated TW
variants), saving BOTH the summary metrics (K/D/Lc/Lt) AND the actual routes
per instance to disk -- not just printing a table that scrolls away.

Output: soft_pomo/lkh_eval_results/{family}_full.json
  {
    "family": "c1",
    "time_limit": 30,
    "instances": [
      {"name": "c101", "tw_width": 60.8, "K": 10, "D": 829.0,
       "Lc": 0, "Lt": 0.0, "status": "feasible", "routes": [[0,3,5,...,0], ...]},
      ...
    ]
  }

Usage:
  python lkh_eval_full_save.py c1
  python lkh_eval_full_save.py r1
  python lkh_eval_full_save.py rc1
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'clustering_routing'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from R_lkh import load_solomon, build_tt_matrix, solve_vrptw_lkh

_HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(os.path.dirname(_HERE), 'data', 'Solomon')
OUT_DIR   = os.path.join(_HERE, 'lkh_eval_results')

FAMILIES = {
    'c1': dict(
        base=[f'c{i:03d}' for i in range(101, 110)],
        generated=[f'CH{i:02d}' for i in range(1, 4)] +
                  [f'CM{i:02d}' for i in range(1, 4)] +
                  [f'CE{i:02d}' for i in range(1, 4)],
    ),
    'r1': dict(
        base=[f'r{i:03d}' for i in range(101, 113)],
        generated=[f'RH{i:02d}' for i in range(1, 11)] +
                  [f'RM{i:02d}' for i in range(1, 11)] +
                  [f'RE{i:02d}' for i in range(1, 11)],
    ),
    'rc1': dict(
        base=[f'rc{i:03d}' for i in range(101, 109)],
        generated=[f'RCH{i:02d}' for i in range(1, 6)] +
                  [f'RCM{i:02d}' for i in range(1, 6)] +
                  [f'RCE{i:02d}' for i in range(1, 6)],
    ),
}


def eval_one(name: str, time_limit: int) -> dict | None:
    path = os.path.join(DATA_DIR, name + '.txt')
    if not os.path.isfile(path):
        print(f'  {name:<10} MISSING')
        return None
    inst = load_solomon(path)
    tw = [c - o for o, c in zip(inst['tw_open'][1:], inst['tw_close'][1:])]
    w = sum(tw) / len(tw)
    tt = build_tt_matrix(inst)
    r = solve_vrptw_lkh(inst, tt, time_limit_sec=time_limit, runs=1)
    Lc, Lt = r['served_late'], r.get('total_late', 0.0)
    print(f'  {name:<10} {w:>8.1f} {r["vehicles_used"]:>4} '
          f'{r["total_distance"]:>10.2f} {Lc:>4} {Lt:>10.2f}  [{r["status"]}]', flush=True)
    return dict(name=name, tw_width=round(w, 2), K=r['vehicles_used'],
                D=round(r['total_distance'], 2), Lc=Lc, Lt=round(Lt, 2),
                status=r['status'], routes=r['routes'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('family', choices=list(FAMILIES))
    ap.add_argument('--time-limit', type=int, default=30)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = FAMILIES[args.family]
    all_results = []

    for label, names in (('base', cfg['base']), ('generated', cfg['generated'])):
        print(f'\n=== {args.family.upper()} {label}  (LKH-3, {args.time_limit}s/instance) ===')
        print(f'  {"Instance":<10} {"TWwidth":>8} {"K":>4} {"D":>10} {"Lc":>4} {"Lt":>10}')
        print('  ' + '-' * 52)
        for name in names:
            r = eval_one(name, args.time_limit)
            if r is not None:
                r['pool'] = label
                all_results.append(r)

    out_path = os.path.join(OUT_DIR, f'{args.family}_full.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(dict(family=args.family, time_limit=args.time_limit,
                       instances=all_results), f, indent=2)

    bad = [r for r in all_results if r['Lc'] > 0]
    print('\n' + '=' * 60)
    print(f'[Saved] {out_path}  ({len(all_results)} instances, routes included)')
    if bad:
        print(f'[WARN] {len(bad)} instance(s) have Lc>0: '
              f'{[(r["name"], r["Lc"]) for r in bad]}')
    else:
        print(f'[OK] all {len(all_results)} {args.family.upper()} instances solved with Lc=0.')


if __name__ == '__main__':
    main()
