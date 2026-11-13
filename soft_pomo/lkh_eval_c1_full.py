"""LKH-3 evaluation of the full C1 training pool: base c101-c109 (9) plus the
generated TW-width variants CH01-03/CM01-03/CE01-03 (9) = 18 instances.

Reports the standard metrics used throughout this project: K (vehicles),
D (distance), Lc (late count), Lt (total late minutes).
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'clustering_routing'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from R_lkh import load_solomon, build_tt_matrix, solve_vrptw_lkh

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'Solomon')

TIME_LIMIT = 30

BASE      = [f'c{i:03d}' for i in range(101, 110)]
GENERATED = [f'CH{i:02d}' for i in range(1, 4)] + \
            [f'CM{i:02d}' for i in range(1, 4)] + \
            [f'CE{i:02d}' for i in range(1, 4)]


def run(names, label):
    print(f'\n=== {label}  (LKH-3, {TIME_LIMIT}s/instance) ===')
    print(f'  {"Instance":<10} {"TWwidth":>8} {"K":>4} {"D":>10} {"Lc":>4} {"Lt":>10}')
    print('  ' + '-' * 52)
    rows = []
    for name in names:
        path = os.path.join(DATA_DIR, name + '.txt')
        if not os.path.isfile(path):
            print(f'  {name:<10} MISSING')
            continue
        inst = load_solomon(path)
        tw = [c - o for o, c in zip(inst['tw_open'][1:], inst['tw_close'][1:])]
        w = sum(tw) / len(tw)
        tt = build_tt_matrix(inst)
        r = solve_vrptw_lkh(inst, tt, time_limit_sec=TIME_LIMIT, runs=1)
        Lc = r['served_late']
        Lt = r.get('total_late', 0.0)
        print(f'  {name:<10} {w:>8.1f} {r["vehicles_used"]:>4} '
              f'{r["total_distance"]:>10.2f} {Lc:>4} {Lt:>10.2f}', flush=True)
        rows.append((name, w, r['vehicles_used'], r['total_distance'], Lc, Lt))
    return rows


if __name__ == '__main__':
    base_rows = run(BASE, 'C1 base (c101-c109)')
    gen_rows  = run(GENERATED, 'C1 generated TW variants')

    bad = [r for r in (base_rows + gen_rows) if r[4] > 0]
    print('\n' + '=' * 60)
    if bad:
        print(f'[WARN] {len(bad)} instance(s) have Lc>0 under LKH-3: '
              f'{[(n, lc) for n, _, _, _, lc, _ in bad]}')
    else:
        print('[OK] all 18 C1 instances solved with zero lateness (Lc=0).')
