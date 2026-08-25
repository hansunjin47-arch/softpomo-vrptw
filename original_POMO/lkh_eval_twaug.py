"""LKH-3 sanity check on the generated TW-variant instances.

If the generated instances are well-formed, LKH-3 should serve all 100 customers
on time (late_stops = 0) with a vehicle count that scales sensibly with TW width.
Solomon r101/r102/r105 are included as reference points.
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

REFERENCE = ['r101', 'r105', 'r102']
GENERATED = ([f'RH{i:02d}' for i in range(1, 11)] +
             [f'RM{i:02d}' for i in range(1, 11)] +
             [f'RE{i:02d}' for i in range(1, 11)])


def run(names, label):
    print(f'\n=== {label}  (LKH-3, {TIME_LIMIT}s/instance) ===')
    print(f'  {"Instance":<10} {"TWwidth":>8} {"K":>4} {"Dist":>10} '
          f'{"late":>6} {"total_late":>11}')
    print('  ' + '-' * 54)
    bad = []
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
        late = r['served_late']
        print(f'  {name:<10} {w:>8.1f} {r["vehicles_used"]:>4} '
              f'{r["total_distance"]:>10.2f} {late:>6} '
              f'{r.get("total_late", 0.0):>11.2f}', flush=True)
        if late > 0:
            bad.append((name, late))
    return bad


if __name__ == '__main__':
    run(REFERENCE, 'Solomon reference')
    bad = run(GENERATED, 'Generated TW variants')

    print('\n' + '=' * 58)
    if bad:
        print(f'[WARN] {len(bad)}/{len(GENERATED)} generated instances have late stops '
              f'under LKH-3: {bad}')
        print('       Tight-TW instances may legitimately be infeasible — check whether '
              'K hit the vehicle limit (25).')
    else:
        print(f'[OK] all {len(GENERATED)} generated instances solved with zero lateness.')
