"""LKH-3 sanity check on the generated C1/RC1 TW-variant instances."""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'clustering_routing'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from R_lkh import load_solomon, build_tt_matrix, solve_vrptw_lkh

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'Solomon')

TIME_LIMIT = 30

GROUPS = {
    'C1 reference':  ['c101', 'c105', 'c102'],
    'C1 generated':  [f'CH{i:02d}' for i in range(1, 4)] +
                      [f'CM{i:02d}' for i in range(1, 4)] +
                      [f'CE{i:02d}' for i in range(1, 4)],
    'RC1 reference': ['rc101', 'rc105', 'rc102'],
    'RC1 generated': [f'RCH{i:02d}' for i in range(1, 6)] +
                      [f'RCM{i:02d}' for i in range(1, 6)] +
                      [f'RCE{i:02d}' for i in range(1, 6)],
}


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
    all_bad = []
    for label, names in GROUPS.items():
        bad = run(names, label)
        all_bad += bad

    print('\n' + '=' * 58)
    if all_bad:
        print(f'[WARN] {len(all_bad)} instance(s) have late stops under LKH-3: {all_bad}')
    else:
        print('[OK] all generated instances solved with zero lateness.')
