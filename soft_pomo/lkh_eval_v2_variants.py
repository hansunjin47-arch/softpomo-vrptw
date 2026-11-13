"""LKH-3 validation of the corrected, gap-targeted TW variants (15 each)."""
import os, sys
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'clustering_routing'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from R_lkh import load_solomon, build_tt_matrix, solve_vrptw_lkh

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'Solomon')
TIME_LIMIT = 30

GROUPS = {
    'C1 (target K~10, width~61)':   [f'CH{i:02d}'  for i in range(1, 16)],
    'RC1 (target K~15, width~30)':  [f'RCH{i:02d}' for i in range(1, 16)],
    'R1 (target K~18, width~10)':   [f'RH{i:02d}'  for i in range(1, 16)],
}
REF = {'C1 (target K~10, width~61)': ('c101', 10), 'RC1 (target K~15, width~30)': ('rc101', 15),
       'R1 (target K~18, width~10)': ('r101', 18)}

for label, names in GROUPS.items():
    ref_name, ref_K = REF[label]
    print(f'\n=== {label}  (LKH-3, {TIME_LIMIT}s/instance, {ref_name} K={ref_K}) ===')
    print(f'  {"Instance":<10} {"width":>6} {"K":>4} {"D":>10} {"Lc":>4} {"Lt":>9}')
    print('  ' + '-' * 48)
    for name in names:
        path = os.path.join(DATA_DIR, name + '.txt')
        inst = load_solomon(path)
        tw = [c - o for o, c in zip(inst['tw_open'][1:], inst['tw_close'][1:])]
        w = sum(tw) / len(tw)
        tt = build_tt_matrix(inst)
        r = solve_vrptw_lkh(inst, tt, time_limit_sec=TIME_LIMIT, runs=1)
        print(f'  {name:<10} {w:>6.1f} {r["vehicles_used"]:>4} '
              f'{r["total_distance"]:>10.2f} {r["served_late"]:>4} '
              f'{r.get("total_late",0.0):>9.2f}', flush=True)
