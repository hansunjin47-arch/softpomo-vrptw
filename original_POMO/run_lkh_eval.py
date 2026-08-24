"""Run LKH-3 on Solomon benchmark instances and print K / dist / late results."""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'clustering_routing'))
from R_lkh import load_solomon, build_tt_matrix, solve_vrptw_lkh

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'Solomon')

GROUPS = {
    'C1':  ['c101','c102','c103','c104','c105','c106','c107','c108','c109'],
    'R1':  ['r101','r102','r103','r104','r105','r106','r107','r108','r109','r110','r111','r112'],
    'RC1': ['rc101','rc102','rc103','rc104','rc105','rc106','rc107','rc108'],
}

TIME_LIMIT = 60  # seconds per instance

for group, names in GROUPS.items():
    print(f'\n=== LKH-3: {group} ===')
    print(f'  {"Instance":<10} {"K":>4} {"Dist":>10} {"late_stops":>11} {"total_late":>12}')
    print(f'  {"-"*52}')
    for name in names:
        path = os.path.join(DATA_DIR, name + '.txt')
        inst = load_solomon(path)
        tt = build_tt_matrix(inst)
        r = solve_vrptw_lkh(inst, tt, time_limit_sec=TIME_LIMIT, runs=1)
        print(f'  {name:<10} {r["vehicles_used"]:>4} {r["total_distance"]:>10.2f} '
              f'{r["served_late"]:>11} {r.get("total_late", 0.0):>12.2f}')
    print()
