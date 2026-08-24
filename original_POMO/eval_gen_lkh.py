"""
eval_gen_lkh.py
각 타입(C/R/RC) 데이터셋에서 첫 번째 인스턴스 n개를 LKH-3으로 풀어 검증.

Usage:
    python eval_gen_lkh.py --n 3
"""
import os, sys, argparse, math
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'clustering_routing'))

from R_lkh import build_tt_matrix, solve_vrptw_lkh

DATASETS = {
    'C':  os.path.join(_HERE, 'data', 'train_C_50k.pt'),
    'R':  os.path.join(_HERE, 'data', 'train_R_50k.pt'),
    'RC': os.path.join(_HERE, 'data', 'train_RC_50k.pt'),
}


def gen_inst_to_lkh(inst: dict) -> dict:
    """텐서 정규화 포맷(generate_vrptw._package) → R_lkh.py load_solomon 포맷."""
    T     = float(inst['T'])
    cap   = float(inst['vehicle_capacity'])
    max_c = float(inst.get('max_coord', 100.0))

    depot_xy = inst['depot_xy'].numpy()[0] * max_c    # [2]
    node_xy  = inst['node_xy'].numpy()  * max_c      # [N, 2]

    coords = [(float(depot_xy[0]), float(depot_xy[1]))]
    coords += [(float(x), float(y)) for x, y in node_xy]

    demands = [0.0] + [float(d) * cap for d in inst['node_demand'].numpy()]
    tw_open  = [0.0] + [float(o) * T for o in inst['node_tw_open'].numpy()]
    tw_close = [T]   + [float(c) * T for c in inst['node_tw_close'].numpy()]
    service  = [0.0] + [float(s) * T for s in inst['node_service'].numpy()]

    return {
        'name':             inst.get('name', 'rand'),
        'coords':           coords,
        'demands':          demands,
        'tw_open':          tw_open,
        'tw_close':         tw_close,
        'service_time':     service,
        'vehicle_limit':    int(inst.get('vehicle_limit', 25)),
        'vehicle_capacity': cap,
        'n_customers':      int(inst['n_customers']),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=3,
                        help='Number of instances per type to solve')
    parser.add_argument('--time-limit', type=int, default=30)
    args = parser.parse_args()

    print(f"LKH-3 validation: first {args.n} instances per type  "
          f"(time_limit={args.time_limit}s)")
    print("=" * 70)

    for typ, path in DATASETS.items():
        if not os.path.isfile(path):
            print(f"  [{typ}] NOT FOUND: {path}")
            continue

        data = torch.load(path, weights_only=False)
        print(f"\n[{typ}]  dataset size = {len(data)}")
        print(f"  {'Name':<24} {'K':>4} {'OnTime':>8} {'Late':>5} {'Dist':>10} {'T(s)':>6}")
        print("  " + "-" * 60)

        for idx in range(min(args.n, len(data))):
            raw  = data[idx]
            inst = gen_inst_to_lkh(raw)
            tt   = build_tt_matrix(inst)

            result = solve_vrptw_lkh(inst, tt, time_limit_sec=args.time_limit, runs=1)
            n      = inst['n_customers']
            print(f"  {inst['name']:<24}"
                  f" {result['vehicles_used']:>4}"
                  f" {result['served_on_time']:>4}/{n:<3}"
                  f" {result['served_late']:>5}"
                  f" {result['total_distance']:>10.2f}"
                  f" {result['elapsed']:>5.1f}s")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == '__main__':
    main()
