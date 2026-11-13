"""
compare_llm_vs_optimal.py

두 가지 분석:
  1. LLM cluster confidence ranking vs optimal solution visit order (Kendall tau)
  2. RL solution에서 cluster 밖 node 선택 비율 체크

Usage:
  python compare_llm_vs_optimal.py --benchmark c1
  python compare_llm_vs_optimal.py --benchmark rc1
  python compare_llm_vs_optimal.py --benchmark r1
  python compare_llm_vs_optimal.py --check-rl-routes   (RL 결과 분석)
"""
import json, sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from vrptw_env import load_solomon

CACHE_DIR  = os.path.join(os.path.dirname(__file__), '..', 'result_soft', 'llm_cache')
SOL_DIR    = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'Solomon')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'result')

BENCHMARKS = {
    'c1':  [f'c{i:03d}' for i in range(102, 110)],
    'rc1': [f'rc{i:03d}' for i in range(102, 109)],
    'r1':  [f'r{i:03d}' for i in range(102, 113)],
}


def load_sol(path):
    routes = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith('Route'):
                nodes = list(map(int, s.split(':')[1].split()))
                routes.append(nodes)
    return routes


def kendall_tau(a, b):
    shared = list(set(a) & set(b))
    if len(shared) <= 1:
        return None
    ra = {v: i for i, v in enumerate(a)}
    rb = {v: i for i, v in enumerate(b)}
    conc = disc = 0
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            u, v = shared[i], shared[j]
            if (ra[u] < ra[v]) == (rb[u] < rb[v]):
                conc += 1
            else:
                disc += 1
    total = conc + disc
    return (conc - disc) / total if total else 0.0


def analyze_llm_vs_optimal(instances):
    print(f'\n{"Instance":<12} {"Clusters":>8} {"FullMatch":>9} {"AvgTau":>8} {"MinTau":>8}')
    print('-' * 52)
    for name in instances:
        cache_path = os.path.join(CACHE_DIR, f'{name.upper()}_cluster.json')
        sol_path   = os.path.join(SOL_DIR,   f'{name}_sol.txt')
        if not os.path.exists(cache_path):
            print(f'{name:<12} {"(no cache)":>8}')
            continue
        if not os.path.exists(sol_path):
            print(f'{name:<12} {"(no sol)":>8}')
            continue

        with open(cache_path) as f:
            cdata = json.load(f)
        clusters   = cdata['clusters']
        confidence = cdata['confidence']
        routes     = load_sol(sol_path)

        taus = []
        full_match = 0
        for ci, cnodes in enumerate(clusters):
            cset = set(cnodes)
            best = max(routes, key=lambda r: len(set(r) & cset))
            overlap = list(set(best) & cset)
            if len(overlap) < 2:
                continue
            llm_order = sorted(overlap, key=lambda n: -confidence.get(str(n), 0))
            opt_order = [n for n in best if n in overlap]
            tau = kendall_tau(llm_order, opt_order)
            if tau is not None:
                taus.append(tau)
            if len(overlap) == len(cset) == len(best):
                full_match += 1

        avg_tau = sum(taus) / len(taus) if taus else float('nan')
        min_tau = min(taus) if taus else float('nan')
        print(f'{name:<12} {len(clusters):>8} {full_match:>9} {avg_tau:>8.3f} {min_tau:>8.3f}')


def check_rl_cluster_violations(benchmark, config='config_E'):
    """
    RL solution 파일에서 vehicle별 방문 노드와 LLM cluster 할당 비교.
    Cluster 밖 node를 선택한 비율 리포트.
    """
    bm_map = {'c1': 'c102-c109', 'rc1': 'rc102-rc108', 'r1': 'r102-r112'}
    test_map = {'c1': 'C101', 'rc1': 'RC101', 'r1': 'R101'}
    if benchmark not in bm_map:
        print(f'Unknown benchmark: {benchmark}')
        return

    result_subdir = bm_map[benchmark]
    test_prefix   = test_map[benchmark]

    plots_dir = os.path.join(RESULT_DIR, config, result_subdir, 'plots')
    cache_path = os.path.join(CACHE_DIR, f'{test_prefix}_cluster.json')

    if not os.path.exists(cache_path):
        print(f'No cache for {test_prefix}')
        return

    with open(cache_path) as f:
        cdata = json.load(f)
    clusters = cdata['clusters']
    node_to_cluster = {}
    for ci, nodes in enumerate(clusters):
        for n in nodes:
            node_to_cluster[n] = ci

    sol_files = [f for f in os.listdir(plots_dir) if f.endswith('_solution.txt')]
    print(f'\n=== RL Cluster Violation Check [{benchmark}] ===')
    print(f'{"Scenario":<20} {"Routes":>6} {"Nodes":>6} {"Violations":>10} {"ViolRate":>9}')
    print('-' * 56)

    for sf in sorted(sol_files):
        sol_path = os.path.join(plots_dir, sf)
        # Parse solution file for route assignments
        routes_found = []
        with open(sol_path) as f:
            cur_route = []
            for line in f:
                line = line.strip()
                if line.startswith('Route') or line.startswith('Vehicle'):
                    if cur_route:
                        routes_found.append(cur_route)
                    cur_route = []
                elif line and all(c.isdigit() or c in ' ,' for c in line):
                    try:
                        nodes = [int(x) for x in line.replace(',', ' ').split() if x.isdigit()]
                        cur_route.extend(nodes)
                    except:
                        pass
            if cur_route:
                routes_found.append(cur_route)

        if not routes_found:
            print(f'{sf:<20} (no routes parsed)')
            continue

        violations = 0
        total_nodes = 0
        for ri, route in enumerate(routes_found):
            expected_cluster = None
            for n in route:
                if n not in node_to_cluster:
                    continue
                total_nodes += 1
                nc = node_to_cluster[n]
                if expected_cluster is None:
                    expected_cluster = nc
                if nc != expected_cluster:
                    violations += 1

        rate = violations / total_nodes if total_nodes > 0 else 0
        scenario = sf.replace('_solution.txt', '')
        print(f'{scenario:<20} {len(routes_found):>6} {total_nodes:>6} {violations:>10} {rate:>9.1%}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', default='c1', choices=['c1', 'rc1', 'r1'])
    parser.add_argument('--check-rl-routes', action='store_true')
    args = parser.parse_args()

    if args.check_rl_routes:
        check_rl_cluster_violations(args.benchmark)
    else:
        instances = BENCHMARKS[args.benchmark]
        print(f'=== LLM vs Optimal [{args.benchmark}] ===')
        analyze_llm_vs_optimal(instances)
