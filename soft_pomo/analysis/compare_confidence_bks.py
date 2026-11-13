"""
compare_confidence_bks.py
Compares Qwen vs DeepSeek LLM confidence with BKS visit order.

For each base instance (C101-C109, R101-R112, RC101-RC108):
  1. Cluster purity: what fraction of BKS route nodes land in the same Kim-cluster?
  2. Rank correlation: within shared (cluster, BKS-route) groups,
     does confidence rank correlate with BKS visit position?
  3. Confidence statistics: mean, std, discriminability.

Usage:
  python analysis/compare_confidence_bks.py --benchmark c1
  python analysis/compare_confidence_bks.py --benchmark r1
  python analysis/compare_confidence_bks.py --benchmark rc1
"""
from __future__ import annotations
import argparse
import json
import math
import os
import re

HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.join(HERE, '..', '..')
DATA  = os.path.join(ROOT, 'data', 'Solomon')

QWEN_DIR     = os.path.join(HERE, '..', 'result_soft',          'llm_cache')
DEEPSEEK_DIR = os.path.join(HERE, '..', 'result_soft_deepseek', 'llm_cache')

BENCHMARKS = {
    'c1':  [f'c{i:03d}' for i in range(101, 110)],
    'r1':  [f'r{i:03d}' for i in range(101, 113)],
    'rc1': [f'rc{i:03d}' for i in range(101, 109)],
}


# ── BKS loader ──────────────────────────────────────────────────────────────

def load_bks_routes(name: str) -> list[list[int]] | None:
    """Returns list of routes; each route is a list of customer node IDs (1-indexed)."""
    path = os.path.join(DATA, f'{name}_sol.txt')
    if not os.path.isfile(path):
        return None
    routes = []
    with open(path, encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln.lower().startswith('route'):
                continue
            colon = ln.find(':')
            if colon < 0:
                continue
            nodes = [int(x) for x in ln[colon+1:].split() if x.isdigit()]
            if nodes:
                routes.append(nodes)
    return routes or None


# ── Cluster cache loader ─────────────────────────────────────────────────────

def load_cluster_cache(cache_dir: str, name: str) -> dict[int, dict[int, float]] | None:
    """Returns {cluster_idx: {node: confidence}}."""
    path = os.path.join(cache_dir, f'{name.upper()}_cluster.json')
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    return {int(k): {int(n): float(v) for n, v in nodes.items()}
            for k, nodes in raw.items()}


# ── Metrics ──────────────────────────────────────────────────────────────────

def node_to_cluster(cache: dict[int, dict[int, float]]) -> dict[int, int]:
    n2c = {}
    for c_idx, nodes in cache.items():
        for n in nodes:
            n2c[n] = c_idx
    return n2c


def cluster_purity(bks_routes: list[list[int]],
                   n2c: dict[int, int]) -> float:
    """
    For each BKS route, find the plurality cluster.
    Purity = (nodes in plurality cluster) / total nodes, averaged over routes.
    """
    purities = []
    for route in bks_routes:
        known = [n for n in route if n in n2c]
        if not known:
            continue
        from collections import Counter
        counts = Counter(n2c[n] for n in known)
        majority = counts.most_common(1)[0][1]
        purities.append(majority / len(known))
    return sum(purities) / len(purities) if purities else float('nan')


def kendall_tau(xs: list[float], ys: list[float]) -> float:
    """Kendall's τ-b between two ranking lists (ties handled)."""
    n = len(xs)
    if n < 2:
        return float('nan')
    concordant = discordant = tie_x = tie_y = tie_both = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 and dy == 0:
                tie_both += 1
            elif dx == 0:
                tie_x += 1
            elif dy == 0:
                tie_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    n0 = n * (n - 1) // 2
    denom = math.sqrt((n0 - tie_x - tie_both) * (n0 - tie_y - tie_both))
    return (concordant - discordant) / denom if denom else float('nan')


def rank_metrics_vs_bks(bks_routes: list[list[int]],
                        cache: dict[int, dict[int, float]],
                        n2c: dict[int, int]) -> dict:
    """
    For each BKS route, find its plurality cluster.
    Among nodes in (route ∩ plurality_cluster), sort by confidence desc
    → treat that as "predicted visit order".
    BKS position in the route → "true visit order".

    Metrics:
      mae        : mean |predicted_rank - bks_rank| per node (in # of positions)
      exact_rate : fraction of nodes where predicted rank == bks rank exactly
      tau        : Kendall-τ (confidence desc rank vs bks position asc)
      n_nodes    : total nodes used
    """
    from collections import Counter

    all_mae   = []
    all_exact = []
    all_tau   = []

    for route in bks_routes:
        known = [(bks_pos, n, n2c[n]) for bks_pos, n in enumerate(route) if n in n2c]
        if not known:
            continue
        counts = Counter(c for _, _, c in known)
        plurality_c = counts.most_common(1)[0][0]
        subset = [(bks_pos, n) for bks_pos, n, c in known if c == plurality_c]
        if len(subset) < 2:
            continue

        bks_positions  = [bks_pos for bks_pos, _ in subset]
        confidences    = [cache[plurality_c][n] for _, n in subset]

        # predicted rank: sort by confidence descending → rank 0 = highest conf = visit first
        sorted_by_conf = sorted(range(len(subset)), key=lambda i: confidences[i], reverse=True)
        pred_rank = [0] * len(subset)
        for pred_pos, orig_i in enumerate(sorted_by_conf):
            pred_rank[orig_i] = pred_pos

        # bks rank: renumber subset positions to 0..k-1
        bks_rank_sorted = sorted(range(len(subset)), key=lambda i: bks_positions[i])
        bks_rank = [0] * len(subset)
        for rank_pos, orig_i in enumerate(bks_rank_sorted):
            bks_rank[orig_i] = rank_pos

        diffs = [abs(pred_rank[i] - bks_rank[i]) for i in range(len(subset))]
        all_mae.extend(diffs)
        all_exact.extend([1 if d == 0 else 0 for d in diffs])
        all_tau.append(kendall_tau(confidences, [-p for p in bks_positions]))

    n_nodes    = len(all_mae)
    mae        = sum(all_mae) / n_nodes if n_nodes else float('nan')
    exact_rate = sum(all_exact) / n_nodes if n_nodes else float('nan')
    valid_tau  = [t for t in all_tau if not math.isnan(t)]
    tau        = sum(valid_tau) / len(valid_tau) if valid_tau else float('nan')
    return {'mae': mae, 'exact_rate': exact_rate, 'tau': tau, 'n_nodes': n_nodes}


def confidence_stats(cache: dict[int, dict[int, float]]) -> dict:
    all_vals = [v for nodes in cache.values() for v in nodes.values()]
    if not all_vals:
        return {}
    mean = sum(all_vals) / len(all_vals)
    var  = sum((v - mean) ** 2 for v in all_vals) / len(all_vals)
    return {
        'min': min(all_vals),
        'max': max(all_vals),
        'mean': mean,
        'std': math.sqrt(var),
        'range': max(all_vals) - min(all_vals),
        'n': len(all_vals),
    }


# ── Per-instance analysis ────────────────────────────────────────────────────

def analyse_instance(name: str) -> dict | None:
    bks = load_bks_routes(name)
    if bks is None:
        return None
    qwen_c = load_cluster_cache(QWEN_DIR, name)
    deep_c = load_cluster_cache(DEEPSEEK_DIR, name)
    if qwen_c is None and deep_c is None:
        return None

    result = {'name': name, 'bks_routes': len(bks),
              'bks_nodes': sum(len(r) for r in bks)}

    for label, cache in [('qwen', qwen_c), ('deepseek', deep_c)]:
        if cache is None:
            result[label] = None
            continue
        n2c   = node_to_cluster(cache)
        stats = confidence_stats(cache)
        purity = cluster_purity(bks, n2c)
        rank   = rank_metrics_vs_bks(bks, cache, n2c)
        result[label] = {
            'purity': purity,
            **rank,
            **stats,
        }
    return result


# ── Pretty print ─────────────────────────────────────────────────────────────

def fmt(v, fmt_str='.3f'):
    return f'{v:{fmt_str}}' if isinstance(v, float) and not math.isnan(v) else 'N/A'


def print_results(results: list[dict]):
    # Table 1: BKS alignment metrics
    h1 = (f"{'Instance':<10} "
          f"{'Q_purity':>8} {'Q_MAE':>6} {'Q_exact%':>8}  "
          f"{'D_purity':>8} {'D_MAE':>6} {'D_exact%':>8}")
    print('── BKS alignment ──────────────────────────────────────────────')
    print(h1)
    print('-' * len(h1))

    def agg(key, sub):
        vals = [r[sub][key] for r in results
                if r.get(sub) and not math.isnan(r[sub].get(key, float('nan')))]
        return sum(vals) / len(vals) if vals else float('nan')

    for r in results:
        q = r.get('qwen') or {}
        d = r.get('deepseek') or {}
        q_exact = q.get('exact_rate', float('nan'))
        d_exact = d.get('exact_rate', float('nan'))
        print(f"{r['name'].upper():<10} "
              f"{fmt(q.get('purity',     float('nan'))):>8} "
              f"{fmt(q.get('mae',        float('nan'))):>6} "
              f"{fmt(q_exact*100 if not math.isnan(q_exact) else float('nan'), '.1f'):>8}  "
              f"{fmt(d.get('purity',     float('nan'))):>8} "
              f"{fmt(d.get('mae',        float('nan'))):>6} "
              f"{fmt(d_exact*100 if not math.isnan(d_exact) else float('nan'), '.1f'):>8}")
    print('-' * len(h1))
    q_exact_avg = agg('exact_rate', 'qwen')
    d_exact_avg = agg('exact_rate', 'deepseek')
    print(f"{'AVG':<10} "
          f"{fmt(agg('purity','qwen')):>8} {fmt(agg('mae','qwen')):>6} "
          f"{fmt(q_exact_avg*100 if not math.isnan(q_exact_avg) else float('nan'), '.1f'):>8}  "
          f"{fmt(agg('purity','deepseek')):>8} {fmt(agg('mae','deepseek')):>6} "
          f"{fmt(d_exact_avg*100 if not math.isnan(d_exact_avg) else float('nan'), '.1f'):>8}")
    print()
    print('  purity : BKS-route nodes in same Kim-cluster (mean over routes)')
    print('  MAE    : mean absolute rank diff per node vs BKS visit order (lower=better)')
    print('  exact% : % nodes where conf-rank == BKS-rank exactly')

    # Table 2: discriminability
    h2 = (f"\n── Confidence discriminability ────────────────────────────────\n"
          f"{'Instance':<10} {'Q_range':>7} {'Q_std':>6}  {'D_range':>7} {'D_std':>6}")
    print(h2)
    print('-' * 55)
    for r in results:
        q = r.get('qwen') or {}
        d = r.get('deepseek') or {}
        print(f"{r['name'].upper():<10} "
              f"{fmt(q.get('range', float('nan'))):>7} {fmt(q.get('std', float('nan'))):>6}  "
              f"{fmt(d.get('range', float('nan'))):>7} {fmt(d.get('std', float('nan'))):>6}")
    print('-' * 55)
    print(f"{'AVG':<10} "
          f"{fmt(agg('range','qwen')):>7} {fmt(agg('std','qwen')):>6}  "
          f"{fmt(agg('range','deepseek')):>7} {fmt(agg('std','deepseek')):>6}")
    print()
    print('  range/std : higher = more spread confidence → stronger POMO soft-guidance')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', default='c1', choices=list(BENCHMARKS))
    args = ap.parse_args()

    instances = BENCHMARKS[args.benchmark]
    print(f'\n=== Qwen vs DeepSeek — confidence × BKS alignment ({args.benchmark.upper()}) ===\n')

    results = []
    for name in instances:
        r = analyse_instance(name)
        if r:
            results.append(r)
        else:
            print(f'  [skip] {name}: no BKS or both caches missing')

    if results:
        print_results(results)
    else:
        print('No data to compare.')


if __name__ == '__main__':
    main()
