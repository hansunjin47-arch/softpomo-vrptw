"""
build_rl_cluster_cache.py
RL best routes (_rl_cluster.json) 기반으로 LLM confidence 호출 → {name}_cluster.json 생성.

Flow:
  {name}_rl_cluster.json (routes as clusters, conf=1.0 placeholder)
      → LLM get_all_clusters_confidence(inst, clusters)
      → {name}_cluster.json (real TW/event-aware confidence scores)

Usage:
  python build_rl_cluster_cache.py --benchmark r1 --config F
  python build_rl_cluster_cache.py --benchmark c1 --config F
  python build_rl_cluster_cache.py --benchmark rc1 --config F
"""
import os, sys, json, argparse, time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from vrptw_env import load_solomon
from SoftClusterLLMModule import get_all_clusters_confidence

LLM_CACHE_DIR = os.path.join(_HERE, 'result_soft', 'llm_cache')
DATA_DIR      = os.path.join(_HERE, '..', 'data', 'Solomon')

BENCH_TRAIN = {
    'c1':  [f'c{i:03d}' for i in range(102, 110)],
    'rc1': [f'rc{i:03d}' for i in range(102, 109)],
    'r1':  [f'r{i:03d}' for i in range(102, 113)],
}


def parse_rl_cluster(path: str) -> list[list[int]]:
    """Load _rl_cluster.json → list[list[int]] (one list per vehicle)."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    keys = sorted(data.keys(), key=lambda x: int(x))
    return [[int(n) for n in data[k]] for k in keys]


def dump_cluster_cache(clusters: list, flat_conf: dict) -> dict:
    return {str(k): {str(n): round(flat_conf.get(n, 0.0), 2) for n in nodes}
            for k, nodes in enumerate(clusters)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', required=True, choices=['c1', 'rc1', 'r1'])
    parser.add_argument('--config',    default='F')
    parser.add_argument('--model',     default='qwen3:32b')
    parser.add_argument('--no-cot',    action='store_true')
    parser.add_argument('--cache-dir', default=LLM_CACHE_DIR)
    parser.add_argument('--force',     action='store_true',
                        help='Overwrite existing _cluster.json')
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    instances = BENCH_TRAIN[args.benchmark]

    print(f'[build_rl_cluster_cache] benchmark={args.benchmark}  '
          f'model={args.model}  config={args.config}')
    print(f'  cache_dir: {args.cache_dir}')

    for name in instances:
        rl_path  = os.path.join(args.cache_dir, f'{name.upper()}_rl_cluster.json')
        out_path = os.path.join(args.cache_dir, f'{name.upper()}_cluster.json')

        if not os.path.isfile(rl_path):
            print(f'  [{name}] _rl_cluster.json not found — run extract_rl_routes.py first')
            continue

        if os.path.isfile(out_path) and not args.force:
            print(f'  [{name}] _cluster.json already exists — skip (--force to overwrite)')
            continue

        sol_file = os.path.join(DATA_DIR, f'{name.upper()}.txt')
        if not os.path.isfile(sol_file):
            print(f'  [{name}] Solomon file not found: {sol_file}')
            continue

        inst     = load_solomon(sol_file)
        clusters = parse_rl_cluster(rl_path)
        K        = len(clusters)
        N        = inst['n_customers']
        print(f'  [{name}] {K} vehicles  N={N}  → LLM call ...', end='', flush=True)

        t0 = time.time()
        while True:
            try:
                flat_conf: dict[int, float] = get_all_clusters_confidence(
                    inst, clusters,
                    model=args.model,
                    use_cot=not args.no_cot,
                    use_ontology=True,
                    reward_config=args.config,
                )
                break
            except Exception as e:
                print(f'\n  [{name}] LLM failed ({e}), retry in 60s...')
                time.sleep(60)

        elapsed = time.time() - t0
        print(f' done ({elapsed:.0f}s)')

        cache = dump_cluster_cache(clusters, flat_conf)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f)
        print(f'  [{name}] saved → {out_path}')

    print('\n[Done] All instances processed.')
    print('Next: python train_soft_cluster.py --benchmark r1 --config F '
          '--resume <ckpt> --epochs 1000')


if __name__ == '__main__':
    main()
