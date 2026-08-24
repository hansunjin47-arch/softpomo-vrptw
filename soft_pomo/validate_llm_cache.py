"""validate_llm_cache.py — Validate LLM cache files; delete invalid ones for retry.

Usage:
    python validate_llm_cache.py                    # report only
    python validate_llm_cache.py --delete           # delete invalid/missing caches
    python validate_llm_cache.py --filter c1        # only check C1xx instances
    python validate_llm_cache.py --benchmark c1     # check expected vs actual for benchmark
    python validate_llm_cache.py --benchmark rc1 --delete  # delete bad caches for RC1

After --delete, simply re-run training — it will only re-call LLM for deleted/missing caches.
"""
from __future__ import annotations

import os
import json
import argparse
import re

SOFT      = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SOFT, 'result_soft', 'llm_cache')

# Expected instance sets per benchmark (base + rain + acc instances)
_BENCHMARK_BASE = {
    'c1':  [f'C{i:03d}' for i in range(102, 110)],   # C102-C109
    'rc1': [f'RC{i:d}'  for i in range(102, 109)],   # RC102-RC108
    'r1':  [f'R{i:d}'   for i in range(102, 113)],   # R102-R112 (approx)
}
_SUFFIXES = ['', '_RAIN_A', '_RAIN_B', '_ACC_A', '_ACC_B']


def expected_instances(benchmark: str) -> list[str]:
    bases = _BENCHMARK_BASE.get(benchmark.lower(), [])
    return [b + s for b in bases for s in _SUFFIXES]


def validate(path: str) -> tuple[bool, str]:
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return False, f"parse error: {e}"

    clusters = data.get('clusters', [])
    if not clusters:
        return False, "empty clusters"

    conf: dict = data.get('confidence', {})
    if not conf:
        return False, "no confidence data"

    for k_str, node_conf in conf.items():
        if not node_conf:
            return False, f"cluster {k_str}: empty"
        if isinstance(node_conf, dict) and all(float(v) == 0.0 for v in node_conf.values()):
            return False, f"cluster {k_str}: all-zero confidence"

    # ACC instances reuse base clusters — confidence count may differ from cluster count
    return True, f"{len(clusters)} clusters, {len(conf)} conf entries"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--delete',    action='store_true', help='Delete invalid cache files')
    ap.add_argument('--filter',    default='', help='Only files whose name contains this string')
    ap.add_argument('--benchmark', default='', help='c1 / rc1 / r1 — check expected set')
    args = ap.parse_args()

    if not os.path.isdir(CACHE_DIR):
        print(f'[ERROR] Cache dir not found: {CACHE_DIR}')
        return

    all_files = {
        f[:-len('_cluster.json')]: os.path.join(CACHE_DIR, f)
        for f in sorted(os.listdir(CACHE_DIR))
        if f.endswith('_cluster.json')
    }

    # Determine which names to check
    if args.benchmark:
        names = expected_instances(args.benchmark)
    elif args.filter:
        names = [n for n in all_files if args.filter.lower() in n.lower()]
    else:
        names = sorted(all_files)

    ok_list:   list[str] = []
    bad_list:  list[str] = []
    miss_list: list[str] = []

    for name in names:
        if name not in all_files:
            miss_list.append(name)
            print(f'  [MISS] {name}')
            continue
        path = all_files[name]
        valid, msg = validate(path)
        if valid:
            ok_list.append(name)
            print(f'  [OK  ] {name:35s}  {msg}')
        else:
            bad_list.append(name)
            print(f'  [FAIL] {name:35s}  {msg}')
            if args.delete:
                os.remove(path)
                print(f'         ^ deleted')

    # Summary
    print()
    print(f'Result: {len(ok_list)} OK  |  {len(bad_list)} FAILED  |  {len(miss_list)} MISSING')
    if bad_list:
        print('Failed:', ', '.join(bad_list))
    if miss_list:
        print('Missing:', ', '.join(miss_list))

    needs_retry = bad_list + miss_list
    if needs_retry:
        if args.delete:
            print('\nDeleted failed caches. Re-run training to retry only these instances.')
        else:
            print('\nRe-run with --delete to remove failed caches, then restart training.')
            print('Training will only re-call LLM for missing/deleted cache files.')
    else:
        print('\nAll caches valid.')


if __name__ == '__main__':
    main()
