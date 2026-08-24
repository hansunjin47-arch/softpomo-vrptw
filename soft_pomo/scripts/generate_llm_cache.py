"""generate_llm_cache.py — Pre-generate all LLM cluster caches for a benchmark.

Runs the same setup as train_soft_cluster.py but exits after cache generation
without any training. Use this to pre-populate caches before training.

Usage:
    python generate_llm_cache.py --benchmark c1
    python generate_llm_cache.py --benchmark rc1
    python generate_llm_cache.py --benchmark r1
    python generate_llm_cache.py --benchmark rc1 --model qwen3:32b
"""
import subprocess, sys, os

PYTHON = sys.executable
TRAIN  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_soft_cluster.py')

def main():
    # Pass all args through to train_soft_cluster.py, appending --cache-only
    extra = sys.argv[1:]
    cmd = [PYTHON, '-u', TRAIN] + extra + ['--cache-only']
    print(f'[generate_llm_cache] cmd: {" ".join(cmd)}')
    ret = subprocess.call(cmd)
    sys.exit(ret)

if __name__ == '__main__':
    main()
