"""
generate_train_data.py
Pre-generate fixed VRPTW training datasets for reproducible training.

Usage:
  python generate_train_data.py --type C --n 50000 --seed 42 --out data/train_C_50k.pt
  python generate_train_data.py --type R  --n 50000 --seed 42 --out data/train_R_50k.pt
  python generate_train_data.py --type RC --n 50000 --seed 42 --out data/train_RC_50k.pt
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_vrptw import generate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type',  required=True, choices=['C', 'R', 'RC', 'UNIFIED'])
    parser.add_argument('--n',     type=int, default=50000)
    parser.add_argument('--seed',  type=int, default=42)
    parser.add_argument('--out',   required=True)
    parser.add_argument('--n-customers', type=int, default=100)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f'Generating {args.n} {args.type}-type instances (n={args.n_customers}, seed={args.seed}) ...')
    t0 = time.time()
    dataset = generate_dataset(
        gen_type=args.type,
        n_instances=args.n,
        seed=args.seed,
        n_customers=args.n_customers,
    )
    elapsed = time.time() - t0
    print(f'  Done in {elapsed:.1f}s  ({args.n / elapsed:.0f} inst/s)')

    torch.save(dataset, args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f'  Saved -> {args.out}  ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
