"""
mix_datasets.py
C / R / RC 50K 데이터셋을 1/3씩 섞어 train_mixed_50k.pt 생성.

Usage:
    python mix_datasets.py
    python mix_datasets.py --n 150000   # 전체 합치기 (1/3 제한 없이)
"""
import os, sys, argparse, random
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

SOURCES = {
    'C':  os.path.join(_HERE, 'data', 'train_C_50k.pt'),
    'R':  os.path.join(_HERE, 'data', 'train_R_50k.pt'),
    'RC': os.path.join(_HERE, 'data', 'train_RC_50k.pt'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',    type=int, default=50000,
                        help='Total instances in mixed dataset (default: 50000)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out',  default=None,
                        help='Output path (default: data/train_mixed_{n}.pt)')
    args = parser.parse_args()

    random.seed(args.seed)

    datasets = {}
    for k, p in SOURCES.items():
        if not os.path.isfile(p):
            print(f'[Error] not found: {p}')
            sys.exit(1)
        data = torch.load(p, weights_only=False)
        datasets[k] = data
        print(f'  {k}: {len(data)} instances loaded')

    # 1/3 each; distribute remainder to first types
    n_each    = args.n // 3
    remainder = args.n - n_each * 3

    mixed = []
    for i, (k, data) in enumerate(datasets.items()):
        n = n_each + (1 if i < remainder else 0)
        sampled = random.sample(data, min(n, len(data)))
        mixed.extend(sampled)
        print(f'  {k}: sampled {len(sampled)}')

    random.shuffle(mixed)

    out = args.out or os.path.join(_HERE, 'data', f'train_mixed_{args.n//1000}k.pt')
    torch.save(mixed, out)
    print(f'\n[Saved] {len(mixed)} instances → {out}')


if __name__ == '__main__':
    main()
