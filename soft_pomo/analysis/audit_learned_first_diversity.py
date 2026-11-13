"""Read-only CPU audit of first-customer diversity across 100 sampled rollouts."""
import argparse
import statistics
import sys
from pathlib import Path

import run_learned_first_train20 as run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=32)
    parser.add_argument('--initial', action='store_true')
    args = parser.parse_args()
    run.verify()
    sys.path.insert(0, str(run.OUT / 'source'))
    import torch
    import train_vrptw as pure
    config = run.io.read(run.OUT / f'late{args.weight}/config.json')
    checkpoint = run.OUT / f'late{args.weight}/training/checkpoint-last.pt'
    saved = {'epoch': 0} if args.initial else torch.load(checkpoint, map_location='cpu', weights_only=False)
    out = run.OUT / f'late{args.weight}/first_customer_audit_epoch{saved["epoch"]}'
    trainer = run.io.build_trainer(config, run.OUT / 'source', out / 'runtime', cpu=True)
    if not args.initial:
        trainer.model.load_state_dict(saved['model_state_dict'], strict=True)
    trainer.model.train()
    pure._set_seed(20261017)
    rows = []
    with torch.no_grad():
        for inst in trainer.instance_pool:
            batch = pure.make_batch(inst, args.repeats, trainer.device, augment8=False)
            trainer.env.load_problems(batch)
            reset, _, _ = trainer.env.reset()
            _, _, rain = pure._get_rain_event(inst)
            reset.rain_tokens = pure._make_rain_tokens(inst, rain, args.repeats, trainer.device)
            reset.acc_tokens = pure._make_acc_tokens([], inst['n_customers'], args.repeats, trainer.device)
            trainer.model.pre_forward(reset)
            state, _, _ = trainer.env.pre_step()
            depot, _ = trainer.model(state)
            assert (depot == 0).all()
            state, _, _ = trainer.env.step(depot)
            selected, probabilities = trainer.model(state)
            assert selected.shape == (args.repeats, 100)
            assert (selected > 0).all()
            counts = [torch.bincount(s, minlength=101) for s in selected]
            unique = [int((c[1:] > 0).sum()) for c in counts]
            maximum = [int(c.max()) for c in counts]
            total = torch.stack(counts).sum(0)
            top = sorted(range(1, 101), key=lambda n: int(total[n]), reverse=True)[:5]
            row = dict(name=inst['name'], unique_mean=statistics.mean(unique),
                       sampled_probability_mean=float(probabilities.mean()),
                       sampled_probability_min=float(probabilities.min()),
                       unique_min=min(unique), unique_max=max(unique),
                       largest_count_mean=statistics.mean(maximum),
                       unique_counts=unique,
                       top_customers=[dict(customer=n, fraction=int(total[n])/(100*args.repeats)) for n in top])
            rows.append(row)
            print(f'{inst["name"]}: distinct={row["unique_mean"]:.2f} '
                  f'range={min(unique)}-{max(unique)} max_share={row["largest_count_mean"]:.2f}%', flush=True)
    summary = dict(epoch=saved['epoch'], weight=args.weight, seed=20261017,
                   repeats=args.repeats, rollouts=100, device='cpu',
                   note='Fresh first-action sampling from saved checkpoint, not historical training logs.',
                   unique_mean=statistics.mean(r['unique_mean'] for r in rows),
                   unique_min=min(r['unique_min'] for r in rows),
                   unique_max=max(r['unique_max'] for r in rows),
                   largest_count_mean=statistics.mean(r['largest_count_mean'] for r in rows),
                   rows=rows)
    run.io.write(out / 'summary.json', summary)
    print(f'SUMMARY epoch={saved["epoch"]} distinct={summary["unique_mean"]:.2f} '
          f'range={summary["unique_min"]}-{summary["unique_max"]} '
          f'max_share={summary["largest_count_mean"]:.2f}%', flush=True)


if __name__ == '__main__':
    main()
