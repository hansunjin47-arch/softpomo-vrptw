"""Cached v2 inference ablation; no training or LLM calls."""
import json

from diagnose_final_runs import ROOT, trainer, soft, torch, load_solomon, _solution_stats


def main():
    torch.set_num_threads(4)
    rows = []
    for family, pool in [('r1', 'r102-r112'), ('rc1', 'rc102-rc108'), ('c1', 'c102-c109')]:
        prefix = family.replace('1', '101').upper()
        for label in ['pure1000', 'v2']:
            directory = (ROOT / f'result/config_E_{family}_final_weightedsum_b32_aug8/{pool}'
                         if label == 'pure1000' else ROOT / f'result_soft/{pool}')
            obj, ignored = trainer(family, pool, directory / 'checkpoint-1000.pt', label == 'pure1000')
            assert not ignored, ignored
            settings = [(0, 'random')] if label == 'pure1000' else [
                (0, 'random'), (1, 'random'), (5, 'random'), (5, 'det_conf')]
            for beta, mode in settings:
                for suffix in ['', '_RAIN2', '_ACC2']:
                    soft._set_seed(42)
                    obj.llm_params['bias_strength'] = beta
                    obj.trainer_params['start_mode'] = mode
                    inst = load_solomon(str(ROOT.parent / 'data/Solomon' / f'{prefix.lower()}{suffix.lower()}.txt'))
                    with torch.inference_mode():
                        routes, reward = obj._best_solution_aug(inst, n_aug=8)
                    stats = _solution_stats(inst, routes)
                    row = dict(checkpoint=label, instance=inst['name'], beta=beta,
                               start_mode=mode, reward=round(reward, 5),
                               vehicles=len(routes), served=stats['n_served'],
                               late=round(float(stats['total_late']), 2),
                               distance=round(float(stats['total_dist']), 2))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    (ROOT / 'analysis/v2_runs_diagnostics.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
