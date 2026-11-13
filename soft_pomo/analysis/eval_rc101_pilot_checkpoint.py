"""Evaluate a saved pilot policy on its frozen test split, without LLM calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_vrptw as training
from vrptw_env import _solution_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    config = json.loads((run / 'provenance/config.json').read_text())
    split = json.loads((run / 'provenance/split.json').read_text())
    data = Path(config['trainer']['data_dir'])
    for entry in split['test']:
        assert hashlib.sha256((data / (entry['name'] + '.txt')).read_bytes()).hexdigest() == entry['sha256']
    checkpoint = run / 'model/RC101' / f'checkpoint-{args.epoch}.pt'
    out = run / f'eval_{args.epoch}'
    assert not (out / 'results.json').exists(), 'Do not overwrite an existing evaluation'
    out.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    training._set_seed(42)
    tp = dict(config['trainer'], result_dir=str(out / 'runtime'))
    tp['model_load'] = dict(enable=False)
    trainer = training.VRPTWTrainer(config['environment'], config['model'], config['optimizer'], tp)
    assert [inst['name'] for inst in trainer.test_pool] == [r['name'] for r in split['test']]
    assert len(trainer.test_pool) == 80
    saved = torch.load(checkpoint, map_location=trainer.device, weights_only=False)
    trainer.model.load_state_dict(saved['model_state_dict'], strict=True)
    trainer.model.eval()
    results = []
    for index, inst in enumerate(trainer.test_pool):
        training._set_seed(42 + index)
        with torch.no_grad():
            routes, reward = trainer._best_solution(inst)
        stats = _solution_stats(inst, routes)
        row = dict(name=inst['name'], kind=split['test'][index]['kind'],
            vehicles=len(stats['per_vehicle']), travel_time=stats['total_travel_time'],
            late_count=stats['late_count'], total_late=stats['total_late'],
            served=stats['n_served'], reward=reward, routes=routes)
        results.append(row)
        (out / 'results.json').write_text(json.dumps(results, indent=2))
        print(f'[{index+1}/80] {row["name"]} K={row["vehicles"]} T={row["travel_time"]:.2f} '
              f'Lc={row["late_count"]} Lt={row["total_late"]:.2f} served={row["served"]}', flush=True)
    summary = {}
    for group in ['RAIN', 'ACCIDENT', 'ALL']:
        rows = [r for r in results if group == 'ALL' or r['kind'] == group]
        summary[group] = dict(count=len(rows), **{key: float(np.mean([r[key] for r in rows]))
            for key in ['vehicles', 'travel_time', 'late_count', 'total_late', 'served']},
            zero_late=sum(r['late_count'] == 0 for r in rows))
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        seed=42, per_instance_seed='42 + index', augmentation=8, pomo=100, llm=False,
        late_count_includes_depot=True, summary=summary)
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
