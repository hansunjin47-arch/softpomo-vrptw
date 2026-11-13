"""Generate independent test guidance, then evaluate the completed Kim policy."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_rc101_kim_conversation import main as generate, decode
from run_fixed_kim_sonnet import train, llm, backend, save, load_solomon
from vrptw_env import _solution_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, default=ROOT/'result/rc101_kim_independent1000')
    args = ap.parse_args()
    run = args.run.resolve()
    cache_run = ROOT/'analysis/rc101_kim_independent_test80'
    out = run/'eval_1000_test80'
    out.mkdir(parents=True, exist_ok=True)
    settings = json.loads((run/'settings.json').read_text())
    sys.argv = ['run_rc101_kim_conversation.py', '--out', str(cache_run), '--independent', '--split', 'test',
                '--data-dir', settings['trainer']['data_dir'], '--prepare-only']
    generate()
    settings = json.loads((run/'settings.json').read_text())
    manifest = json.loads((cache_run/'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['completed'] == 80
    data = Path(settings['trainer']['data_dir'])
    split = json.loads((data/'split.json').read_text())
    clusters = manifest['settings']['clusters']
    names = [row['name'] for row in split['test']]
    assert names == manifest['settings']['order']
    caches = out/'cache'
    caches.mkdir(exist_ok=True)
    shutil.copy2(run/'cache/RC101_cluster.json', caches/'RC101_cluster.json')
    for index, entry in enumerate(split['test']):
        name = entry['name']
        path = data/(name+'.txt')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
        turn = cache_run/'turns'/f'{index+1:02}_{name}'
        request = json.loads((turn/'request.json').read_text())
        assert len(request['messages']) == 2
        record = json.loads((turn/'validated.json').read_text())
        assert record['request_sha256'] == hashlib.sha256(json.dumps(request['messages']).encode()).hexdigest()
        _, cache = decode(record['assistant']['content'], clusters)
        assert cache == record['cache']
        save(caches/(name+'_cluster.json'), cache)
        inst = load_solomon(str(path))
        for event in inst['preset_events']:
            if event['type'] == 'ACCIDENT':
                start = float(event['trigger_time'])/float(inst['T'])
                save(caches/f'{name}_acc_refresh_{start:.6f}.json', cache)
    def no_call(*args, **kwargs):
        raise RuntimeError('Evaluation must use validated caches, not new LLM calls')
    llm.query_llm = backend.query_llm = no_call
    torch.set_num_threads(2)
    train._set_seed(42)
    tp = dict(settings['trainer'], test_instances=names,
              test_only=True, llm_cache_dir=str(caches), result_dir=str(out/'runtime'),
              model_load=dict(enable=False))
    trainer = train.SoftClusterTrainer(settings['environment'], settings['model'],
        settings['optimizer'], tp, settings['llm'])
    assert len(trainer.test_pool) == len(names) == 80
    assert [inst['name'] for inst in trainer.test_pool] == names
    checkpoint = run/'model/RC101/checkpoint-1000.pt'
    state = torch.load(checkpoint, map_location=trainer.device, weights_only=False)
    trainer.model.load_state_dict(state['model_state_dict'], strict=True)
    trainer.model.eval()
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        augmentation=8, pomo=100, seed='42 + index', llm=True,
        base_guidance=settings['base_guidance'], cache_run=str(cache_run),
        late_count_includes_depot=True, status='evaluating')
    save(out/'protocol.json', report)
    results = []
    for index, inst in enumerate(trainer.test_pool):
        train._set_seed(42+index)
        routes, reward = trainer._best_solution_aug(inst, n_aug=8)
        stats = _solution_stats(inst, routes)
        row = dict(name=inst['name'], kind=split['test'][index]['kind'],
            vehicles=len(stats['per_vehicle']), travel_time=stats['total_travel_time'],
            late_count=stats['late_count'], total_late=stats['total_late'],
            served=stats['n_served'], reward=float(reward), routes=routes)
        results.append(row)
        save(out/'results.json', results)
        print(f'EVAL {index+1}/80 {row["name"]}: K={row["vehicles"]} '
              f'T={row["travel_time"]:.2f} Lc={row["late_count"]} Lt={row["total_late"]:.2f}', flush=True)
    summary = {}
    for group in ['RAIN', 'ACCIDENT', 'ALL']:
        rows = [r for r in results if group == 'ALL' or r['kind'] == group]
        summary[group] = dict(count=len(rows), **{key: float(np.mean([r[key] for r in rows]))
            for key in ['vehicles', 'travel_time', 'late_count', 'total_late', 'served']},
            zero_late=sum(r['late_count'] == 0 for r in rows))
    report.update(status='complete', summary=summary)
    save(out/'summary.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
