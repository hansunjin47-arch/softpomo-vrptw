"""Serial local Phase2 continuation/seed experiments using frozen v5 caches."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'result/rc101_ontology_v5_train20_b128_seed43'
BUNDLE = SOURCE / 'training_bundle'
TEST = BUNDLE / 'eval_addon'
CACHES = SOURCE / 'eval_1000_test80/cache'
OUT = ROOT / 'result/rc101_ontology_v5_followups_b128'
sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(BUNDLE / 'server'))
from train_cached import read, write, sha, verify_bundle, validate_cache, validate_resume


def immutable(path, value):
    if path.exists():
        assert read(path) == value, f'Experiment contract changed: {path}'
    else:
        write(path, value)


def check():
    import torch
    digest = verify_bundle(BUNDLE)
    saved = read(SOURCE / 'training_fixed/settings.json')
    state = torch.load(SOURCE / 'training_fixed/checkpoint-1000.pt', map_location='cpu', weights_only=False)
    assert state['epoch'] == 1000 and state['server_contract'] == saved
    assert saved['bundle_sha256'] == digest
    config = saved['config']
    assert config['trainer']['seed'] == 43 and config['trainer']['train_batch_size'] == 128
    assert config['llm']['bias_strength'] == 5
    split = read(TEST / 'split.json')
    assert len(split['train']) == 20 and len(split['test']) == 80
    hashes = {}
    for group, folder in [('train', BUNDLE / 'cache'), ('test', CACHES)]:
        for entry in split[group]:
            path = folder / (entry['name'] + '_rl_cluster.json')
            validate_cache(path)
            hashes[group + '/' + path.name] = sha(path)
    plan = dict(order=[43, 44, 45], final_epoch=2000, batch=128, beta=5,
        initial_reference_seed=43, reference_epoch=500,
        scope='Phase2 RNG variation only; the Pure500 checkpoint, reference routes and all LLM caches remain fixed.',
        seed43_start_epoch=1000, new_seed_start_epoch=500,
        evaluation_epochs={'43':[2000], '44':[1000,2000], '45':[1000,2000]},
        api_calls=0, max_hours_per_segment=11.5, bundle_sha256=digest, cache_sha256=hashes,
        checkpoint1000_sha256=sha(SOURCE / 'training_fixed/checkpoint-1000.pt'),
        pure500_sha256=sha(BUNDLE / 'checkpoints/pure500.pt'))
    immutable(OUT / 'plan.json', plan)
    print('CHECK PASSED: checkpoint, frozen bundle, train20/test80 caches; no API calls.', flush=True)
    return plan


def evaluate(run, epoch):
    from train_cached import build_trainer
    import torch
    output = run / f'eval_{epoch}_test80'
    checkpoint = run / f'checkpoint-{epoch}.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['epoch'] == epoch
    contract = state['server_contract']
    validate_resume(contract, read(run / 'settings.json'))
    assert contract['bundle_sha256'] == verify_bundle(BUNDLE)
    config = copy.deepcopy(contract['config'])
    entries = read(TEST / 'split.json')['test']
    names = [e['name'] for e in entries]
    assert len(names) == len(set(names)) == 80
    assert not set(names) & set(config['trainer']['train_instances'])
    hashes = {}
    for entry in entries:
        assert sha(TEST / 'data' / (entry['name'] + '.txt')) == entry['sha256']
        path = CACHES / (entry['name'] + '_rl_cluster.json')
        validate_cache(path)
        hashes[entry['name']] = sha(path)
        assert hashes[entry['name']] == read(OUT/'plan.json')['cache_sha256']['test/'+path.name]
    immutable(output / 'protocol.json', dict(checkpoint_sha256=sha(checkpoint), cache_sha256=hashes,
        bundle_sha256=contract['bundle_sha256'], epoch=epoch, names=names, augmentation=8,
        pomo=100, per_instance_seed='42 + original test index', beta=5, api_calls=0))
    if (output / 'status.json').exists() and read(output / 'status.json').get('status') == 'complete':
        assert len(read(output / 'results.json')) == 80
        return
    config['cache_dir'] = str(CACHES)
    config['trainer'].update(train_instances=names, use_augmentation=True, pomo_size=100, n_mc_samples=1)
    config['model']['eval_type'] = 'argmax'
    trainer = build_trainer(config, TEST, output / 'runtime')
    trainer.model.load_state_dict(state['model_state_dict'], strict=True)
    trainer.model.eval()
    import train_vrptw as pure
    from vrptw_env import _solution_stats
    results = read(output / 'results.json') if (output / 'results.json').exists() else []
    assert [r['name'] for r in results] == names[:len(results)]
    for index, inst in enumerate(trainer.instance_pool):
        if index < len(results):
            continue
        pure._set_seed(42 + index)
        with torch.no_grad():
            routes, reward = trainer._best_solution_aug(inst, n_aug=8)
        assert sorted(int(n) for route in routes for n in route if n) == list(range(1,101))
        s = _solution_stats(inst, routes)
        assert s['n_served'] == 100
        row = dict(name=inst['name'], kind=entries[index]['kind'], vehicles=len(s['per_vehicle']),
            travel_time=s['total_travel_time'], late_count=s['late_count'], total_late=s['total_late'],
            served=s['n_served'], routes=routes, reward=float(reward))
        row['cost'] = .04*row['vehicles'] + row['travel_time']/5760 + .0075*row['total_late']
        assert abs(row['cost'] + row['reward']) < 1e-4
        results.append(row)
        write(output/'results.json', results)
        write(output/'status.json', dict(status='evaluating', completed=len(results), total=80))
        print('EVAL',epoch,len(results),inst['name'],row['cost'],flush=True)
    keys = ('vehicles','travel_time','late_count','total_late','served','cost')
    summary = {}
    for kind in ('ALL','RAIN','ACCIDENT'):
        rows = [r for r in results if kind == 'ALL' or r['kind'] == kind]
        assert len(rows) == (80 if kind == 'ALL' else 40)
        summary[kind] = dict(count=len(rows), zero_late=sum(r['late_count']==0 for r in rows),
            **{k:statistics.mean(r[k] for r in rows) for k in keys},
            std={k:statistics.stdev(r[k] for r in rows) for k in keys})
    write(output/'summary.json', dict(summary=summary))
    write(output/'status.json', dict(status='complete', completed=80))


def run_logged(command, folder, tag):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/(tag+'_stdout.log')).open('a', encoding='utf-8') as stdout, \
         (folder/(tag+'_stderr.log')).open('a', encoding='utf-8') as stderr:
        subprocess.run(command, cwd=BUNDLE, stdout=stdout, stderr=stderr, check=True)


def collect():
    rows = []
    for seed in (43,44,45):
        for epoch in (1000,2000):
            folder = SOURCE/'eval_1000_test80' if seed==43 and epoch==1000 else OUT/f'seed{seed}'/'training'/f'eval_{epoch}_test80'
            if not (folder/'summary.json').exists():
                continue
            summary = read(folder/'summary.json')['summary']['ALL']
            rows.append(dict(seed=seed, epoch=epoch, source=str(folder), summary=summary))
    comparisons = {}
    with tarfile.open(ROOT/'server_export/train20_seed43_results.tar.gz') as archive:
        for row in rows:
            if row['seed'] != 43:
                continue
            epoch = row['epoch']
            pure = json.load(archive.extractfile(f'train20/runs/pure128_4000_seed43/eval_{epoch}_test80/results.json'))
            guided = read(Path(row['source'])/'results.json')
            assert [r['name'] for r in pure] == [r['name'] for r in guided]
            for record in pure:
                record['cost'] = .04*record['vehicles'] + record['travel_time']/5760 + .0075*record['total_late']
            comparisons[str(epoch)] = {key:dict(pure_mean=statistics.mean(r[key] for r in pure),
                guided_mean=statistics.mean(r[key] for r in guided),
                improved=sum(b[key]<a[key]-1e-8 for a,b in zip(pure,guided)),
                worsened=sum(b[key]>a[key]+1e-8 for a,b in zip(pure,guided)))
                for key in ('vehicles','travel_time','late_count','total_late','cost')}
    aggregates = {}
    for epoch in (1000,2000):
        selected = [r for r in rows if r['epoch']==epoch]
        if len(selected)==3:
            aggregates[str(epoch)] = {k:dict(mean=statistics.mean(r['summary'][k] for r in selected),
                std_across_phase2_seeds=statistics.stdev(r['summary'][k] for r in selected))
                for k in ('vehicles','travel_time','late_count','total_late','cost')}
    write(OUT/'results_summary.json',dict(runs=rows,pure_seed43_comparisons=comparisons,
        aggregate=aggregates,scope='Fixed seed43 Pure500 and fixed v5 caches; only Phase2 training RNG varies.'))


def queue():
    plan = check()
    immutable(OUT/'runner_provenance.json', dict(script_sha256=sha(Path(__file__)),
        source=str(Path(__file__).resolve()), python=sys.executable))
    lock = OUT/'queue.lock'
    with lock.open('x') as handle:
        handle.write(str(os.getpid()))
    try:
        for seed in plan['order']:
            run = OUT/f'seed{seed}'/'training'
            if seed == 43:
                run.mkdir(parents=True, exist_ok=True)
                if not (run/'settings.json').exists():
                    assert not any(run.iterdir()), 'Inspect incomplete initialization before retrying'
                    for src, dest in [('checkpoint-1000.pt','checkpoint-last.pt'),
                                      ('checkpoint-1000.pt','checkpoint-1000.pt'),('settings.json','settings.json')]:
                        shutil.copy2(SOURCE/'training_fixed'/src, run/dest)
            for segment in range(8):
                if (OUT/'STOP').exists():
                    write(OUT/'status.json',dict(stage='stopped',seed=seed))
                    return
                status = read(run/'status.json') if (run/'status.json').exists() else {}
                if status.get('status') == 'complete' and status.get('epoch') == 2000:
                    break
                resume = (run/'checkpoint-last.pt').exists()
                write(OUT/'status.json',dict(stage='training',seed=seed,target=2000,segment=segment+1))
                command = [sys.executable,'-u',str(BUNDLE/'server/train_cached.py'),'--method','phase2',
                    '--out',str(run),'--seed',str(seed),'--epochs','2000','--max-hours','11.5']
                if resume:
                    command.append('--resume')
                before = status.get('epoch',1000 if seed==43 else 500)
                run_logged(command,run.parent,'train')
                after = read(run/'status.json')
                assert after['epoch'] > before, 'Training did not advance; refusing a retry loop'
                if after['status'] == 'complete' and after['epoch'] == 2000:
                    break
            else:
                raise RuntimeError('Segment budget exhausted')
            for epoch in plan['evaluation_epochs'][str(seed)]:
                if (OUT/'STOP').exists():
                    write(OUT/'status.json',dict(stage='stopped',seed=seed))
                    return
                write(OUT/'status.json',dict(stage='evaluating',seed=seed,epoch=epoch))
                run_logged([sys.executable,'-u',str(Path(__file__).resolve()),'evaluate',
                    '--run',str(run),'--epoch',str(epoch)],run.parent,f'eval_{epoch}')
                collect()
        write(OUT/'status.json',dict(stage='complete',seeds=plan['order'],epoch=2000))
    except Exception as exc:
        write(OUT/'status.json',dict(stage='failed',error=str(exc)))
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['check','queue','evaluate'])
    parser.add_argument('--run',type=Path)
    parser.add_argument('--epoch',type=int)
    args = parser.parse_args()
    if args.stage == 'evaluate':
        if args.run is None or args.epoch is None:
            parser.error('evaluate requires --run and --epoch')
        evaluate(args.run.resolve(),args.epoch)
    else:
        globals()[args.stage]()
