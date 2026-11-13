"""Wait for independent train20 caches, validate, then train without test calls."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import train_vrptw as base
from run_rc101_kim_conversation import decode
from run_fixed_kim_sonnet import train, llm, backend, MODEL, save, validate, load_solomon


class PilotTrainer(train.SoftClusterTrainer):
    def _train_one_epoch(self, epoch):
        started = time.time()
        result = super()._train_one_epoch(epoch)
        print(f'[Epoch {epoch}] reward={result[0]:.6f} loss={result[1]:.6f} '
              f'seconds={time.time()-started:.1f}', flush=True)
        return result

    def _train_one_batch(self, inst, batch_size):
        result = super()._train_one_batch(inst, batch_size)
        if not getattr(self, '_first_batch_ok', False):
            print(f'[First batch OK] {inst["name"]} reward={result[0]} loss={result[1]}', flush=True)
            self._first_batch_ok = True
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, default=ROOT/'data/rc101_event_pilot_discrete_v1')
    ap.add_argument('--base-guidance', choices=['llm', 'off'], required=True)
    args = ap.parse_args()
    cache_run, out = args.cache_run.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'settings.json').exists(), 'Training already initialized'
    deadline = time.monotonic() + 3600
    while True:
        manifest = json.loads((cache_run/'manifest.json').read_text())
        assert manifest['settings']['mode'] == 'independent'
        if manifest['status'] == 'complete':
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Cache generation did not complete within one hour; training not started')
        print('WAITING FOR CACHES', manifest['completed'], '/', len(manifest['settings']['order']), flush=True)
        time.sleep(20)
    data = args.data_dir.resolve()
    split = json.loads((data/'split.json').read_text())
    names = [row['name'] for row in split['train']]
    assert names == manifest['settings']['order'] and len(names) in (20, 40, 80)
    assert not set(names) & {r['name'] for r in split['test']}
    clusters = manifest['settings']['clusters']
    caches = out/'cache'
    caches.mkdir(exist_ok=True)
    for index, name in enumerate(names):
        assert hashlib.sha256((data/(name+'.txt')).read_bytes()).hexdigest() == manifest['settings']['inputs'][name]['instance']
        request = json.loads((cache_run/'turns'/f'{index+1:02}_{name}'/'request.json').read_text())
        assert len(request['messages']) == 2, 'Non-independent request detected'
        record = json.loads((cache_run/'turns'/f'{index+1:02}_{name}'/'validated.json').read_text())
        _, cache = decode(record['assistant']['content'], clusters)
        assert cache == record['cache']
        assert record['request_sha256'] == hashlib.sha256(json.dumps(request['messages']).encode()).hexdigest()
        validate(cache, clusters)
        save(caches/(name+'_cluster.json'), cache)
        inst = load_solomon(str(data/(name+'.txt')))
        accidents = [e for e in inst['preset_events'] if e['type'] == 'ACCIDENT']
        if accidents:
            assert len(accidents) == 1
            start = float(accidents[0]['trigger_time'])/float(inst['T'])
            save(caches/f'{name}_acc_refresh_{start:.6f}.json', cache)
    baseline = caches/'RC101_cluster.json'
    if not baseline.exists():
        if args.base_guidance == 'llm':
            messages = [dict(role='system', content=manifest['settings']['common_prompt']),
                dict(role='user', content='Scenario RC101 BASE: no rain or accident. '
                     'Assign full confidence to the same Kim groups with no event effects.')]
            fmt = manifest['settings']['response_format']
            save(out/'base_request.json', dict(model=MODEL, messages=messages, response_format=fmt))
            response = backend.query_llm('', MODEL, messages=messages, response_format=fmt)
            (out/'base_response.txt').write_text(response, encoding='utf-8')
            _, cache = decode(response, clusters)
        else:
            cache = {str(k): {str(n): 0.0 for n in ns} for k, ns in enumerate(clusters)}
            cache['_depot'] = {str(k): 0.0 for k in range(len(clusters))}
        save(baseline, cache)
    validate(json.loads(baseline.read_text()), clusters)
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    train._set_seed(42)
    tp = copy.deepcopy(train.trainer_params)
    tp.update(data_dir=str(data), train_instances=names, test_instances=[],
        result_dir=str(out/'model'), epochs=1000, train_batch_size=64, train_episodes=2500,
        pomo_size=100, seed=42, model_load=dict(enable=False), cluster_mode='kim2006',
        bias_mode='cluster', start_mode='random', benchmark='rc1', llm_cache_dir=str(caches),
        experience_refresh_epoch=-1, rl_recluster_epoch=None, use_augmentation=True,
        train_aug8=False, curriculum_epoch=0, type_ratio=None,
        logging=dict(model_save_interval=500, log_interval=10))
    lp = dict(train.llm_params, enabled=True, model=MODEL, reward_config='E', bias_strength=5.,
        force_refresh=False, use_fewshot_cache=False, use_misfit=False, no_init_llm=False)
    ep = dict(base.env_params, **base._REWARD_CONFIGS_RC1['E'], pomo_size=100)
    save(out/'settings.json', dict(trainer=tp, llm=lp, environment=ep,
        model=train.model_params, optimizer=train.optimizer_params,
        cache_source=str(cache_run), base_guidance=args.base_guidance,
        test_deferred=True, initialized_from='random', base_training_instance=False))
    provenance = out/'provenance'
    provenance.mkdir(exist_ok=True)
    for src in [data/'split.json', cache_run/'manifest.json', ROOT/'train_soft_cluster.py',
                ROOT/'train_vrptw_llm.py', ROOT/'event_observation.py', ROOT/'vrptw_env.py', Path(__file__)]:
        shutil.copy2(src, provenance/src.name)
    def no_call(*a, **kw):
        raise RuntimeError('Training is cache-only; unexpected LLM call')
    llm.query_llm = backend.query_llm = no_call
    trainer = PilotTrainer(ep, copy.deepcopy(train.model_params), copy.deepcopy(train.optimizer_params), tp, lp)
    assert len(trainer.instance_pool) == len(names) and not trainer.test_pool
    for inst in trainer.instance_pool:
        if 'ACC' in inst['name']:
            stages = trainer._acc_stage_events[inst['name']]
            assert len(stages) == 1 and stages[0]['t_start'] > 0
    save(out/'pipeline.json', dict(stage='training', train=len(names), start_epoch=1, end_epoch=1000,
        initialized_from='random', depot_confidence='retained until depot return'))
    print(f'[START] Kim + independent full confidence, 1000 epochs, b64, train{len(names)} only', flush=True)
    trainer.run()
    save(out/'pipeline.json', dict(stage='complete',epoch=1000,train=len(names)))


if __name__ == '__main__':
    main()
