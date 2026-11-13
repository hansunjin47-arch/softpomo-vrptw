"""Pack three offline BKS training runs; Kim train80 API calls are deferred."""
import argparse
import copy
import json
import shutil
import tarfile

from call_kim_rc101_bks_gpt import ROOT, OUT as TRAIN20, io, immutable, decode, call_once
from phase2_routes_only import PUBLIC

OUT = ROOT / 'result/rc101_kim_train80_bks_customer_only_gpt54_v1'
BUNDLE = ROOT / 'server_export/rc101_bks_train20_80_e1000_20261013'
DATA = ROOT / 'data/rc101_train80_test80_20260929'


def prepare():
    from vrptw_env import load_solomon
    import torch
    protocol = io.read(TRAIN20 / 'protocol.json')
    assert io.read(TRAIN20 / 'status.json')['stage'] == 'complete'
    template = io.read(TRAIN20 / 'requests' / protocol['train_names'][0] / 'request.json')
    prefix, raw = template['messages'][0]['content'].rsplit('\n', 1)
    fixed = json.loads(raw)['fixed_instance']
    split = io.read(DATA / 'split.json')
    names = [r['name'] for r in split['train']]
    assert len(set(names)) == 80 and not set(names) & {r['name'] for r in split['test']}
    assert set(protocol['train_names']) <= set(names)
    basepath = ROOT / 'server_export/kim_tt_ac_train20_20261010/A/data' / (protocol['train_names'][0] + '.txt')
    base = load_solomon(str(basepath))
    for row in split['train']:
        name = row['name']
        path = DATA / (name + '.txt')
        assert io.sha(path) == row['sha256']
        inst = load_solomon(str(path))
        for key in ('depot_xy', 'node_xy', 'node_demand', 'node_tw_open', 'node_tw_close', 'node_service'):
            assert torch.equal(base[key], inst[key]), (name, key)
        observation = io.read(DATA / (name + '.observations.json'))
        assert set(observation) <= PUBLIC
        context = dict(scenario=name, fixed_instance=fixed, public_event=observation)
        request = copy.deepcopy(template)
        request['messages'][0]['content'] = prefix + '\n' + json.dumps(context, separators=(',', ':'), allow_nan=False)
        immutable(OUT / 'requests' / name / 'request.json', request)
        if name in protocol['train_names']:
            original = io.read(TRAIN20 / 'requests' / name / 'request.json')
            assert original == request, name
            for filename in ('response.txt',):
                shutil.copy2(TRAIN20 / 'requests' / name / filename, OUT / 'requests' / name / filename)
            immutable(OUT / 'requests' / name / 'reused.json', dict(source=str(TRAIN20 / 'requests' / name)))
    protocol.update(version=OUT.name, train_names=names)
    immutable(OUT / 'protocol.json', protocol)
    immutable(OUT / 'request_hashes.json', {n: io.sha(OUT / 'requests' / n / 'request.json') for n in names})
    print('PREPARED 80 requests; 20 exact requests reused, 60 new calls', flush=True)
    return protocol


def call(protocol):
    hashes = io.read(OUT / 'request_hashes.json')
    for index, name in enumerate(protocol['train_names']):
        folder = OUT / 'requests' / name
        assert io.sha(folder / 'request.json') == hashes[name]
        response = folder / 'response.txt'
        io.write(OUT / 'status.json', dict(stage='calling', completed=index, total=80, scenario=name))
        if not response.exists():
            if (folder / 'attempted.json').exists():
                raise RuntimeError('Previous attempt requires inspection: ' + name)
            immutable(folder / 'attempted.json', dict(model=protocol['model']))
            print('CALL', index + 1, name, flush=True)
            response.write_text(call_once(io.read(folder / 'request.json'), folder), encoding='utf-8')
        cache = decode(response.read_text(encoding='utf-8'), protocol['clusters'])
        immutable(OUT / 'cache' / (name + '_cluster.json'), cache)
        print('VALIDATED', index + 1, name, flush=True)
    immutable(OUT / 'cache_manifest.json', dict(files={p.name: io.sha(p) for p in (OUT / 'cache').glob('*.json')}))


def pack():
    if BUNDLE.exists():
        raise RuntimeError('Bundle already exists; do not overwrite frozen artifacts')
    for size in (20, 80):
        source = ROOT / ('server_export/kim_tt_ac_train20_20261010/A' if size == 20 else
                         'server_export/pure_tt_ac_train80_e2000_20261012/A')
        split_source = ROOT / 'result/rc101_pure_train20_b128_s42_configA_travel_time_1000_20261010_v2/split.json' if size == 20 else source / 'split.json'
        split = io.read(split_source)
        cache_source = TRAIN20 if size == 20 else OUT
        names = [r['name'] for r in split['train']]
        if size == 20:
            assert names == io.read(cache_source / 'protocol.json')['train_names']
        assert names == [r['name'] for r in split['train']]
        for method in (('pure', 'kim') if size == 20 else ('pure',)):
            target = BUNDLE / f'train{size}_{method}'
            (target / 'server').mkdir(parents=True)
            for p in source.glob('*.py'):
                shutil.copy2(p, target / p.name)
            shutil.copy2(ROOT / 'server/train_cached.py', target / 'server/train_cached.py')
            (target / 'data').mkdir()
            for row in split['train'] + split['test']:
                p = source / 'data' / (row['name'] + '.txt')
                assert io.sha(p) == row['sha256']
                shutil.copy2(p, target / 'data' / p.name)
            config_source = ROOT / ('server_export/kim_tt_ac_train20_20261010/A/configs/kim.json' if method == 'kim' else
                                  'result/rc101_pure_train20_b128_s42_configA_travel_time_1000_20261010_v2/config.json')
            config = io.read(config_source)
            config['trainer'].update(train_instances=names, epochs=1000, seed=42, train_batch_size=128)
            config['environment'].update(vehicle_penalty=100/14, time_penalty=1.0,
                TT_max=1696.95/240, late_penalty=1.0, Lt_max=1696.95/240,
                dist_penalty_weight=0.0, late_count_penalty=0.0)
            if method == 'kim':
                config['llm'].update(model='openai/gpt-5.4', reward_config='BKS')
                for name in names:
                    p = cache_source / 'cache' / (name + '_cluster.json')
                    cache = io.validate_cache(p)
                    assert all(v == 0 for v in cache['_depot'].values())
                    immutable(target / 'cache/kim' / p.name, cache)
                immutable(target / 'provenance/cache_protocol.json', io.read(cache_source / 'protocol.json'))
            immutable(target / 'configs' / (method + '.json'), config)
            immutable(target / 'split.json', split)
            immutable(target / 'provenance/objective.json', dict(objective='K/14+(T+L)/1696.95',
                horizon=240, fresh_training=True, seed=42, epochs=1000))
            files = [p for p in target.rglob('*') if p.is_file()]
            for p in files:
                text = p.read_text(encoding='utf-8')
                assert not any(s in text for s in ('sk-or-v1-', 'sk-ant-api', 'BEGIN OPENSSH PRIVATE KEY')), p
            immutable(target / 'bundle_manifest.json', dict(sha256={p.relative_to(target).as_posix(): io.sha(p) for p in files}))
            io.verify_bundle(target)
    script = (ROOT / 'server/rc101_bks_four.slurm').read_text(encoding='utf-8')
    (BUNDLE / 'rc101_bks_four.slurm').write_text(script, encoding='ascii', newline='\n')
    archive = BUNDLE.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(BUNDLE, arcname=BUNDLE.name)
    archive.with_suffix('.gz.sha256').write_text(io.sha(archive) + '  ' + archive.name + '\n', encoding='ascii', newline='\n')
    io.write(BUNDLE / 'status.json', dict(stage='ready_for_transfer', training_runs=3,
        archive=str(archive), server_submitted=False))
    print('PACKED', archive, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-api', action='store_true')
    parser.add_argument('--pack-only', action='store_true')
    args = parser.parse_args()
    try:
        if args.pack_only:
            assert not args.allow_api
            pack()
        elif args.allow_api:
            raise RuntimeError('Kim train80 calls deferred by user; additional API calls disabled')
        else:
            protocol = prepare()
    except Exception as error:
        io.write(OUT / 'status.json', dict(stage='failed', error=str(error)))
        raise
