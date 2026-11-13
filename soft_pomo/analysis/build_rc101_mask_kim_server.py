"""Freeze ten RC101 Kim train-only jobs; never submits jobs itself."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'rc101_pair100_200_l5_mask_kim_s42_46_20261111'
OUT = ROOT/'server_export'/NAME
SOURCE = ROOT/'result/rc101_phase2_online_gpt_train20_v2_2_20261018/source'
WEIGHTS = (100, 200)
SEEDS = tuple(range(42, 47))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate_inputs():
    split = read(SOURCE/'split.json')
    assert len(split['train']) == 20 and len(split['test']) == 80
    artifacts = {}
    for weight in WEIGHTS:
        art = ROOT/f'result/rc101_pair{weight}_l5_mask_kim_cache_20261111'
        assert read(art/'complete.json')['train_count'] == 20
        assert read(art/'split.json') == split
        cfg = read(art/'config.json')
        assert cfg['trainer']['epochs'] == 1000
        assert cfg['trainer']['train_instances'] == [r['name'] for r in split['train']]
        assert cfg['environment']['protect_last_service_vehicle'] is True
        assert cfg['environment']['normalize_advantage_std'] is True
        assert cfg['environment']['late_penalty'] == 5
        assert abs(cfg['environment']['vehicle_penalty']-weight*100/1696.95) < 1e-8
        assert cfg['model']['logit_clipping'] == 10
        expected = {r['name']+'.json' for r in split['train']}
        assert {p.name for p in (art/'train_cache').glob('*.json')} == expected
        groups = read(art/'groups.json')
        assert sorted(n for group in groups for n in group) == list(range(1, 101))
        assert (art/'reference.json').is_file()
        assert (art/'prompt_source.py').is_file()
        for row in split['train']:
            cached = read(art/'train_cache'/(row['name']+'.json'))
            assert cached['initial'] is not None
            assert (cached['post_disclosure'] is not None) == ('_ACC_' in row['name'])
        artifacts[weight] = art
    return artifacts


def main():
    artifacts = validate_inputs()
    OUT.mkdir(exist_ok=False)
    for path in list(ROOT.glob('*.py')) + list((ROOT/'experiments').rglob('*.py')):
        dest = OUT/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    (OUT/'dataset').mkdir()
    shutil.copy2(SOURCE/'split.json', OUT/'dataset/split.json')
    shutil.copytree(SOURCE/'data', OUT/'dataset/data')
    for weight, art in artifacts.items():
        target = OUT/'artifacts'/f'vehicle{weight}'
        target.mkdir(parents=True)
        # Copy structured artifacts only; no API call logs or credential files.
        for path in art.glob('*.json'):
            shutil.copy2(path, target/path.name)
        shutil.copy2(art/'prompt_source.py', target/'prompt_source.py')
        shutil.copytree(art/'train_cache', target/'train_cache')
    default = OUT/'result/rc101_learned_first_train20_20261017/late5/config.json'
    default.parent.mkdir(parents=True)
    shutil.copy2(artifacts[100]/'config.json', default)
    path = OUT/'experiments/concurrent_fleet_pair/train.py'
    code = path.read_text(encoding='utf-8')
    marker = "        status('evaluation'); model.eval(); results=[]"
    assert code.count(marker) == 1
    code = code.replace(marker,
        "        write(folder/'training_complete.json',dict(epoch=tp['epochs'],test_count=0,evaluation_pending=True))\n"
        "        write(args.out/'status.json',dict(stage='training_complete',evaluation_pending=True))\n"
        "        return\n"+marker)
    path.write_text(code, encoding='utf-8', newline='\n')
    shutil.copy2(ROOT/'analysis/rc101_mask_kim_server_train.py', OUT/'run_training.py')
    submit = (ROOT/'analysis/submit_seq_v100_lanes.sh').read_text(encoding='utf-8')
    assert 'seq 0 9' in submit and 'jobs[index-2]' in submit and '^sj_' in submit
    (OUT/'submit.sh').write_text(submit, encoding='ascii', newline='\n')
    tasks = [(weight, seed) for weight in WEIGHTS for seed in SEEDS]
    (OUT/'tasks.txt').write_text(''.join(f'{w} {s}\n' for w,s in tasks), encoding='ascii', newline='\n')
    task = (ROOT/'analysis/pair_return_mask_task.slurm').read_text(encoding='utf-8')
    task = task.replace('sj_pair_mask_l5', 'sj_rc_kim_mask_l5').replace('"$index" -lt 20', '"$index" -lt 10')
    start = task.index('read -r case weight seed denominator')
    end = task.index('args=()', start)
    task = task[:start]+'read -r weight seed < <(sed -n "$((index+1))p" tasks.txt)\nout="runs/vehicle$weight/seed$seed"\n'+task[end:]
    start = task.index(' python -u -m experiments.concurrent_fleet_pair.train')
    end = task.index('\ncode=$?', start)
    task = task[:start]+' python -u run_training.py --out "$out" --seed "$seed" --vehicle-cost "$weight" "${args[@]}"'+task[end:]
    start = task.index("    python3 -c '")
    task = task[:start]+'''    python3 -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); c=json.load(open(p/"late5/training_complete.json")); assert c["epoch"]==1000 and c["evaluation_pending"] and c["test_count"]==0; assert (p/"late5/checkpoint-1000.pt").is_file()' "$out"
fi
'''
    (OUT/'task.slurm').write_text(task, encoding='ascii', newline='\n')
    hashes = {p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
              for p in OUT.rglob('*') if p.is_file()}
    manifest = dict(sha256=hashes, seeds=SEEDS, vehicle_weights=WEIGHTS, late=5, beta=5,
        mask=True, normalize_advantage_std=True, max_concurrent_jobs=2,
        tasks=[dict(index=i, vehicle_weight=w, seed=s) for i,(w,s) in enumerate(tasks)],
        evaluation='train only on server; downloaded checkpoints require causal local Test/API evaluation')
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8', newline='\n')
    archive = OUT.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(OUT, arcname=NAME)
    Path(str(archive)+'.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n', encoding='ascii', newline='\n')
    print('BUILT', archive)


if __name__ == '__main__':
    main()
