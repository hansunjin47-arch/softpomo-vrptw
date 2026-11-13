"""Build an allowlisted, relocatable RL-only archive from local experiment files."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_cached import read, sha, validate_cache, write

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['train_vrptw.py', 'train_vrptw_llm.py', 'train_soft_cluster.py',
    'vrptw_env.py', 'generate_vrptw.py', 'event_observation.py', 'advantage.py',
    'VRPTWLLMModule.py', 'VRPTWOntology.py', 'SoftClusterLLMModule.py',
    'SoftClusterOntology.py', 'server/train_cached.py', 'server/train.slurm',
    'server/README.md']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'server_export' / 'soft_pomo_rl_train80')
    ap.add_argument('--phase-run', type=Path,
        default=ROOT / 'result/rc101_pure500_train80_phase2_20260930')
    ap.add_argument('--pure-batch-size', type=int, default=64)
    args = ap.parse_args()
    if args.pure_batch_size < 1:
        ap.error('--pure-batch-size must be positive')
    out = args.out.resolve()
    if out.exists() or out.with_suffix('.zip').exists():
        raise FileExistsError('Choose a new output path; existing bundles are never overwritten')
    data = ROOT / 'data/rc101_train80_test80_20260929'
    pure = ROOT / 'result/rc101_pure_train80_fresh1000_20260929'
    kim = ROOT / 'result/rc101_kim_train80_fresh1000_20260929'
    phase = args.phase_run.resolve()
    split = read(data / 'split.json')
    names = [r['name'] for r in split['train']]
    if len(names) != 80 or len(set(names)) != 80 or set(names) & {r['name'] for r in split['test']}:
        raise ValueError('Unexpected train80 split')
    checkpoint = pure / 'model/RC101/checkpoint-500.pt'
    protocol = read(phase / 'protocol.json')
    if protocol['checkpoint_sha256'] != sha(checkpoint) or protocol['train_names'] != names:
        raise ValueError('Phase2 reference checkpoint/split mismatch')
    if protocol.get('require_reference_start', False):
        raise ValueError('This bundle implements ungated Phase2 only')
    configs = dict(pure=read(pure / 'provenance/config.json'),
                   kim=read(kim / 'settings.json'), phase2=read(phase / 'settings.json'))
    selected = {}
    def include(source, target):
        if not source.is_file():
            raise FileNotFoundError(source)
        selected[target] = source
    for name in SOURCES:
        include(ROOT / name, name)
    for row in split['train']:
        name = row['name']
        if sha(data / (name + '.txt')) != row['sha256']:
            raise ValueError(f'Data digest mismatch: {name}')
        for suffix in ('.txt', '.observations.json'):
            include(data / (name + suffix), 'data/' + name + suffix)
        for method, source, suffix in [('kim', kim, '_cluster.json'), ('phase2', phase, '_rl_cluster.json')]:
            cache = source / 'cache' / (name + suffix)
            validate_cache(cache)
            include(cache, 'cache/' + method + '/' + cache.name)
        # Cached Kim accident stages are distinct inputs used by the existing trainer.
        if '_ACC_' in name:
            stages = list((kim / 'cache').glob(name + '_acc_refresh_*.json'))
            if len(stages) != 1:
                raise ValueError(f'Expected one Kim accident stage: {name}')
            validate_cache(stages[0])
            include(stages[0], 'cache/kim/' + stages[0].name)
    base = kim / 'cache/RC101_cluster.json'
    validate_cache(base)
    include(base, 'cache/kim/RC101_cluster.json')
    include(checkpoint, 'checkpoints/pure500.pt')
    # Do not copy old run settings verbatim: they contain local absolute paths.
    out.mkdir(parents=True)
    for target, source in selected.items():
        destination = out / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if destination.suffix == '.slurm':
            destination.write_text(destination.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    write(out / 'data/split.json', {k: split[k] for k in ('train', 'test')})
    for method, old in configs.items():
        config = {k: copy.deepcopy(old[k]) for k in ('trainer', 'environment', 'model', 'optimizer')}
        tp = config['trainer']
        if tp['train_instances'] != names:
            raise ValueError(f'Configuration split mismatch: {method}')
        for key in ('data_dir', 'result_dir', 'llm_cache_dir', 'model_load', 'logging'):
            tp.pop(key, None)
        tp.update(test_instances=[], cuda_device_num=0, train_aug8=False)
        config['method'] = method
        if method == 'pure':
            tp['train_batch_size'] = args.pure_batch_size
        if method != 'pure':
            config['llm'] = copy.deepcopy(old['llm'])
            config['cache_dir'] = 'cache/' + method
            tp.update(experience_refresh_epoch=-1, rl_recluster_epoch=None)
        if method == 'phase2':
            config['initial_checkpoint'] = 'checkpoints/pure500.pt'
            config['prompt_version'] = protocol.get('prompt_version', 'original')
        write(out / 'configs' / (method + '.json'), config)
    files = [p for p in out.rglob('*') if p.is_file()]
    # Fail closed for common credential patterns. No .env, key files or request logs are allowlisted.
    for path in files:
        if path.suffix in ('.py', '.json', '.md', '.slurm'):
            text = path.read_text(encoding='utf-8')
            if any(marker in text for marker in ('sk-or-v1-', 'sk-ant-api', 'BEGIN OPENSSH PRIVATE KEY')):
                raise ValueError(f'Possible secret in {path.name}; bundle not archived')
    write(out / 'bundle_manifest.json', dict(format=1, train_count=80, test_files_included=False,
        phase_prompt=protocol.get('prompt_version', 'original'),
        sha256={p.relative_to(out).as_posix(): sha(p) for p in sorted(files)}))
    archive = out.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                z.write(path, Path(out.name) / path.relative_to(out))
    print(json.dumps(dict(folder=str(out), archive=str(archive), files=len(files) + 1,
                          archive_mb=round(archive.stat().st_size / 2**20, 2)), indent=2))


if __name__ == '__main__':
    main()
