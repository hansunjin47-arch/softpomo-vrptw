"""Copy submitted Kim code/config exactly; replace only guidance caches."""
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'server'))
import train_cached as io

OLD = ROOT / 'server_export/kim_l5_rc_r_c_s42_46_20261026'
OUT = ROOT / 'server_export/kim_rl_example_l5_rc_r_c_s42_46_20261027'


def main():
    from generate_kim_rl_example_all import audit
    audit('train')
    audit('test')
    OUT.mkdir(exist_ok=False)
    tasks = io.read(OLD / 'manifest.json')['tasks']
    assert len(tasks) == 15 and len(set(tasks)) == 15
    for relative in tasks:
        src, dest = OLD / relative, OUT / relative
        io.verify_bundle(src)
        case, seed_name = relative.split('/')
        short = case.replace('101', '')
        original_manifest = io.read(src / 'bundle_manifest.json')['sha256']
        for name in original_manifest:
            if name.startswith('cache/') or name == 'provenance.json':
                continue
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / name, target)
        config = io.read(dest / 'configs/kim.json')
        assert config == io.read(src / 'configs/kim.json')
        assert config['trainer']['epochs'] == 1000
        assert config['trainer']['seed'] == int(seed_name[4:])
        assert 'initial_checkpoint' not in config
        assert config.get('accident_guidance', 'zero_then_acc') == 'zero_then_acc'
        protocols = {}
        for split in ('train', 'test'):
            source = ROOT / f'result/kim_rl_example_{short}_{split}_20261027'
            protocols[split] = io.read(source / 'protocol.json')
            assert protocols[split]['accident_guidance'] == 'zero_then_acc'
            assert protocols[split]['names'] == [r['name'] for r in io.read(src / 'split.json')[split]]
            for filename, digest in io.read(source / 'cache_manifest.json')['files'].items():
                cache = source / 'cache' / filename
                assert io.sha(cache) == digest
                io.validate_cache(cache)
                target = dest / 'cache' / split / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cache, target)
            provenance = dict(protocols[split], checkpoint_independent=False,
                              target_checkpoint_independent=True,
                              reference_usage='Fixed separate RL example, identical across train/test and seeds')
            io.write(dest / 'cache' / split / 'provenance.json', provenance)
        io.write(dest / 'provenance.json', dict(
            method='Kim plus BASE prompt reference and fixed separate RL example; fresh 1000',
            source_bundle_sha256=io.sha(src / 'bundle_manifest.json'),
            code_and_config_unchanged=True, cache_protocols=protocols,
            evaluation='100 starts x 8 augmentations; greedy; test80; no target test RL solution',
            accident_guidance='zero_then_acc; BASE is prompt reference only'))
        io.write(dest / 'bundle_manifest.json', {'sha256': {
            p.relative_to(dest).as_posix(): io.sha(p) for p in dest.rglob('*') if p.is_file()}})
        io.verify_bundle(dest)
    shutil.copy2(OLD / 'eval_kim.py', OUT / 'eval_kim.py')
    (OUT / 'tasks.txt').write_text('\n'.join(tasks) + '\n', encoding='ascii', newline='\n')
    slurm = (OLD / 'sweep.slurm').read_text(encoding='utf-8')
    slurm = slurm.replace('sj_kim_l5_all', 'sj_kim_rl_example')
    slurm = slurm.replace('#SBATCH --array=0-14%2\n', '')
    slurm = slurm.replace('slurm-%x-%A_%a.log', 'slurm-%x-%j.log')
    slurm = slurm.replace('base=$PWD\n', 'base=$PWD\nindex=${1:?Missing task index}\n[[ "$index" =~ ^[0-9]+$ && "$index" -lt 15 ]]\n')
    slurm = slurm.replace('SLURM_ARRAY_TASK_ID+1', 'index+1')
    slurm = slurm.replace('${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}', '${SLURM_JOB_ID}')
    assert 'SLURM_ARRAY' not in slurm and '#SBATCH --gres=gpu:1' in slurm
    (OUT / 'task.slurm').write_text(slurm, encoding='ascii', newline='\n')
    (OUT / 'submit.sh').write_text((ROOT / 'analysis/submit_kim_rl_example_lanes.sh').read_text(), encoding='ascii', newline='\n')
    io.write(OUT / 'manifest.json', dict(tasks=tasks, max_gpus=2, gpus_per_task=1,
        scheduling='15 non-array jobs in two afterany chains; wait for existing sj_kim_* jobs',
        epochs=1000, sha256={p.relative_to(OUT).as_posix():io.sha(p) for p in OUT.rglob('*') if p.is_file()}))
    archive = OUT.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(OUT, arcname=OUT.name)
    Path(str(archive)+'.sha256').write_text(io.sha(archive)+'  '+archive.name+'\n', encoding='ascii', newline='\n')
    print(archive, flush=True)


if __name__ == '__main__':
    main()
