"""Reproduce sequential Pure RL bundles with only vehicle cost changed to 100 min."""
import copy
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'server'))
import train_cached as io

RC = ROOT/'server_export/rc101_seq_train20_late5_seeds42_46_20261023'
CR = ROOT/'server_export/cr101_seq_l5_10_15_20_s42_46_20261023'
OUT = ROOT/'server_export/seq_pure_v100_l5_rc_r_s42_46_20261028'


def main():
    OUT.mkdir(exist_ok=False)
    tasks = []
    for seed in range(42,47):
        for case, source, denominator in [('rc101', RC/f'seed{seed}', 1696.95),
                                         ('r101', CR/f'r101/late5/seed{seed}', 1650.80)]:
            io.verify_bundle(source)
            relative = f'{case}/seed{seed}'
            dest = OUT/relative
            manifest = io.read(source/'bundle_manifest.json')['sha256']
            for name in manifest:
                assert not name.startswith(('runs/', 'eval_')), 'Never bundle previous results'
                target = dest/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source/name, target)
            original = io.read(source/'configs/pure.json')
            config = copy.deepcopy(original)
            assert config['trainer']['seed'] == seed and config['trainer']['epochs'] == 1000
            assert config['environment']['late_penalty'] == 5
            assert config['environment']['time_penalty'] == 1 and config['environment']['dist_penalty_weight'] == 0
            assert 'initial_checkpoint' not in config
            config['environment']['vehicle_penalty'] = 100 * config['environment']['problem_size'] / denominator
            io.write(dest/'configs/pure.json', config)
            check = copy.deepcopy(config)
            check['environment']['vehicle_penalty'] = original['environment']['vehicle_penalty']
            assert check == original
            for name, digest in manifest.items():
                if name not in ('configs/pure.json', 'provenance.json'):
                    assert io.sha(dest/name) == digest
            io.write(dest/'provenance.json', dict(source=str(source), source_manifest_sha256=io.sha(source/'bundle_manifest.json'),
                change='Only environment.vehicle_penalty changed; all training/model/evaluation/data settings preserved',
                old_vehicle_penalty=original['environment']['vehicle_penalty'],
                new_vehicle_penalty=config['environment']['vehicle_penalty'],
                objective=f'(100*K + TT + 5*L)/{denominator}; unserved penalty unchanged',
                initialization='fresh seed; no checkpoint', evaluation='100 starts x 8 augmentation; greedy best; test80'))
            io.write(dest/'bundle_manifest.json', {'sha256':{
                p.relative_to(dest).as_posix():io.sha(p) for p in dest.rglob('*')
                if p.is_file() and p.name != 'bundle_manifest.json'}})
            io.verify_bundle(dest)
            tasks.append(relative)
    shutil.copy2(CR/'eval_pure_tt_bundle.py', OUT/'eval_pure_tt_bundle.py')
    (OUT/'tasks.txt').write_text('\n'.join(tasks)+'\n', encoding='ascii', newline='\n')
    slurm = (CR/'sweep.slurm').read_text()
    slurm = slurm.replace('sj_cr_seq_sweep','sj_pure_v100_l5')
    slurm = slurm.replace('#SBATCH --array=0-39%2\n','')
    slurm = slurm.replace('slurm-%x-%A_%a.log','slurm-%x-%j.log')
    slurm = slurm.replace('base=$PWD\n','base=$PWD\nindex=${1:?Missing task index}\n[[ "$index" =~ ^[0-9]+$ && "$index" -lt 10 ]]\n')
    slurm = slurm.replace('SLURM_ARRAY_TASK_ID+1','index+1')
    slurm = slurm.replace('${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}','${SLURM_JOB_ID}')
    assert 'SLURM_ARRAY' not in slurm and '#SBATCH --gres=gpu:1' in slurm
    (OUT/'task.slurm').write_text(slurm, encoding='ascii', newline='\n')
    (OUT/'submit.sh').write_text((ROOT/'analysis/submit_seq_v100_lanes.sh').read_text(), encoding='ascii', newline='\n')
    io.write(OUT/'manifest.json',dict(tasks=tasks, max_gpus=2, vehicle_cost_minutes=100, late=5,
        epochs=1000, sha256={p.relative_to(OUT).as_posix():io.sha(p) for p in OUT.rglob('*') if p.is_file()}))
    archive = OUT.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(OUT, arcname=OUT.name)
    Path(str(archive)+'.sha256').write_text(io.sha(archive)+'  '+archive.name+'\n',encoding='ascii',newline='\n')
    print('BUILT',archive,'bytes',archive.stat().st_size)
    print('\n'.join(tasks))


if __name__ == '__main__':
    main()
