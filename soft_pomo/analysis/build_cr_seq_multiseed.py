"""Package matched C/R sequential runs using the frozen corrected RC sources."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'server_export/rc101_seq_train20_late15_seeds42_46_20261022'
OUT = ROOT / 'server_export/cr101_seq_l5_10_15_20_s42_46_20261023'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def main():
    src = SOURCE / 'seed42'
    manifest = read(src / 'bundle_manifest.json')['sha256']
    for name, expected in manifest.items():
        assert digest(src / name) == expected, name
    OUT.mkdir(exist_ok=False)
    tasks = []
    for case, k0, t0, horizon, strength in (
        ('c101', 10, 828.94, 1236, 2.0),
        ('r101', 19, 1650.80, 230, 0.5),
    ):
        prior = ROOT / 'result/bks_normalized_train20_20261013' / case
        split = read(prior / 'split.json')
        assert len(split['train']) == 20 and len(split['test']) == 80
        assert not {x['name'] for x in split['train']} & {x['name'] for x in split['test']}
        for entry in split['train'] + split['test']:
            assert digest(prior / 'data' / (entry['name'] + '.txt')) == entry['sha256']
        for late in (5, 10, 15, 20):
            for seed in range(42, 47):
                relative = f'{case}/late{late}/seed{seed}'
                dest = OUT / relative
                dest.mkdir(parents=True)
                shutil.copytree(prior / 'data', dest / 'data')
                shutil.copy2(prior / 'split.json', dest / 'split.json')
                for name in manifest:
                    if name.endswith('.py') and not name.startswith('data/'):
                        target = dest / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src / name, target)
                cfg = copy.deepcopy(read(src / 'configs/pure.json'))
                cfg['trainer'].update(seed=seed, train_instances=[x['name'] for x in split['train']])
                cfg['environment'].update(vehicle_penalty=100/k0, TT_max=t0/horizon,
                                          Lt_max=t0/horizon, late_penalty=late)
                write(dest / 'configs/pure.json', cfg)
                write(dest / 'provenance.json', dict(
                    train_seed=seed, instance=case, event_increment_strength=strength,
                    objective=f'K/{k0} + (travel_minutes + {late}*late_minutes)/{t0}',
                    source=SOURCE.name, data_source=str(prior.relative_to(ROOT)),
                    training='fresh1000; save every10; pure RL; no LLM',
                    evaluation='fixed test80; seed42+index; all100 starts; greedy; aug8',
                    comparison_warning='Corrected depot-first sources; do not pool old pre-fix runs'))
                write(dest / 'bundle_manifest.json', {'sha256': {
                    p.relative_to(dest).as_posix(): digest(p)
                    for p in dest.rglob('*') if p.is_file()}})
                tasks.append(relative)
    assert len(tasks) == 40
    (OUT / 'tasks.txt').write_text('\n'.join(tasks)+'\n', encoding='ascii', newline='\n')
    shutil.copy2(SOURCE / 'eval_pure_tt_bundle.py', OUT / 'eval_pure_tt_bundle.py')
    slurm = (SOURCE / 'seq_late15_seeds.slurm').read_text()
    slurm = slurm.replace('sj_rc_seq_l15', 'sj_cr_seq_sweep').replace('--array=0-4%1', '--array=0-39%2')
    slurm = slurm.replace('seed=$((42+SLURM_ARRAY_TASK_ID))\nbase=$PWD\nbundle="$base/seed$seed"',
        'base=$PWD\nrelative=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$base/tasks.txt")\n'
        '[[ -n "$relative" ]]\nbundle="$base/$relative"\nseed=${relative##*seed}')
    assert 'seed=$((42+' not in slurm
    (OUT / 'sweep.slurm').write_text(slurm, encoding='ascii', newline='\n')
    # Depend on all currently queued project jobs, not unrelated users' workloads.
    submit = '''#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
[[ ! -e submitted_job.txt ]] || { echo "Already submitted: $(cat submitted_job.txt)"; exit 1; }
deps=$(squeue -h -u "$USER" -o '%F %j' | awk '$2 ~ /^sj_/ {print $1}' | sort -u | paste -sd: -)
args=()
if [[ -n "$deps" ]]; then args+=(--dependency="afterany:$deps"); fi
job=$(sbatch --parsable "${args[@]}" sweep.slurm)
printf '%s\\n' "$job" > submitted_job.txt
printf 'Submitted %s; dependencies=%s; max simultaneous GPUs=2\\n' "$job" "${deps:-none}"
squeue -u "$USER" -o '%.18i %.24j %.10T %.30E'
'''
    (OUT / 'submit.sh').write_text(submit, encoding='ascii', newline='\n')
    archive = OUT.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(OUT, arcname=OUT.name)
    Path(str(archive)+'.sha256').write_text(digest(archive)+'  '+archive.name+'\n', encoding='ascii', newline='\n')
    print(archive)
    print('40 fresh runs, frozen RC sources, verified C/R split hashes, one array %2')


if __name__ == '__main__':
    main()
