"""Queue Config C after the entire existing seed sweep, without reserving a GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

IMAGE = '/scratch/byk_postech/shared/singularity/pytorch_24.12.sif'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(sweep, out):
    source = sweep / 'train80'
    if out.exists():
        raise FileExistsError('Use a new output directory: ' + str(out))
    manifest = read(source / 'bundle_manifest.json')
    for name, digest in manifest['sha256'].items():
        if sha(source / name) != digest:
            raise ValueError('Source bundle changed: ' + name)
    out.mkdir(parents=True)
    for name in manifest['sha256']:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    shutil.copytree(source / 'eval_addon', out / 'eval_addon', dirs_exist_ok=True)
    shutil.copy2(Path(__file__), out / 'queue_reward.py')
    config = read(out / 'configs/pure.json')
    write(out / 'original_config.json', config)
    config['environment'].update(vehicle_penalty=4.0, late_penalty=3.0,
        late_count_penalty=0.0, D_max=8.0, Lt_max=5.0,
        dist_penalty_weight=1.0, time_penalty=0.0)
    config['trainer'].update(epochs=1000, train_batch_size=128, seed=42)
    write(out / 'configs/pure.json', config)
    manifest['sha256']['configs/pure.json'] = sha(out / 'configs/pure.json')
    write(out / 'bundle_manifest.json', manifest)
    write(out / 'reward_protocol.json', dict(
        sweep=str(sweep), objective='J_C = 0.04*K + D/1920 + 0.0025*L',
        note='Weighted preference, not lexicographic. D excludes event multipliers.',
        evaluation='Saved cost field remains Config E; reward is negative Config C.',
        batch=128, seed=42, epochs=1000, api=False))
    write(out / 'queue_state.json', dict(status='prepared'))


def submit(out, mode, dependency=None):
    command = ['sbatch', '--parsable', '--partition=jobs', '--nodes=1', '--ntasks=1',
               '--cpus-per-task=' + ('1' if mode == 'watch' else '4'),
               '--time=' + ('00:05:00' if mode == 'watch' else '12:00:00'),
               '--chdir=' + str(out), '--job-name=sj_t80_s42_C_' + mode,
               '--output=' + str(out / 'slurm-%x-%j.log')]
    if mode != 'watch':
        command += ['--gres=gpu:1', '--signal=USR1@300']
    if dependency:
        if not str(dependency).isdigit():
            raise ValueError('Invalid dependency')
        command += ['--dependency=afterok:' + str(dependency), '--kill-on-invalid-dep=yes']
    wrapped = shlex.join(['python3', '-u', str(out / 'queue_reward.py'), mode, '--out', str(out)])
    command += ['--wrap=' + wrapped]
    job = subprocess.check_output(command, universal_newlines=True).strip().split(';')[0]
    if not job.isdigit():
        raise ValueError('Unexpected sbatch response: ' + job)
    write(out / 'queue_state.json', dict(status='submitted', stage=mode, job=job))
    print('Submitted {} job {} (dependency {})'.format(mode, job, dependency), flush=True)


def watch(out):
    sweep = Path(read(out / 'reward_protocol.json')['sweep'])
    state = read(sweep / 'queue_state.json')
    if state['status'] == 'complete':
        # Check the results too, rather than relying solely on the queue flag.
        sys.path.insert(0, str(sweep))
        import seed_queue
        if seed_queue.next_task(sweep, read(sweep / 'plan.json')) is not None:
            raise RuntimeError('Sweep says complete but results are incomplete')
        submit(out, 'train', os.environ.get('SLURM_JOB_ID'))
    elif state['status'] in ('submitted', 'running'):
        job = str(state['job'])
        active = subprocess.check_output(['squeue', '-h', '-j', job, '-o', '%i'],
                                         universal_newlines=True).strip()
        if job not in active.split():
            raise RuntimeError('Sweep state refers to an inactive job; inspect before scheduling')
        submit(out, 'watch', job)
    else:
        raise RuntimeError('Existing sweep is stopped or failed: ' + str(state))


def work(out, mode):
    run = out / 'runs/pure128_configC_seed42'
    base = ['srun', 'singularity', 'exec', '--nv', '--bind', '/scratch:/scratch',
            IMAGE, 'python', '-u']
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MPLBACKEND'] = 'Agg'
    for key in ('OPENAI_API_KEY', 'OPENROUTER_API_KEY', 'ANTHROPIC_API_KEY'):
        os.environ.pop(key, None)
    if mode == 'train':
        command = base + ['server/train_cached.py', '--method', 'pure', '--out', str(run),
                         '--seed', '42', '--epochs', '1000', '--max-hours', '11.5']
        previous = 0
        if (run / 'checkpoint-last.pt').exists():
            previous = read(run / 'status.json')['epoch']
            command += ['--resume']
        subprocess.check_call(command, cwd=str(out))
        status = read(run / 'status.json')
        if status['epoch'] <= previous or status['status'] not in ('complete', 'needs_resume'):
            raise RuntimeError('No training progress: ' + str(status))
        submit(out, 'eval' if status['status'] == 'complete' else 'train', os.environ['SLURM_JOB_ID'])
    else:
        command = base + ['eval_addon/eval_pure.py', '--run', str(run), '--epoch', '1000']
        if (run / 'eval_1000_test80/protocol.json').exists():
            command += ['--resume']
        subprocess.check_call(command, cwd=str(out))
        rows = read(run / 'eval_1000_test80/results.json')
        write(run / 'eval_1000_test80/reward_comparison.json', [dict(
            name=r['name'], cost_C=-r['reward'], cost_E=r['cost']) for r in rows])
        write(out / 'queue_state.json', dict(status='complete', epoch=1000, evaluated=len(rows)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('start', 'prepare', 'watch', 'train', 'eval'))
    parser.add_argument('--sweep', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.action in ('start', 'prepare'):
        if args.sweep is None:
            parser.error('--sweep is required')
        prepare(args.sweep.resolve(), out)
        if args.action == 'start':
            watch(out)
    else:
        try:
            if args.action == 'watch':
                watch(out)
            else:
                work(out, args.action)
        except Exception as exc:
            write(out / 'queue_state.json', dict(status='failed', error=repr(exc), stage=args.action))
            raise
