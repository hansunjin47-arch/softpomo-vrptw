"""Serial Slurm queue; run with host Python 3 (no ML dependencies)."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
IMAGE = '/scratch/byk_postech/shared/singularity/pytorch_24.12.sif'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    os.replace(str(temp), str(path))


def next_task(root, plan):
    for case in plan['cases']:
        bundle = root / case['bundle']
        run = bundle / 'runs' / ('pure128_4000_seed' + str(case['seed']))
        status = read(run / 'status.json') if (run / 'status.json').exists() else {}
        if status.get('status') != 'complete':
            if run.exists() and any(run.iterdir()) and not (run / 'checkpoint-last.pt').exists():
                raise RuntimeError('Partial run without checkpoint: ' + str(run))
            return dict(case=case, stage='train', resume=(run / 'checkpoint-last.pt').exists())
        if status.get('epoch') != 4000 or status.get('target') != 4000:
            raise RuntimeError('Unexpected training target: ' + str(run))
        for epoch in (1000, 2000, 3000, 4000):
            out = run / ('eval_%s_test80' % epoch)
            done = read(out / 'status.json') if (out / 'status.json').exists() else {}
            if done.get('status') == 'complete' and (out / 'summary.json').exists():
                continue
            return dict(case=case, stage='eval', epoch=epoch,
                        resume=(out / 'protocol.json').exists())
    return None


def submit(root, task, dependency=None):
    case = task['case']
    name = 'sj_t%s_s%s_%s' % (case['pool'], case['seed'], task['stage'])
    command = ['sbatch', '--parsable', '--chdir=' + str(root), '--job-name=' + name]
    if dependency:
        command.append('--dependency=afterok:' + dependency)
    command.append(str(root / 'queue.slurm'))
    job = subprocess.check_output(command, universal_newlines=True).strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Unrecognized sbatch reply: ' + job)
    write(root / 'queue_state.json', dict(job=job, task=task, status='submitted'))
    print('Submitted %s: %s' % (job, name), flush=True)


def work(root, task):
    case = task['case']
    bundle = root / case['bundle']
    run = 'runs/pure128_4000_seed' + str(case['seed'])
    command = ['singularity', 'exec', '--nv', '--bind', '/scratch:/scratch', IMAGE,
               'python', '-u']
    if task['stage'] == 'train':
        command += ['server/train_cached.py', '--method', 'pure', '--seed', str(case['seed']),
                    '--epochs', '4000', '--max-hours', '11.5', '--out', run]
    else:
        command += ['eval_addon/eval_pure.py', '--run', run, '--epoch', str(task['epoch'])]
    if task['resume']:
        command.append('--resume')
    previous = read(bundle / run / 'status.json').get('epoch', 0) if task['stage'] == 'train' and task['resume'] else 0
    subprocess.check_call(['srun'] + command, cwd=str(bundle))
    if task['stage'] == 'train':
        status = read(bundle / run / 'status.json')
        if status['status'] not in ('complete', 'needs_resume') or status['epoch'] <= previous:
            raise RuntimeError('Training made no progress; queue stopped')
    else:
        status = read(bundle / run / ('eval_%s_test80' % task['epoch']) / 'status.json')
        if status['status'] != 'complete':
            raise RuntimeError('Evaluation incomplete; queue stopped')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'work', 'status'])
    args = parser.parse_args()
    plan = read(ROOT / 'plan.json')
    if args.action == 'status':
        print(json.dumps(read(ROOT / 'queue_state.json') if (ROOT / 'queue_state.json').exists() else {}, indent=2))
        print('Next:', next_task(ROOT, plan))
        return
    # Prevent duplicate launches, including two simultaneous start commands.
    import fcntl
    with (ROOT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = ROOT / 'queue_state.json'
        if args.action == 'start':
            if state_path.exists():
                raise RuntimeError('Queue already started. Inspect state/logs before recovery.')
        else:
            state = read(state_path)
            if state.get('job') != os.environ.get('SLURM_JOB_ID') or state['status'] != 'submitted':
                raise RuntimeError('Job does not own queue state')
            if (ROOT / 'STOP').exists():
                write(state_path, dict(status='stopped', job=state['job']))
                return
            try:
                write(state_path, dict(state, status='running'))
                work(ROOT, state['task'])
            except Exception as exc:
                write(state_path, dict(state, status='failed', error=str(exc)))
                raise
        if (ROOT / 'STOP').exists():
            write(state_path, dict(status='stopped'))
            return
        task = next_task(ROOT, plan)
        if task is None:
            write(state_path, dict(status='complete'))
            print('All training and evaluations complete')
        else:
            submit(ROOT, task, os.environ.get('SLURM_JOB_ID') if args.action == 'work' else None)


if __name__ == '__main__':
    main()
