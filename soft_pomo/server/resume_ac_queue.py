"""Serial 12-hour Slurm continuations of the six trusted local A/C checkpoints."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
IMAGE = '/scratch/byk_postech/shared/singularity/pytorch_24.12.sif'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    tmp.replace(p)


def next_task():
    for pool in (20,80,200):
        for variant in ('A','C'):
            suffix = '20261008' if pool == 80 else 'followup'
            name = f'rc101_pure_train{pool}_b128_s42_config{variant}_1000_{suffix}'
            run = ROOT / 'result' / name
            if not (run / 'training/checkpoint-last.pt').exists():
                raise FileNotFoundError(run)
            continuation = run / 'continuation'
            trained = read(continuation / 'progress.json')['epoch'] if (continuation / 'progress.json').exists() else 1000
            for ep in (2000,3000):
                if trained < ep:
                    return dict(name=name, stage='train', target=ep)
                if not (continuation / f'eval_{ep}_test80/summary.json').exists():
                    return dict(name=name, stage='eval', target=ep)
    return None


def submit(task):
    command = ['sbatch','--parsable','--partition=jobs','--nodes=1','--ntasks=1',
        '--cpus-per-task=4','--gres=gpu:1','--time=12:00:00','--signal=USR1@300',
        '--job-name=sj_AC_'+task['stage']+'_'+str(task['target']),
        '--chdir='+str(ROOT),'--output='+str(ROOT/'slurm-%x-%j.log')]
    if os.environ.get('SLURM_JOB_ID'):
        command += ['--dependency=afterok:'+os.environ['SLURM_JOB_ID'], '--kill-on-invalid-dep=yes']
    command += ['--wrap='+shlex.join(['python3','-u',str(Path(__file__).resolve()),'work'])]
    job = subprocess.check_output(command, universal_newlines=True).strip().split(';')[0]
    if not job.isdigit():
        raise ValueError(job)
    write(ROOT/'resume_queue_state.json', dict(status='submitted',job=job,task=task))
    print('Submitted',job,task,flush=True)


def worker(task):
    run = ROOT/'result'/task['name']
    for name, digest in read(run/'manifest.json').items():
        path = run / name.replace('\\','/')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Snapshot changed: '+str(path))
    sys.path.insert(0,str(run/'source'))
    sys.path.insert(0,str(run/'source/server'))
    import train_cached as helper
    config = read(run/'config.json')
    contract = read(run/'training/settings.json')
    if config != contract['config']:
        raise ValueError('Original settings mismatch')
    out = run/'continuation'
    out.mkdir(exist_ok=True)
    trainer = helper.build_trainer(config,run/'source',out)
    last = out/'checkpoint-last.pt'
    checkpoint = (out/f"checkpoint-{task['target']}.pt") if task['stage']=='eval' else (
        last if last.exists() else run/'training/checkpoint-last.pt')
    epoch = helper.restore(trainer,checkpoint,contract)
    import torch
    print('Restored epoch',epoch,'torch',torch.__version__,'GPU',torch.cuda.get_device_name(0),flush=True)
    write(out/'execution.json',dict(target=3000, original_contract=contract,
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)))
    if task['stage']=='eval':
        if epoch != task['target']:
            raise ValueError('Wrong evaluation checkpoint')
        import resume_ac_evaluator as evaluator
        # Evaluator reads immutable inputs from run; writes only beneath continuation.
        evaluator.evaluate_at(run,out,trainer,epoch)
        return
    stop = [False]
    for sig in (signal.SIGTERM,signal.SIGUSR1):
        signal.signal(sig,lambda *a: stop.__setitem__(0,True))
    started = time.monotonic()
    duration = 0
    initial = epoch
    while epoch < task['target'] and not stop[0] and time.monotonic()-started+duration < 11.5*3600:
        begin=time.monotonic()
        trainer._current_epoch=epoch+1
        reward,loss=trainer._train_one_epoch(epoch+1)
        if not math.isfinite(reward) or not math.isfinite(loss):
            raise RuntimeError('Nonfinite training result')
        epoch+=1
        trainer.reward_log.append(reward)
        trainer.loss_log.append(loss)
        duration=time.monotonic()-begin
        if epoch%10==0 or epoch==task['target']:
            helper.save_checkpoint(trainer,epoch,last,contract)
            write(out/'progress.json',dict(epoch=epoch))
        print(f'Epoch {epoch}/3000 reward={reward:.6f} loss={loss:.6f} seconds={duration:.1f}',flush=True)
    helper.save_checkpoint(trainer,epoch,last,contract)
    write(out/'progress.json',dict(epoch=epoch))
    if epoch==task['target']:
        helper.save_checkpoint(trainer,epoch,out/f'checkpoint-{epoch}.pt',contract)
    if epoch<=initial:
        raise RuntimeError('No training progress')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('start','status','work','worker'))
    args=p.parse_args()
    state_path=ROOT/'resume_queue_state.json'
    if args.action=='status':
        print(json.dumps(read(state_path) if state_path.exists() else {},indent=2))
        print('Next:',next_task())
        return
    if args.action=='worker':
        worker(read(state_path)['task'])
        return
    import fcntl
    with (ROOT/'resume_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='start':
            if state_path.exists():
                raise RuntimeError('Queue already exists; inspect status before restarting')
        else:
            state=read(state_path)
            if state['job']!=os.environ.get('SLURM_JOB_ID'):
                raise RuntimeError('Wrong queue owner')
            os.environ['OMP_NUM_THREADS']='4'
            os.environ['MPLBACKEND']='Agg'
            try:
                subprocess.check_call(['srun','singularity','exec','--nv','--bind','/scratch:/scratch',
                    IMAGE,'python','-u',str(Path(__file__).resolve()),'worker'])
            except Exception as exc:
                write(state_path,dict(state,status='failed',error=str(exc)))
                raise
        task=next_task()
        if task is None:
            write(state_path,dict(status='complete'))
        else:
            submit(task)


if __name__=='__main__':
    main()
