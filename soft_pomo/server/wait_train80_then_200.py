"""Linux login-node watcher: submit train200 once both train80 variants finish."""
import argparse
import getpass
import json
from pathlib import Path
import subprocess
import time


def completed(root):
    for variant in ('A','C'):
        path=root/variant/'runs/pure/status.json'
        if not path.exists():
            return False
        state=json.loads(path.read_text())
        if state.get('status')!='complete' or state.get('epoch',0)<2000:
            return False
        if not (path.parent/'checkpoint-2000.pt').exists():
            return False
    return True


def main():
    import fcntl
    parser=argparse.ArgumentParser()
    parser.add_argument('--previous',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    previous=args.previous.resolve(strict=True)
    lock=(root/'after80.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    marker=root/'after80_submission.json'
    if marker.exists():
        raise RuntimeError('Submission already recorded or ambiguous; inspect after80_submission.json')
    print('Waiting for train80 A and C, epoch 2000:',previous,flush=True)
    deadline=time.monotonic()+14*86400
    while time.monotonic()<deadline:
        if completed(previous):
            # Wait for the final Slurm job to release its GPU as well.
            try:
                active=subprocess.run(['squeue','-u',getpass.getuser(),'-h','-o','%j'],
                    text=True,capture_output=True,timeout=30)
            except (OSError, subprocess.TimeoutExpired) as error:
                print('Queue query failed; no submission; retry in 60s:',error,flush=True)
                time.sleep(60)
                continue
            if active.returncode:
                print('Queue query failed; no submission; retry in 60s:',active.stderr.strip(),flush=True)
            elif 'sj_pure_tt80' not in active.stdout.split():
                break
        time.sleep(60)
    else:
        raise TimeoutError('Train80 did not finish within 14 days; nothing submitted')
    marker.write_text(json.dumps({'status':'submitting','previous':str(previous)}))
    result=subprocess.run(['sbatch','--parsable','pure_tt200.slurm','A'],cwd=root,text=True,capture_output=True)
    marker.write_text(json.dumps({'status':'submitted' if result.returncode==0 else 'failed',
        'job':result.stdout.strip(),'stderr':result.stderr,'previous':str(previous)},indent=2))
    result.check_returncode()
    print('Submitted train200 A:',result.stdout.strip(),flush=True)


if __name__=='__main__':
    main()
