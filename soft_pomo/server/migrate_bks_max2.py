"""Save the active legacy training job and replace its chain with a capped array."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True, type=int)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    old = root / 'rc101_bks_train40_80_late5_10_20261014'
    new = root / 'rc101_bks_train40_80_late20_20261015'
    script = root / 'bks_remaining_max2.slurm'
    assert script.is_file()
    job = str(args.job)
    details = command('scontrol', 'show', 'job', '-o', job)
    assert 'JobState=RUNNING' in details, details
    assert 'Command=' + str(old / 'bks_pool_sensitivity.slurm') + ' 1 train' in details, details
    queue = command('squeue', '-u', 'byk_postech', '-h', '-o', '%i|%j')
    ours = [line.split('|')[0] for line in queue.splitlines()
            if line.split('|')[-1].startswith('sj_bks_pool') or line.split('|')[-1] == 'sj_bks_max2']
    assert ours == [job], 'Other BKS jobs present; inspect before migration: ' + str(ours)
    for parent, cases in ((old, ['train40_late10', 'train80_late5', 'train80_late10']),
                          (new, ['train40_late20', 'train80_late20'])):
        assert (parent / 'eval_pure_tt_bundle.py').is_file()
        for case in cases:
            assert (parent / case / 'server/train_cached.py').is_file()
            assert (parent / case / 'configs/pure.json').is_file()
    assert json.loads((old / 'train40_late5/eval_1000_test80_gpu/status.json').read_text())['completed'] == 80
    run = old / 'train40_late10/runs/pure'
    before = json.loads((run / 'status.json').read_text())
    assert before['status'] == 'training'
    record = root / 'bks_max2_migration.json'
    # Exclusive creation prevents accidental duplicate arrays on a repeated command.
    with record.open('x') as handle:
        json.dump(dict(stage='preflight_complete', old_job=job), handle)

    def report(stage, **kwargs):
        record.write_text(json.dumps(dict(stage=stage, old_job=job, **kwargs), indent=2))
        print(stage, kwargs, flush=True)

    array = command('sbatch', '--parsable', '--hold', '--dependency=afterany:' + job, str(script)).split(';')[0]
    assert array.isdigit(), array
    report('array_held', array_job=array)
    # Stop only the batch shell so it cannot submit another legacy-chain job.
    command('scancel', '--batch', '--signal=STOP', job)
    command('scancel', '--signal=USR1', job)
    report('saving_checkpoint', array_job=array)
    for _ in range(180):
        try:
            state = json.loads((run / 'status.json').read_text())
        except (FileNotFoundError, ValueError):
            state = {}
        checkpoint = run / 'checkpoint-last.pt'
        if state.get('status') in ('needs_resume', 'complete') and state.get('epoch', 0) >= before['epoch'] and checkpoint.is_file():
            assert checkpoint.stat().st_size > 0
            report('checkpoint_saved', array_job=array, epoch=state['epoch'])
            command('scancel', job)
            # afterany also waits for old-job cleanup and GPU release.
            command('scontrol', 'release', array)
            report('array_released', array_job=array, saved_epoch=state['epoch'], max_concurrent=2)
            return
        time.sleep(5)
    report('save_timeout_keep_array_held', array_job=array)
    raise RuntimeError('Checkpoint not confirmed. Show migration record; do not force-cancel or resubmit.')


if __name__ == '__main__':
    main()
