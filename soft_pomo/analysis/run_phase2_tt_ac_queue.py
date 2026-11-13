"""Sequential corrected A/C cache, Phase2 training and GPU evaluation queue."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / 'result/phase2_tt_ac_queue_20261010'


def status(**value):
    QUEUE.mkdir(parents=True, exist_ok=True)
    (QUEUE / 'status.json').write_text(json.dumps(value, indent=2), encoding='utf-8')


def run(variant, stage, script, *args):
    status(variant=variant, stage=stage)
    print('STAGE', variant, stage, flush=True)
    subprocess.run([sys.executable, '-u', str(ROOT / 'analysis' / script), *args, '--variant', variant], cwd=ROOT, check=True)


def main():
    for v in ('A', 'C'):
        run(v, 'prepare_train', 'prepare_phase2_tt_train20.py', 'prepare')
        run(v, 'call_train', 'call_phase2_tt_cache.py', '--allow-api')
    for v in ('A', 'C'):
        run(v, 'train', 'prepare_phase2_tt_train20.py', 'train')
        from travel_time_prompt_contract import VERSION
        state = json.loads((ROOT / f'result/rc101_phase2_train20_b128_s42_config{v}_{VERSION}/training/status.json').read_text())
        if state['status'] != 'complete' or state['epoch'] != 1000:
            raise RuntimeError(f'{v} training needs resume: {state}')
    for v in ('A', 'C'):
        run(v, 'prepare_test', 'eval_phase2_tt_test80.py', 'prepare')
        run(v, 'call_test', 'call_phase2_tt_cache.py', '--test', '--allow-api')
        run(v, 'evaluate_gpu', 'eval_phase2_tt_test80.py', 'evaluate', '--device', 'cuda')
    status(stage='complete', variants=['A', 'C'])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status(stage='failed', error=repr(exc))
        raise
