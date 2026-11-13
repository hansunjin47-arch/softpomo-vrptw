"""Run only the requested GPT train-cache and Phase2 training stages."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/rc101_phase2_online_gpt_train20_v2_2_20261018'
RUNNER = OUT / 'source/run_phase2_online_v2.py'
STATUS = OUT / 'train5_10_queue_status.json'


def status(**values):
    values['updated_at'] = datetime.now(timezone.utc).isoformat()
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(values, indent=2), encoding='utf-8')
    temporary.replace(STATUS)


def main():
    jobs = [(w, stage) for w in (5, 10) for stage in ('prepare', 'call-train')]
    jobs += [(w, stage) for w in (5, 10) for stage in ('smoke', 'train')]
    for weight, stage in jobs:
        if (OUT / 'STOP').exists():
            status(status='stopped', weight=weight, stage=stage)
            return
        status(status='running', weight=weight, stage=stage, evaluation=False)
        print(f'START lambda={weight} stage={stage}', flush=True)
        try:
            subprocess.run([sys.executable, '-u', str(RUNNER), stage,
                            '--weight', str(weight), '--allow-api'],
                           cwd=ROOT, check=True)
        except Exception as exc:
            status(status='failed', weight=weight, stage=stage, error=str(exc))
            raise
    status(status='complete', weights=[5, 10], evaluation=False)


if __name__ == '__main__':
    main()
