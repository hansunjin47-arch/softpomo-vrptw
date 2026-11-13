"""Generate train80 news-only caches and compare; no training or test calls."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'result/rc101_phase2_train80_news_only_20261001'


def main():
    OUT.mkdir(exist_ok=True)
    jobs = [
        ('training', 'run_rc101_pure500_phase2.py', [
            '--out', str(OUT), '--news-only', '--cache-only', '--beta', '5',
            '--source-run', str(ROOT/'result/rc101_pure_train80_fresh1000_20260929'),
            '--reference-run', str(ROOT/'result/rc101_pure500_train80_phase2_20260930'),
            '--kim-evaluation', str(ROOT/'result/rc101_kim_train80_fresh1000_20260929/eval_1000_test80/summary.json')]),
        ('comparison', 'compare_phase2_news_cache.py', [])]
    status = OUT/'queue.json'
    try:
        for stage, script, args in jobs:
            status.write_text(json.dumps(dict(stage=stage)))
            with (OUT/(stage+'_stdout.log')).open('w') as stdout, (OUT/(stage+'_stderr.log')).open('w') as stderr:
                subprocess.run([sys.executable, '-u', str(ROOT/'analysis'/script)] + args,
                               cwd=ROOT, stdout=stdout, stderr=stderr, check=True)
        status.write_text(json.dumps(dict(stage='complete')))
    except Exception as exc:
        status.write_text(json.dumps(dict(stage='failed', error=str(exc))))
        raise


if __name__ == '__main__':
    main()
