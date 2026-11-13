"""Sequential cache-only train80 start-gated Phase2 and frozen test80 evaluation."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'result/rc101_phase2_train80_startgate_20261001'
SOURCE = ROOT/'result/rc101_pure_train80_fresh1000_20260929'
REFERENCE = ROOT/'result/rc101_pure500_train80_phase2_20260930'


def main():
    OUT.mkdir(exist_ok=True)
    status = OUT/'queue.json'
    jobs = [
        ('training', 'run_rc101_pure500_phase2.py', ['--out',str(OUT),
         '--source-run',str(SOURCE),'--reference-run',str(REFERENCE),
         '--reuse-cache','--require-reference-start','--beta','5']),
        ('evaluation','eval_rc101_phase2_test80.py',['--run',str(OUT),
         '--reuse-test',str(REFERENCE/'eval_1000_test80')])]
    try:
        for stage, script, args in jobs:
            status.write_text(json.dumps(dict(stage=stage),indent=2))
            with (OUT/(stage+'_stdout.log')).open('w') as stdout, (OUT/(stage+'_stderr.log')).open('w') as stderr:
                subprocess.run([sys.executable,'-u',str(ROOT/'analysis'/script)]+args,
                               cwd=ROOT,stdout=stdout,stderr=stderr,check=True)
        status.write_text(json.dumps(dict(stage='complete'),indent=2))
    except Exception as exc:
        status.write_text(json.dumps(dict(stage='failed',error=str(exc)),indent=2))
        raise


if __name__ == '__main__':
    main()
