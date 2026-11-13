"""Run validated v5 cache generation then local offline Phase2, with stage logs."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/rc101_ontology_v5_train20_b128_seed43'


def main():
    marker = OUT / 'launcher_status.json'
    def status(**value):
        marker.write_text(json.dumps(value, indent=2), encoding='utf-8')
    for stage, command in [
        ('calling', [sys.executable, '-u', str(ROOT / 'analysis/run_with_openrouter_key.py'),
                     str(ROOT / 'analysis/run_ontology_b128_train20.py'), 'call']),
        ('training', [sys.executable, '-u', str(ROOT / 'analysis/run_ontology_b128_train20.py'), 'train'])]:
        status(stage=stage)
        with (OUT / (stage + '_stdout.log')).open('a', encoding='utf-8') as stdout, \
             (OUT / (stage + '_stderr.log')).open('a', encoding='utf-8') as stderr:
            result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        if result.returncode:
            status(stage='failed', failed_stage=stage, returncode=result.returncode)
            return
    status(stage='finished', pipeline=json.loads((OUT / 'pipeline.json').read_text()))


if __name__ == '__main__':
    main()
