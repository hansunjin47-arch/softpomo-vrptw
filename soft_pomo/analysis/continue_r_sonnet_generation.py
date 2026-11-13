"""Generate remaining R caches, auditing each instance before advancing."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    out = Path(sys.argv[1]).resolve()
    scripts = Path(__file__).resolve().parent
    status = dict(state='running', completed=[], current=None, started=time.strftime('%Y-%m-%d %H:%M:%S'))

    def save():
        target = out/'generation_status.json'
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(status, indent=2), encoding='utf-8')
        temp.replace(target)

    try:
        # R102 pilot must have passed the actual-loader audit before expansion.
        audit = json.loads((out/'audit_R102.json').read_text())
        assert audit['originals_unchanged'] and audit['instances'][0]['temporal_selection']
        for number in list(range(103,113))+[101]:
            status['current'] = f'R{number}'
            save()
            subprocess.run([sys.executable, '-u', str(scripts/'generate_kim500_openrouter.py'), str(out), '--benchmark', 'r1', '--instances', str(number), '--confidence-only'], check=True)
            subprocess.run([sys.executable, '-u', str(scripts/'audit_feedback_cache.py'), str(out), f'R{number}'], check=True)
            status['completed'].append(f'R{number}')
            save()
        status.update(state='complete', current=None)
    except Exception as exc:
        status.update(state='failed', error=type(exc).__name__)
        raise
    finally:
        status['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        save()


if __name__ == '__main__':
    main()
