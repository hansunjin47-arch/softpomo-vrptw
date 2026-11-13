"""Evaluate saved epoch-500 models without modifying the training run."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    source = (root / sys.argv[1]).resolve()
    status = json.loads((source / 'status.json').read_text(encoding='utf-8'))
    pure = '--pure' in sys.argv[2:]
    output = source / (('eval_pure500_' if pure else 'eval_500_') + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(exist_ok=False)
    manifest = {'source': str(source), 'jobs': []}
    print(str(output), flush=True)
    for job in status['jobs']:
        family = job['family']
        checkpoint_root = (root / 'result' / f'config_E_{family}_final_weightedsum_b32_aug8') if pure else source / family
        checkpoints = list(checkpoint_root.rglob('checkpoint-500.pt'))
        if len(checkpoints) != 1:
            raise RuntimeError(f'{family}: expected exactly one epoch-500 checkpoint')
        checkpoint = checkpoints[0]
        command = job['command'].copy()
        command[command.index('--result-dir') + 1] = str(output / family)
        command.extend(['--test-only', '--resume', str(checkpoint)])
        if pure:
            command.append('--no-llm')
        record = {'family': family, 'command': command,
                  'checkpoint_sha256': digest(checkpoint),
                  'started': datetime.datetime.now().isoformat()}
        print(f'{family}: starting', flush=True)
        with (output / f'{family}_out.log').open('x', encoding='utf-8') as stdout, (output / f'{family}_err.log').open('x', encoding='utf-8') as stderr:
            result = subprocess.run(command, cwd=root, stdout=stdout, stderr=stderr,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
        record['returncode'] = result.returncode
        record['finished'] = datetime.datetime.now().isoformat()
        record['checkpoint_unchanged'] = digest(checkpoint) == record['checkpoint_sha256']
        manifest['jobs'].append(record)
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'{family}: exit={result.returncode}', flush=True)
        if result.returncode or not record['checkpoint_unchanged']:
            raise RuntimeError(f'{family}: evaluation failed; inspect logs')


if __name__ == '__main__':
    main()
