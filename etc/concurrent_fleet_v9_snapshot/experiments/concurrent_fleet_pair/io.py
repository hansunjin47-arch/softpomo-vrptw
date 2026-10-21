"""Bounded retry for transient Windows reader locks; preserve resume history."""
import json
from pathlib import Path
import shutil
import time
import uuid


def replace(source, target):
    for attempt in range(40):
        try:
            Path(source).replace(target)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    replace(temp, path)


def reconcile_logs(folder, update, epoch):
    for name, key, limit in [('training.jsonl', 'update', update), ('epochs.jsonl', 'epoch', epoch)]:
        path = folder / name
        if not path.exists():
            continue
        rows = {}
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row[key] <= limit:
                rows[row[key]] = row
        shutil.copy2(path, path.with_name(path.name + '.before-resume-' + uuid.uuid4().hex))
        temp = path.with_suffix('.tmp')
        temp.write_text(''.join(json.dumps(rows[k]) + '\n' for k in sorted(rows)), encoding='utf-8')
        replace(temp, path)
    # Preserve, but remove from the active diagnostic namespace, discarded updates.
    for name in ('early_history', 'checks'):
        for path in (folder / name).glob('update_*.json'):
            if int(path.stem.split('_')[-1]) > update:
                archive = folder / 'discarded_on_resume' / name
                archive.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(archive / (path.name + '.' + uuid.uuid4().hex)))
