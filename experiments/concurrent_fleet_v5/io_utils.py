"""Portable inputs and atomic JSON output for the frozen train20 experiment."""
import hashlib
import json
from pathlib import Path
from vrptw_env import load_solomon

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent / 'inputs'


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    temp.replace(path)


def load(row):
    path = ROOT / 'data' / (row['name'] + '.txt')
    if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
        raise ValueError('Scenario hash mismatch: ' + row['name'])
    return load_solomon(str(path)), [json.loads(path.with_suffix('.observations.json').read_text())]
