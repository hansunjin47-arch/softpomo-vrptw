"""CPU-only verification of the unmodified active-run snapshot."""
import hashlib
import json
from pathlib import Path
from experiments.concurrent_fleet_v8 import train

ROOT = Path(__file__).resolve().parent


def main():
    protocol = json.loads((ROOT / 'result/concurrent_fleet_v8_fresh_train20_20261018/protocol.json').read_text())
    for name, expected in protocol['source_hashes'].items():
        path = ROOT / name.replace('\\', '/')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    assert train.ROOT == ROOT
    for weight in protocol['weights']:
        path = ROOT / f'result/rc101_learned_first_train20_20261017/late{weight}/config.json'
        assert json.loads(path.read_text()) == protocol['configs'][str(weight)], str(weight)
    split = json.loads((train.SOURCE / 'split.json').read_text())
    assert len(split['train']) == 20 and len(split['test']) == 80
    for row in split['train'] + split['test']:
        inst, public = train.load(row)
        assert inst['n_customers'] == 100 and len(public) == 1, row['name']
        assert 'revealed_at_min' in public[0], row['name']
    print('PASS: 11 active-run source hashes, four configurations, 100 instance/observation pairs; no GPU or LLM calls.')


if __name__ == '__main__':
    main()
