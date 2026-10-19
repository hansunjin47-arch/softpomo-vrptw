import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
protocol = json.loads((root / 'result/concurrent_fleet_v9_train20_late10_vehicle200_20261019/protocol.json').read_text())
for name, digest in protocol['source_hashes'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
ep = protocol['configs']['10']['environment']
assert abs(ep['vehicle_penalty'] / 100 / (ep['time_penalty'] / 240 / ep['TT_max']) - 200) < 1e-9
print('Verified active source hashes and RC101 200-minute vehicle coefficient.')
