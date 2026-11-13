"""Attach saved launch settings without resuming or ending active runs."""
import sys
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_tracking import sanitize, dataset_manifest
from mlflow.tracking import MlflowClient


def main():
    client = MlflowClient(tracking_uri='http://127.0.0.1:5000')
    runs = {
        '5e683948bca14d4696c3fc30123e368f': 'result_soft_fixed_kim_sonnet_20260921/rc1/training_settings.json',
        'fd2ec2baeeef4b13b5fe3d9fdfcca54c': 'result_soft_fixed_kim_sonnet_20260921/r1/training_settings.json',
        '0d50e3f5731c42b3939fd0fc8af14bb7': 'result_soft_fixed_kim_startgate_rc_20260921/settings.json',
    }
    for run, source in runs.items():
        settings = json.loads((ROOT / source).read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as tmp:
            for name, data in {
                'saved_launch_settings.json': sanitize(settings),
                'dataset_at_backfill.json': {'files': dataset_manifest(settings['trainer']),
                    'note': 'Hashes measured at backfill, not at original launch.'},
            }.items():
                path = Path(tmp) / name
                path.write_text(json.dumps(data, indent=2), encoding='utf-8')
                client.log_artifact(run, str(path), 'provenance')
        client.set_tag(run, 'provenance_source', source)
        client.set_tag(run, 'detailed_test_logging', 'Not installed in already-running process')
        print('UPDATED', run)


if __name__ == '__main__':
    main()
