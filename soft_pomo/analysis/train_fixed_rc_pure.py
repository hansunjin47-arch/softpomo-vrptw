"""Use the existing Pure RL entry point with an isolated fixed-pool data root."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_vrptw as training

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--fixed-data-dir', default=str(ROOT / 'data/fixed_rc_v2_20260920_223253'))
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    data = Path(args.fixed_data_dir).resolve()
    import json
    assert json.loads((data / 'validation.json').read_text())['validated']
    training.trainer_params['data_dir'] = str(data)
    training.main()
