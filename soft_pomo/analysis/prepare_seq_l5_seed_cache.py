"""Seed-specific RC101 lambda5 GPT caches with measured training feedback."""
import argparse
import json
import run_gpt_bks_late5 as late5

shared = late5.shared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'call'])
    parser.add_argument('--seed', type=int, choices=range(42,47), required=True)
    parser.add_argument('--allow-api', action='store_true')
    args = parser.parse_args()
    shared.BASE = shared.ROOT / 'server_export/rc101_seq_train20_late5_seeds42_46_20261023' / f'seed{args.seed}'
    shared.OUT = shared.ROOT / 'result/rc101_seq_phase2_l5_feedback_seeds_20261026' / f'seed{args.seed}'
    config = json.loads((shared.BASE/'configs/pure.json').read_text())
    assert config['trainer']['seed'] == args.seed
    if args.stage == 'prepare':
        shared.prepare(execution_feedback_enabled=True)
    else:
        if not args.allow_api:
            parser.error('Paid calls require --allow-api')
        protocol = shared.io.read(shared.OUT/'protocol.json')
        assert protocol['feedback'] is True
        assert protocol['checkpoint_sha256'] == shared.io.sha(shared.BASE/'runs/pure/checkpoint-500.pt')
        shared.call(shared.OUT)


if __name__ == '__main__':
    main()
