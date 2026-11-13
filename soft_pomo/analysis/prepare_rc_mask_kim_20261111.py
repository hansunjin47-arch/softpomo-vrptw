"""Generate isolated corrected-prompt RC Kim caches for two vehicle costs."""
import argparse
import hashlib
import shutil

import torch
from analysis import pair_kim_rl_example as shared
from experiments.concurrent_fleet_v3 import run as data_module


def main(allow_api=False):
    if not allow_api:
        raise RuntimeError('Explicit --allow-api required')
    torch.set_num_threads(1)
    for weight in (100, 200):
        pure = shared.ROOT / f'result/rc_pair_returnmask_v{weight}_l5_s42_20261031'
        art = shared.ROOT / f'result/rc101_pair{weight}_l5_mask_kim_cache_20261111'
        art.mkdir(parents=True, exist_ok=True)
        cfg = shared.read(pure / 'protocol.json')['configs']['5']
        assert cfg['environment']['protect_last_service_vehicle']
        assert cfg['environment']['normalize_advantage_std']
        assert cfg['trainer']['seed'] == 42
        split = shared.read(shared.OLD / 'split.json')
        assert len(split['train']) == 20 and len(split['test']) == 80
        for row in split['train'] + split['test']:
            shared.load(row)
        source = shared.ROOT / 'analysis/pair_kim_rl_example.py'
        provenance = dict(
            vehicle_weight=weight, late=5, mask=True, seeds=list(range(42, 47)),
            shared_cache_across_seeds=True, pure=str(pure), dataset=str(data_module.SOURCE),
            prompt_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            prompt_fix='After dispatch the environment selects the vehicle; RL selects its destination.',
            checkpoint_sha256=hashlib.sha256((pure / 'late5/checkpoint-1000.pt').read_bytes()).hexdigest(),
            test='Local causal evaluation after server training; no test rollout feedback.')
        if (art / 'generation_protocol.json').exists():
            assert shared.read(art / 'generation_protocol.json') == provenance
        else:
            shared.write(art / 'generation_protocol.json', provenance)
            shutil.copy2(source, art / 'prompt_source.py')
        shared.PURE, shared.ART = pure, art
        shared.LATE, shared.VEHICLE_COST = 5, weight
        shared.EXAMPLE = f'RC101_RAIN_REFERENCE_PAIR_V{weight}_20261111'
        shared.write(art / 'status.json', dict(stage='generating', weight=weight))
        try:
            shared.prepare()
            assert len(list((art / 'train_cache').glob('*.json'))) == 20
            shared.write(art / 'status.json', dict(stage='complete', weight=weight, train_count=20))
            print('COMPLETE', weight, art, flush=True)
        except Exception as error:
            shared.write(art / 'status.json', dict(stage='failed', weight=weight, error=repr(error)))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-api', action='store_true')
    main(parser.parse_args().allow_api)
