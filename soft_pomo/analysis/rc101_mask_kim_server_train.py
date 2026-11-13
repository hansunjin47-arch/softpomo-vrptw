"""Train frozen RC101 Kim caches without API calls or server-side evaluation."""
import argparse
from pathlib import Path

from experiments.concurrent_fleet_pair import train
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv, GuidedPolicy
from experiments.concurrent_fleet_bias_guided.support import load, read
from experiments.concurrent_fleet_v9_guided.prompts import parse_response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=range(42, 47), required=True)
    parser.add_argument('--vehicle-cost', type=int, choices=(100, 200), required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke-updates', type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    art = root/'artifacts'/f'vehicle{args.vehicle_cost}'
    groups = read(art/'groups.json')
    split = read(root/'dataset/split.json')
    assert split == read(art/'split.json')
    assert len(split['train']) == 20 and len(split['test']) == 80
    assert read(art/'complete.json')['train_count'] == 20
    for row in split['train']:
        cache = read(art/'train_cache'/(row['name']+'.json'))
        parse_response(cache['initial'], groups)
        if cache['post_disclosure'] is not None:
            parse_response(cache['post_disclosure'], groups)

    class CachedEnv(GuidedEnv):
        def __init__(self, inst, count, params, device='cuda'):
            cache = read(art/'train_cache'/(inst['name']+'.json'))
            super().__init__(inst, count, params, device, groups=groups,
                scores=parse_response(cache['initial'], groups), beta=5.)
            if cache['post_disclosure'] is not None:
                t = cache['revealed_at_min']
                self.update_event_confidence(parse_response(cache['post_disclosure'], groups),
                    revealed_at_min=t, effective_time_min=t)

    train.TensorFleetEnv = CachedEnv
    train.FleetPolicy = GuidedPolicy
    train.load = load
    args.config = art/'config.json'
    args.source = root/'dataset'
    args.weights = [5]
    args.reward_denominator = 1696.95
    args.return_mask = True
    args.no_advantage_std = False
    args.new_customer_epsilon = .05
    args.epsilon_final = .01
    args.epsilon_zero_epoch = 1000
    args.resume_from = None
    train.run(args)


if __name__ == '__main__':
    main()
