"""Kim guidance adapter; reuse the pair trainer without changing its source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
from experiments.concurrent_fleet_pair import train
from experiments.concurrent_fleet_pair.env import TensorFleetEnv
from experiments.concurrent_fleet_pair.policy import FleetPolicy, mix_new_customers
from experiments.concurrent_fleet_guided.guidance import ConfidenceGuidance
from experiments.concurrent_fleet_v9_guided import online
from experiments.concurrent_fleet_v9_guided.prompts import parse_response, parse_test_response
from experiments.concurrent_fleet_bias_guided.llm_retry import RetryingClient
from experiments.concurrent_fleet_bias_guided.support import load, read

ARTIFACT = train.ROOT/'result/v9_bias_guided300_late10_20261022/kim/artifacts'


class GuidedEnv(TensorFleetEnv):
    def __init__(self, *args, groups, scores, beta=5., **kwargs):
        super().__init__(*args, **kwargs)
        self.guidance = ConfidenceGuidance(self, groups, scores, beta)
        self.recorded_history = []

    def step(self, action, validate=False):
        new, vehicle, dest = self.decode_action(action)
        live = ~self.done
        if self.count == 1:
            self.recorded_history.append(torch.stack([self.time, vehicle, dest, new, live], 1).clone())
        ids = self.ids[new & live]
        vehicles = vehicle[new & live].clone()
        groups = self.guidance.membership[dest[new & live]]
        super().step(action, validate)
        self.guidance.vehicle_group[ids, vehicles] = groups

    def update_event_confidence(self, scores, *, revealed_at_min, effective_time_min):
        if effective_time_min < revealed_at_min:
            raise ValueError('Cannot use guidance before disclosure')
        self.guidance.update(scores, effective_time_min=effective_time_min)


class GuidedPolicy(FleetPolicy):
    def probabilities(self, frame, starts, encodings, env):
        epsilon = self.new_customer_epsilon
        self.new_customer_epsilon = 0.
        try:
            probabilities = super().probabilities(frame, starts, encodings, env)
        finally:
            self.new_customer_epsilon = epsilon
        guidance = env.guidance
        group = guidance.vehicle_group.gather(1, frame['vehicles'])
        confidence = guidance.initial[group.clamp(min=0)]
        for effective, target, scores in guidance.updates:
            eligible = target & (env.time >= effective)
            confidence = torch.where(eligible[:, None, None], scores[group.clamp(min=0)], confidence)
        active = (group >= 0) & ~env.dispatch_open[:, None] & ~env.done[:, None]
        bias = guidance.beta * confidence * active[:, :, None]
        bias = torch.cat([bias.new_zeros(env.count, env.n), bias.flatten(1)], 1)
        # Equivalent to adding beta*confidence to logits; keep masked zeros exact.
        weighted = probabilities * torch.exp(bias)
        probabilities = weighted / weighted.sum(1, keepdim=True)
        return mix_new_customers(probabilities, frame['mask'], env.n, epsilon if self.training else 0.)


class OnlineEnv(GuidedEnv):
    def __init__(self, *args, reports, on_disclosure, **kwargs):
        self.disclosure_times = sorted({float(e['revealed_at_min']) for e in reports
                                       if e['event_type'] == 'ACCIDENT' and e['revealed_at_min'] > 0})
        self.disclosure_index = 0
        self.on_disclosure = on_disclosure
        super().__init__(*args, **kwargs)
        if self.count != 1:
            raise ValueError('One-shot evaluation requires one rollout')

    def _process(self):
        target = self.time.clone()
        while self.disclosure_index < len(self.disclosure_times):
            disclosure = self.disclosure_times[self.disclosure_index]
            tick = self.time.new_tensor(disclosure)
            if float(tick) < disclosure:
                tick = torch.nextafter(tick, tick.new_tensor(float('inf')))
            if float(tick) > float(target[0]):
                break
            self.time[:] = tick
            super()._process()
            self.disclosure_index += 1
            if not bool(self.done.all()):
                self.on_disclosure(self, disclosure)
        self.time.copy_(target)
        super()._process()


def audit():
    split = read(train.SOURCE/'split.json')
    assert split == read(ARTIFACT/'split.json')
    assert read(ARTIFACT/'reuse_audit.json')['rl_reference'] is False
    groups = read(ARTIFACT/'groups.json')
    for row in split['train']:
        cached = read(ARTIFACT/'train_cache'/(row['name']+'.json'))
        parse_response(cached['initial'], groups)
        if cached.get('post_disclosure') is not None:
            parse_response(cached['post_disclosure'], groups)
    return split, groups


def run(out, *, late=10, vehicle_cost=300.):
    split, groups = audit()
    if out.exists():
        raise ValueError('Refusing duplicate output')
    out.mkdir(parents=True)
    shutil.copytree(ARTIFACT, out/'artifacts')
    artifact = out/'artifacts'
    files = list(Path(__file__).parent.glob('*.py'))
    files += list((train.ROOT/'experiments/concurrent_fleet_v9_guided').glob('*.py'))
    files += [train.ROOT/'experiments/concurrent_fleet_guided/guidance.py']
    train.write(out/'kim_protocol.json', dict(beta=5., reference_start_gate=False,
        guidance='Existing-vehicle routing only after dispatch; NEW and depot unbiased',
        initialization='fresh seed42; no RL checkpoint', train_cache=str(ARTIFACT),
        test='One greedy trajectory; instance-only initial prompt and causal accident disclosure; no test RL reference',
        source_hashes={str(p.relative_to(train.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        cache_hashes={str(p.relative_to(artifact)):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifact.rglob('*.json')}))
    for p in files:
        dest=out/'source'/p.relative_to(train.ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p,dest)
    class CachedEnv(GuidedEnv):
        def __init__(self, inst, count, params, device='cuda'):
            cache = read(artifact/'train_cache'/(inst['name']+'.json'))
            super().__init__(inst, count, params, device, groups=groups,
                             scores=parse_response(cache['initial'], groups))
            if cache.get('post_disclosure') is not None:
                t=cache['revealed_at_min']
                self.update_event_confidence(parse_response(cache['post_disclosure'], groups),
                                             revealed_at_min=t, effective_time_min=t)
    original_collect = train.collect
    test_names = {r['name'] for r in split['test']}
    template = read(artifact/'request_template.json')
    client = RetryingClient(out/'calls/test', lambda req, payload: parse_test_response(
        payload, groups, json.loads(req['messages'][1]['content'])['selectable_customers']), allow_paid=True)
    def collect(model, inst, public, params, count, amp=True, greedy=False, progress=None):
        if inst['name'] not in test_names:
            return original_collect(model, inst, public, params, count, amp, greedy, progress)
        assert count == 1 and greedy
        captured=[]
        class CapturedEnv(OnlineEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured.append(self)
        online.OnlineTestEnv = CapturedEnv
        online.evaluate_instance(model, inst, params, groups, public, template=template,
            client=client, beta=5., out=out/f'late{late}/online_test'/inst['name'], amp=amp)
        env=captured[0]
        return env, [], env.recorded_history
    train.TensorFleetEnv=CachedEnv
    train.FleetPolicy=GuidedPolicy
    train.load=load
    train.collect=collect
    args=argparse.Namespace(out=out, weights=[late], vehicle_cost=vehicle_cost, no_advantage_std=False,
        new_customer_epsilon=.05, epsilon_final=.01, epsilon_zero_epoch=1000,
        resume=False, resume_from=None, smoke_updates=0)
    # Run in a child directory so the unchanged trainer can enforce fresh initialization.
    args.out=out/'training'
    train.run(args)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    parser.add_argument('--audit', action='store_true')
    args=parser.parse_args()
    if args.audit:
        audit()
        print('20 train caches and split validated; no paid calls')
    else:
        if args.out is None:
            parser.error('--out required')
        run(args.out)
