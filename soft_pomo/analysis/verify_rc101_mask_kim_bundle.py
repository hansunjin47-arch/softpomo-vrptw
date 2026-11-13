"""Offline bundle checks: ten mocked launches and four CPU rollout smokes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
NAME = 'rc101_pair100_200_l5_mask_kim_s42_46_20261111'
BUNDLE = ROOT/'server_export'/NAME


def main():
    manifest = json.loads((BUNDLE/'manifest.json').read_text())
    assert all(hashlib.sha256((BUNDLE/p).read_bytes()).hexdigest() == h for p,h in manifest['sha256'].items())
    tasks = [tuple(map(int, row.split())) for row in (BUNDLE/'tasks.txt').read_text().splitlines()]
    assert tasks == [(weight,seed) for weight in (100,200) for seed in range(42,47)]
    for filename in ('tasks.txt','submit.sh','task.slurm'):
        assert b'\r' not in (BUNDLE/filename).read_bytes()
    submit = (BUNDLE/'submit.sh').read_text()
    task = (BUNDLE/'task.slurm').read_text()
    assert 'jobs[index-2]' in submit and '--hold' in submit and '^sj_' in submit
    assert '--gres=gpu:1' in task and 'scontrol requeue' in task
    assert 'unset OPENROUTER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY' in task
    sys.path.insert(0, str(BUNDLE))
    import torch
    torch.set_num_threads(1)
    from experiments.concurrent_fleet_v3 import run as data_module
    from experiments.concurrent_fleet_bias_guided.support import read, load
    assert Path(data_module.__file__).resolve().is_relative_to(BUNDLE)
    data_module.SOURCE = BUNDLE/'dataset'
    spec = importlib.util.spec_from_file_location('frozen_rc_runner', BUNDLE/'run_training.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    captured = []
    smoke = []
    split = read(BUNDLE/'dataset/split.json')

    def mock_run(args):
        weight, seed = args.vehicle_cost, args.seed
        assert args.config == BUNDLE/'artifacts'/f'vehicle{weight}'/'config.json'
        assert args.source == BUNDLE/'dataset'
        assert args.reward_denominator == 1696.95 and args.weights == [5]
        assert args.return_mask and not args.no_advantage_std and args.resume_from is None
        assert (args.new_customer_epsilon,args.epsilon_final,args.epsilon_zero_epoch) == (.05,.01,1000)
        captured.append(dict(weight=weight,seed=seed,config=str(args.config.relative_to(BUNDLE))))
        if seed != 42:
            return
        cfg = read(args.config)
        model = runner.GuidedPolicy(**cfg['model']).cpu().eval()
        for kind in ('RAIN','ACC'):
            row = next(r for r in split['train'] if f'_{kind}_' in r['name'])
            inst, public = load(row)
            env = runner.train.TensorFleetEnv(inst,1,cfg['environment'],device='cpu')
            assert env.guidance.beta == 5.
            assert len(env.guidance.updates) == int(kind == 'ACC')
            with torch.no_grad():
                starts, enc = model.encode_events(inst, public)
                for step in range(600):
                    if bool(env.done.all()):
                        break
                    frame = env.frame()
                    p = model.probabilities(frame,starts,enc,env)
                    assert torch.isfinite(p).all()
                    action = p.argmax(1)
                    assert bool(frame['mask'].gather(1,action[:,None]).all())
                    env.step(action,validate=True)
                else:
                    raise RuntimeError('Smoke rollout exceeded 600 steps')
            smoke.append(dict(weight=weight,kind=kind,steps=step,
                metrics={k:float(v[0]) for k,v in env.metrics().items()}))
    runner.train.run = mock_run
    for weight,seed in tasks:
        sys.argv = ['run_training.py','--out','unused_smoke_output','--seed',str(seed),'--vehicle-cost',str(weight)]
        runner.main()
    assert len(captured) == 10 and len(smoke) == 4
    report = dict(manifest_verified=True,lf_verified=True,tasks=captured,cpu_random_weight_smoke=smoke,
        caveat='CPU random initialized model plumbing smoke only; no learning/performance claim, no GPU, API, or submission')
    dest = ROOT/'server_export'/(NAME+'_validation.json')
    dest.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
