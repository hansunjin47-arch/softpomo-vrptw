"""Train-only continuation with full Train20 greedy checks and isolated checkpoints."""
import copy
import hashlib
import random
import statistics
import torch
from analysis.evaluate_r101_mask_kim_20261109 import BUNDLE, RECEIVED
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv, GuidedPolicy
from experiments.concurrent_fleet_pair import train
from experiments.concurrent_fleet_v3 import run as data_module
from experiments.concurrent_fleet_bias_guided.support import ROOT, read, load
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_pair.train import append

OUT = ROOT/'result/r101_kim44_continuation10_20261109'


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    data_module.SOURCE = BUNDLE/'dataset'
    run = RECEIVED/'runs/seed44'
    source = run/'late5/checkpoint-1000.pt'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    saved = torch.load(source, map_location='cpu', weights_only=False)
    cfg = read(run/'protocol.json')['configs']['5']
    rows = read(run/'split.json')['train']
    groups = read(BUNDLE/'artifacts/groups.json')

    class CachedEnv(GuidedEnv):
        def __init__(self, inst, count, params, device='cuda'):
            cache = read(BUNDLE/'artifacts/train_cache'/(inst['name']+'.json'))
            super().__init__(inst, count, params, device, groups=groups,
                             scores=parse_response(cache['initial'], groups))
            if cache['post_disclosure'] is not None:
                t = cache['revealed_at_min']
                self.update_event_confidence(parse_response(cache['post_disclosure'], groups),
                                             revealed_at_min=t, effective_time_min=t)

    train.TensorFleetEnv = CachedEnv
    data = {r['name']: load(r) for r in rows}
    tp = cfg['trainer']
    names = tp['train_instances']
    assert len(rows) == 20 and set(names) == set(data)
    model = GuidedPolicy(**cfg['model']).cuda()
    model.load_state_dict(saved['model_state_dict'])
    opt = torch.optim.Adam(model.parameters(), **cfg['optimizer']['optimizer'])
    opt.load_state_dict(saved['optimizer_state_dict'])
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    scaler.load_state_dict(saved['scaler'])
    state = copy.deepcopy(saved['state'])
    reward_log, loss_log = list(saved['reward_log']), list(saved['loss_log'])
    assert saved['epoch'] == 1000 and state['epoch'] == 1001 and state['episodes'] == 0
    random.setstate(saved['python_rng'])
    torch.set_rng_state(saved['cpu_rng'])
    torch.cuda.set_rng_state(saved['cuda_rng'])
    write(OUT/'protocol.json', dict(source=str(source), checkpoint_sha256=digest, config=cfg,
        epochs=10, epsilon=.01, amp=True, optimizer='original Adam, no reset',
        rng='restored checkpoint RNG; evaluation preserves RNG; hardware differs from server, not exact replay',
        evaluation='Train20 greedy each epoch; no Test and no API', gpu=torch.cuda.get_device_name()))

    def status(stage, **extra):
        write(OUT/'status.json', dict(stage=stage, **state, target_epoch=1010, **extra))

    def checkpoint(epoch):
        path = OUT/f'checkpoint-{epoch}.pt'
        temp = path.with_suffix('.tmp')
        torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
            optimizer_state_dict=opt.state_dict(), scaler=scaler.state_dict(), state=state,
            reward_log=reward_log, loss_log=loss_log, python_rng=random.getstate(),
            cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), temp)
        temp.replace(path)

    def evaluate(epoch):
        status('evaluating_train20', evaluating_epoch=epoch)
        rng = random.getstate(), torch.get_rng_state(), torch.cuda.get_rng_state()
        model.eval()
        results = []
        try:
            for name in names:
                inst, public = data[name]
                env, _, _ = train.collect(model, inst, public, cfg['environment'], 1, amp=True, greedy=True)
                results.append(dict(name=name, **{k: float(v[0]) for k, v in env.metrics().items()}))
        finally:
            random.setstate(rng[0]); torch.set_rng_state(rng[1]); torch.cuda.set_rng_state(rng[2])
            model.train()
        means = {k: statistics.mean(r[k] for r in results)
                 for k in ('reward', 'vehicles', 'travel_time', 'total_late', 'unserved')}
        write(OUT/f'train20_epoch{epoch}.json', dict(epoch=epoch, mean=means, results=results))
        append(OUT/'greedy_epochs.jsonl', dict(epoch=epoch, **means))
        print('EVAL', epoch, means, flush=True)

    evaluate(1000)
    while state['epoch'] <= 1010:
        if (OUT/'STOP').exists():
            checkpoint('interrupted'); status('stopped'); return
        name = random.choice(names)
        inst, public = data[name]
        batch = min(tp['train_batch_size'], tp['train_episodes']-state['episodes'])
        model.train(); model.new_customer_epsilon = .01
        status('training', name=name, batch_size=batch)
        summary, _ = train.train_batch(model, opt, scaler, inst, public, cfg['environment'],
            batch, cfg['environment']['pomo_size'], amp=True, max_steps=tp.get('max_steps', 600))
        state['update'] += 1; state['episodes'] += batch
        state['trajectories'] += batch*cfg['environment']['pomo_size']
        state['epoch_reward_sum'] += summary['best_rollout_reward']*batch
        state['epoch_loss_sum'] += summary['loss']*batch
        append(OUT/'training.jsonl', dict(**state, name=name, batch_size=batch, **summary))
        print('UPDATE', state['epoch'], state['update'], summary['reward'],
              summary['optimizer_step_skipped'], flush=True)
        if state['episodes'] == tp['train_episodes']:
            epoch = state['epoch']
            reward_log.append(state['epoch_reward_sum']/state['episodes'])
            loss_log.append(state['epoch_loss_sum']/state['episodes'])
            append(OUT/'epochs.jsonl', dict(epoch=epoch, best_rollout_reward=reward_log[-1], loss=loss_log[-1]))
            state['epoch'] += 1; state['episodes'] = 0
            state['epoch_reward_sum'] = 0.; state['epoch_loss_sum'] = 0.
            checkpoint(epoch)
            evaluate(epoch)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    status('complete', original_checkpoint_unchanged=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            write(OUT/'failure.json', dict(error=repr(error)))
        raise
