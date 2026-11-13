"""Two preregistered remaining-seed curriculum pilots; no API or Test access."""
import copy
import json
import math
import os
from pathlib import Path
import random
from statistics import mean

import torch
from analysis import train_rc_k17_curriculum_20261112 as pilot
from analysis.validate_rc_curriculum_events_20261112 import DATA, OUT as VALIDATION, KEYS

ROOT = pilot.ROOT
OUT = ROOT / 'result/rc_curriculum_confirmatory_20261112'
SEEDS = (43, 46)


def means(rows):
    return {k: mean(r[k] for r in rows) for k in KEYS}


def validation(model, cfg):
    from vrptw_env import load_solomon
    protocol = pilot.read(VALIDATION / 'protocol.json')
    assert pilot.digest(DATA / 'manifest.json') == protocol['data_sha256']
    state = pilot.rng()
    model.target = None
    model.eval()
    rows = []
    with torch.no_grad():
        for case in pilot.read(DATA / 'manifest.json')['cases']:
            path = DATA / (case['name'] + '.txt')
            assert pilot.digest(path) == case['file_sha256']
            observation = path.with_suffix('.observations.json')
            assert pilot.digest(observation) == protocol['observation_hashes'][case['name']]
            inst = load_solomon(str(path))
            env, _, _ = pilot.trainer.collect(model, inst, [pilot.read(observation)], cfg['environment'], 1, True, greedy=True)
            rows.append(dict(name=case['name'], **{k: float(v[0]) for k, v in env.metrics().items()}))
    model.train()
    pilot.restore(state)
    return rows


def compare(arms):
    avg = {arm: means(rows) for arm, rows in arms.items()}
    control = {r['name']: r for r in arms['control']}
    paired = [dict(name=r['name'], delta={k: r[k] - control[r['name']][k] for k in KEYS})
              for r in arms['curriculum']]
    groups = {}
    for family in ('all', 'RAIN', 'ACC'):
        rows = [r for r in paired if family == 'all' or f'_{family}_' in r['name']]
        groups[family] = dict(delta={k: mean(r['delta'][k] for r in rows) for k in KEYS},
            wins=sum(r['delta']['reward'] > 1e-6 for r in rows),
            losses=sum(r['delta']['reward'] < -1e-6 for r in rows),
            late_increase_cases=sum(r['delta']['total_late'] > 1e-6 for r in rows))
    return dict(means=avg, paired=paired, groups=groups,
                passed=avg['curriculum']['reward'] > avg['control']['reward'] and avg['curriculum']['unserved'] == 0)


def main():
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    if OUT.exists():
        raise RuntimeError('Fresh output required')
    OUT.mkdir()
    pilot.write(OUT / 'preregistration.json', dict(seeds=list(SEEDS),
        rule='target=max(1,floor(initial Train20 mean vehicles)-1)',
        schedule='Each seed free40 vs target20 then free20; checkpoint1000 continuation',
        success='Each seed final free Train20 reward improves over matched control and unserved0',
        validation='After training, final40 checkpoints only once on frozen new20; all outcomes retained',
        restrictions='No target/schedule tuning, no API, no original Test or baseline modifications',
        budget='160 total epochs, 3200 updates, about100GPU minutes',
        stop='Exception, NaN/Inf, 20 consecutive AMP skips, input hash change',
        source_sha256=pilot.digest(Path(__file__)), validation_protocol_sha256=pilot.digest(VALIDATION / 'protocol.json')))
    for seed in SEEDS:
        source = ROOT / f'result/rc_pair_returnmask_v100_l5_s{seed}_20261102'
        folder = OUT / f'seed{seed}'
        folder.mkdir()
        cfg = pilot.read(source / 'protocol.json')['configs']['5']
        assert cfg['trainer']['seed'] == seed
        assert cfg['environment']['late_penalty'] == 5
        assert cfg['environment']['protect_last_service_vehicle']
        assert cfg['environment']['normalize_advantage_std']
        rows = pilot.read(source / 'split.json')['train']
        assert len(rows) == 20
        assert set(r['name'] for r in rows) == set(cfg['trainer']['train_instances'])
        lookup = {r['name']: r for r in rows}
        checkpoint = source / 'late5/checkpoint-1000.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved['epoch'] == 1000
        state = saved['python_rng'], saved['cpu_rng'], saved['cuda_rng']
        order_rng = random.Random()
        order_rng.setstate(saved['python_rng'])
        order = [order_rng.choice(cfg['trainer']['train_instances']) for _ in range(800)]
        hashes = dict(pilot.read(source / 'protocol.json')['source_hashes'])
        hashes[str(checkpoint.relative_to(ROOT))] = pilot.digest(checkpoint)
        model = pilot.Policy(**cfg['model']).cuda().eval()
        model.load_state_dict(saved['model_state_dict'], strict=True)
        pilot.restore(state)
        initial = pilot.evaluate(model, rows, cfg, None)
        target = max(1, math.floor(means(initial)['vehicles']) - 1)
        expected = pilot.read(source / 'late5/checks/update_020000.json')['results']
        initial_by_name = {r['name']: r for r in initial}
        for row in expected:
            assert all(abs(initial_by_name[row['name']][k] - row[k]) < 1e-5 for k in KEYS)
        pilot.write(folder / 'initial_train20.json', initial)
        pilot.write(folder / 'protocol.json', dict(seed=seed, config=cfg, target=target,
            initial_mean=means(initial), data_order=order, input_hashes=hashes,
            model_optimizer_scaler_rng_restore=True, original_probes_reproduced=True,
            schedule='fixed20 constrained +20 free; control40free', epochs=40, updates_per_arm=800))
        print('TARGET_FIXED', seed, target, means(initial), flush=True)
        del model
        for arm in ('control', 'curriculum'):
            arm_folder = folder / arm
            arm_folder.mkdir()
            model = pilot.Policy(**cfg['model']).cuda().train()
            model.load_state_dict(saved['model_state_dict'], strict=True)
            opt = torch.optim.Adam(model.parameters(), **cfg['optimizer']['optimizer'])
            opt.load_state_dict(copy.deepcopy(saved['optimizer_state_dict']))
            scaler = torch.amp.GradScaler('cuda')
            scaler.load_state_dict(copy.deepcopy(saved['scaler']))
            pilot.restore(state)
            assert pilot.evaluate(model, rows, cfg, None) == initial
            skips = 0
            for epoch in range(1, 41):
                assert all(pilot.digest(ROOT / p) == digest for p, digest in hashes.items())
                model.target = target if arm == 'curriculum' and epoch <= 20 else None
                model.new_customer_epsilon = .01
                for batch_index in range(20):
                    update = (epoch - 1) * 20 + batch_index
                    name = order[update]
                    inst, public = pilot.trainer.load(lookup[name])
                    model.fallback_count = 0
                    model.constrained_states = 0
                    pilot.write(OUT / 'status.json', dict(stage='training', pid=os.getpid(), seed=seed,
                        arm=arm, epoch=epoch, update=update, target=model.target))
                    summary, _ = pilot.trainer.train_batch(model, opt, scaler, inst, public,
                        cfg['environment'], min(128, 2500 - batch_index * 128), 100, True)
                    assert math.isfinite(summary['loss'])
                    assert all(torch.isfinite(p).all() for p in model.parameters())
                    skips = skips + 1 if summary['optimizer_step_skipped'] else 0
                    assert skips < 20
                    pilot.trainer.append(arm_folder / 'training.jsonl', dict(epoch=epoch, update=update + 1,
                        name=name, target=model.target, fallback_count=model.fallback_count, **summary))
                if epoch % 10 == 0:
                    evaluated = pilot.evaluate(model, rows, cfg, None)
                    pilot.write(arm_folder / f'free_epoch{epoch:03d}.json', evaluated)
                    torch.save(dict(model_state_dict=model.state_dict(), optimizer_state_dict=opt.state_dict(),
                        scaler=scaler.state_dict(), rng=pilot.rng(), epoch=epoch), arm_folder / f'checkpoint-{epoch}.pt')
                    print('MILESTONE', seed, arm, epoch, means(evaluated), flush=True)
            del model, opt, scaler
            torch.cuda.empty_cache()
        train_arms = {arm: pilot.read(folder / arm / 'free_epoch040.json') for arm in ('control', 'curriculum')}
        pilot.write(folder / 'train_comparison.json', compare(train_arms))
        validation_arms = {}
        for arm in ('control', 'curriculum'):
            model = pilot.Policy(**cfg['model']).cuda().eval()
            model.load_state_dict(torch.load(folder / arm / 'checkpoint-40.pt', map_location='cpu', weights_only=False)['model_state_dict'])
            validation_arms[arm] = validation(model, cfg)
            pilot.write(folder / arm / 'validation20.json', validation_arms[arm])
            del model
        pilot.write(folder / 'validation_comparison.json', compare(validation_arms))
        print('SEED_COMPLETE', seed, compare(validation_arms)['groups']['all'], flush=True)
    pilot.write(OUT / 'status.json', dict(stage='complete', seeds=list(SEEDS), api_calls=0,
        test_performance_accessed=False, total_epochs=160))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        pilot.write(OUT / 'failure.json', dict(error=repr(error)))
        raise
