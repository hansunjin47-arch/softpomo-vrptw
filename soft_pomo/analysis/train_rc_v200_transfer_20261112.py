"""Preregistered unchanged-method transfer to vehicle weight200, seed42."""
import copy
import math
import os
import random
from pathlib import Path
import torch
from analysis import train_rc_k17_curriculum_20261112 as pilot
from analysis.train_rc_curriculum_confirmatory_20261112 import means, compare, validation, KEYS

ROOT = pilot.ROOT
OUT = ROOT / 'result/rc_v200_curriculum_transfer_20261112'
SOURCE = ROOT / 'result/rc_pair_returnmask_v200_l5_s42_20261031'


def main():
    torch.set_num_threads(1)
    assert not OUT.exists()
    cfg = pilot.read(SOURCE / 'protocol.json')['configs']['5']
    ep = cfg['environment']
    assert cfg['trainer']['seed'] == 42 and ep['late_penalty'] == 5
    assert ep['protect_last_service_vehicle'] and ep['normalize_advantage_std']
    assert abs(ep['vehicle_penalty'] / 100 - 200 / 1696.95) < 1e-12
    rows = pilot.read(SOURCE / 'split.json')['train']
    assert len(rows) == 20 and set(r['name'] for r in rows) == set(cfg['trainer']['train_instances'])
    checks = pilot.tests(rows[0], ep)
    checkpoint = SOURCE / 'late5/checkpoint-1000.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['epoch'] == 1000
    state = saved['python_rng'], saved['cpu_rng'], saved['cuda_rng']
    order_rng = random.Random()
    order_rng.setstate(saved['python_rng'])
    order = [order_rng.choice(cfg['trainer']['train_instances']) for _ in range(800)]
    hashes = dict(pilot.read(SOURCE / 'protocol.json')['source_hashes'])
    for path in (checkpoint, Path(__file__), ROOT / 'analysis/train_rc_k17_curriculum_20261112.py',
                 ROOT / 'analysis/train_rc_curriculum_confirmatory_20261112.py',
                 ROOT / 'result/rc_curriculum_validation_20261112/protocol.json', SOURCE / 'split.json'):
        hashes[str(path.relative_to(ROOT))] = pilot.digest(path)
    OUT.mkdir()
    pilot.write(OUT / 'preregistration.json', dict(seed=42, vehicle_weight=200, late_weight=5,
        rule='target=max(1,floor(initial Train20 mean vehicles)-1)',
        schedule='fixed constrained20 then free20 vs matched free40; final40 only',
        scope='Single prespecified objective transfer, not favorable-weight search. No Test/API.',
        success='Final Train reward improves vs matched control with U0; report validation separately.',
        budget='80 epochs,1600updates,approximately50GPUminutes',
        stop='NaN/Inf,exception,20 consecutive AMP skips,source hash change',
        config=cfg, input_hashes=hashes, data_order=order, cpu_tests=checks))
    assert all(pilot.digest(ROOT / p) == h for p, h in hashes.items())
    model = pilot.Policy(**cfg['model']).cuda().eval()
    model.load_state_dict(saved['model_state_dict'], strict=True)
    pilot.restore(state)
    initial = pilot.evaluate(model, rows, cfg, None)
    target = max(1, math.floor(means(initial)['vehicles']) - 1)
    lookup = {r['name']: r for r in rows}
    initial_lookup = {r['name']: r for r in initial}
    for row in pilot.read(SOURCE / 'late5/checks/update_020000.json')['results']:
        assert all(abs(initial_lookup[row['name']][k]-row[k]) < 1e-5 for k in KEYS)
    pilot.write(OUT / 'initial_train20.json', initial)
    pilot.write(OUT / 'protocol.json', dict(config=cfg, target=target, initial=means(initial),
        original_probes_reproduced=True, source_hashes=hashes))
    print('TARGET_FIXED', target, means(initial), flush=True)
    del model
    for arm in ('control', 'curriculum'):
        destination = OUT / arm
        destination.mkdir()
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
            assert all(pilot.digest(ROOT / p) == h for p, h in hashes.items())
            model.target = target if arm == 'curriculum' and epoch <= 20 else None
            model.new_customer_epsilon = .01
            for batch_index in range(20):
                update = (epoch-1)*20 + batch_index
                name = order[update]
                inst, public = pilot.trainer.load(lookup[name])
                model.fallback_count = 0
                pilot.write(OUT / 'status.json', dict(stage='training', pid=os.getpid(), arm=arm, epoch=epoch, update=update))
                stats, _ = pilot.trainer.train_batch(model, opt, scaler, inst, public, ep,
                    min(128, 2500-batch_index*128), 100, True)
                assert math.isfinite(stats['loss']) and all(torch.isfinite(p).all() for p in model.parameters())
                skips = skips+1 if stats['optimizer_step_skipped'] else 0
                assert skips < 20
                pilot.trainer.append(destination / 'training.jsonl', dict(epoch=epoch, update=update+1,
                    name=name, target=model.target, fallback_count=model.fallback_count, **stats))
            if epoch % 10 == 0:
                result = pilot.evaluate(model, rows, cfg, None)
                pilot.write(destination / f'free_epoch{epoch:03d}.json', result)
                torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(), optimizer_state_dict=opt.state_dict(),
                    scaler=scaler.state_dict(), rng=pilot.rng()), destination / f'checkpoint-{epoch}.pt')
                print('MILESTONE', arm, epoch, means(result), flush=True)
        del model, opt, scaler
        torch.cuda.empty_cache()
    pilot.write(OUT / 'train_comparison.json', compare({a: pilot.read(OUT/a/'free_epoch040.json') for a in ('control','curriculum')}))
    val = {}
    for arm in ('control', 'curriculum'):
        model = pilot.Policy(**cfg['model']).cuda().eval()
        model.load_state_dict(torch.load(OUT/arm/'checkpoint-40.pt', map_location='cpu', weights_only=False)['model_state_dict'])
        val[arm] = validation(model, cfg)
        pilot.write(OUT/arm/'validation20.json', val[arm])
        del model
    pilot.write(OUT/'validation_comparison.json', compare(val))
    pilot.write(OUT/'status.json', dict(stage='complete', total_epochs=80, api_calls=0, test_accessed=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        pilot.write(OUT/'failure.json', dict(error=repr(error)))
        raise
