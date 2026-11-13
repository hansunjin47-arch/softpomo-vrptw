"""Fixed-checkpoint matched fine-tuning RNG replication; no repeated Test."""
import argparse
import copy
import math
import os
from pathlib import Path
import random

import torch
from analysis import train_rc_k17_curriculum_20261112 as pilot
from analysis.train_rc_curriculum_confirmatory_20261112 import means, validation, compare

ROOT = pilot.ROOT
OUT = ROOT / 'result/rc_curriculum_rng_repeats_20261112'
SOURCE = ROOT / 'result/rc_pair_returnmask_v100_l5_s44_20261102'
CHECKPOINT = SOURCE / 'late5/checkpoint-1000.pt'
RNG_SEEDS = (144, 244)
TARGET = 17


def prepare():
    if OUT.exists():
        raise RuntimeError('Fresh output required')
    cfg = pilot.read(SOURCE / 'protocol.json')['configs']['5']
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    assert saved['epoch'] == 1000
    model = pilot.Policy(**cfg['model']).cpu().eval()
    model.load_state_dict(saved['model_state_dict'], strict=True)
    hashes = dict(pilot.read(SOURCE / 'protocol.json')['source_hashes'])
    for path in (CHECKPOINT, Path(__file__), ROOT / 'analysis/train_rc_k17_curriculum_20261112.py',
                 ROOT / 'analysis/train_rc_curriculum_confirmatory_20261112.py',
                 ROOT / 'result/rc_curriculum_validation_20261112/protocol.json'):
        hashes[str(path.relative_to(ROOT))] = pilot.digest(path)
    orders = {}
    for seed in RNG_SEEDS:
        rng = random.Random(seed)
        orders[str(seed)] = [rng.choice(cfg['trainer']['train_instances']) for _ in range(800)]
    OUT.mkdir()
    pilot.write(OUT / 'protocol.json', dict(checkpoint=str(CHECKPOINT), config=cfg,
        model_training_seed=44, fine_tune_rng_seeds=list(RNG_SEEDS), target=TARGET,
        selection='Original seed44 checkpoint1000, the first original-Pure curriculum discovery checkpoint; not Test-selected.',
        initialization='Same model/Adam/scaler for all arms; Python/CPU/CUDA RNG reset to144 or244.',
        pairing='Within each RNG seed, identical initial RNG and precomputed800-instance dataorder across arms.',
        schedule='Each RNG seed free40 vs K17-constrained20 then free20.',
        hypotheses='Distinguish model initialization seed effects from fine-tune stochasticity.',
        evaluation='Train20 free at0/10/20/30/40; frozen newvalidation20 only final40; no Test rerun.',
        success='Both RNG replicates finalTrain reward delta>0 andU0; report validation direction, lateness, failures.',
        no_tuning='Keep target17,20+20 and final40 regardless of results; no checkpoint selection.',
        budget='160totalepochs/3200updates/about100GPUminutes',
        stop='Exception,NaN/Inf,20consecutiveAMPskips,sourcehashchange',
        input_hashes=hashes, data_orders=orders, api_calls=0))
    pilot.write(OUT / 'status.json', dict(stage='prepared_cpu_only', gpu_started=False))
    print('Prepared RNG144/244 matched160epoch protocol; GPU not started.')


def main():
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    protocol = pilot.read(OUT / 'protocol.json')
    assert pilot.read(OUT / 'status.json')['stage'] == 'prepared_cpu_only'
    assert all(pilot.digest(ROOT / path) == value for path, value in protocol['input_hashes'].items())
    cfg = protocol['config']
    rows = pilot.read(SOURCE / 'split.json')['train']
    lookup = {r['name']: r for r in rows}
    assert set(lookup) == set(cfg['trainer']['train_instances']) and len(rows) == 20
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    initial = None
    for seed in RNG_SEEDS:
        folder = OUT / f'rng{seed}'
        folder.mkdir()
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        state = pilot.rng()
        order = protocol['data_orders'][str(seed)]
        for arm in ('control', 'curriculum'):
            destination = folder / arm
            destination.mkdir()
            model = pilot.Policy(**cfg['model']).cuda().train()
            model.load_state_dict(saved['model_state_dict'], strict=True)
            opt = torch.optim.Adam(model.parameters(), **cfg['optimizer']['optimizer'])
            opt.load_state_dict(copy.deepcopy(saved['optimizer_state_dict']))
            scaler = torch.amp.GradScaler('cuda')
            scaler.load_state_dict(copy.deepcopy(saved['scaler']))
            pilot.restore(state)
            start = pilot.evaluate(model, rows, cfg, None)
            if initial is None:
                initial = start
            assert start == initial
            pilot.write(destination / 'free_epoch000.json', start)
            skips = 0
            for epoch in range(1, 41):
                assert all(pilot.digest(ROOT / path) == value for path, value in protocol['input_hashes'].items())
                model.target = TARGET if arm == 'curriculum' and epoch <= 20 else None
                model.new_customer_epsilon = .01
                for batch_index in range(20):
                    update = (epoch - 1) * 20 + batch_index
                    name = order[update]
                    inst, public = pilot.trainer.load(lookup[name])
                    model.fallback_count = 0
                    pilot.write(OUT / 'status.json', dict(stage='training', pid=os.getpid(), rng_seed=seed,
                        arm=arm, epoch=epoch, update=update, target=model.target))
                    summary, _ = pilot.trainer.train_batch(model, opt, scaler, inst, public,
                        cfg['environment'], min(128, 2500 - batch_index * 128), 100, True)
                    assert math.isfinite(summary['loss']) and all(torch.isfinite(p).all() for p in model.parameters())
                    skips = skips + 1 if summary['optimizer_step_skipped'] else 0
                    assert skips < 20
                    pilot.trainer.append(destination / 'training.jsonl', dict(epoch=epoch, update=update + 1,
                        name=name, target=model.target, fallback_count=model.fallback_count, **summary))
                if epoch % 10 == 0:
                    result = pilot.evaluate(model, rows, cfg, None)
                    pilot.write(destination / f'free_epoch{epoch:03d}.json', result)
                    torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(), optimizer_state_dict=opt.state_dict(),
                        scaler=scaler.state_dict(), rng=pilot.rng()), destination / f'checkpoint-{epoch}.pt')
                    print('MILESTONE', seed, arm, epoch, means(result), flush=True)
            del model, opt, scaler
            torch.cuda.empty_cache()
        train = {arm: pilot.read(folder / arm / 'free_epoch040.json') for arm in ('control', 'curriculum')}
        pilot.write(folder / 'train_comparison.json', compare(train))
        val = {}
        for arm in ('control', 'curriculum'):
            model = pilot.Policy(**cfg['model']).cuda().eval()
            model.load_state_dict(torch.load(folder / arm / 'checkpoint-40.pt', map_location='cpu', weights_only=False)['model_state_dict'])
            val[arm] = validation(model, cfg)
            pilot.write(folder / arm / 'validation20.json', val[arm])
            del model
        pilot.write(folder / 'validation_comparison.json', compare(val))
        print('RNG_COMPLETE', seed, compare(train)['groups']['all'], compare(val)['groups']['all'], flush=True)
    pilot.write(OUT / 'status.json', dict(stage='complete', rng_seeds=list(RNG_SEEDS), total_epochs=160, api_calls=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-gpu', action='store_true')
    args = parser.parse_args()
    if args.run_gpu:
        try:
            main()
        except Exception as error:
            pilot.write(OUT / 'failure.json', dict(error=repr(error)))
            raise
    else:
        prepare()
