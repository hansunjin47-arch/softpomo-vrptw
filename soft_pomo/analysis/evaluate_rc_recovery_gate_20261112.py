"""Exploratory Train-only recovery gate, with fixed validation assessment."""
import argparse
from pathlib import Path
import torch
from analysis import train_rc_k17_curriculum_20261112 as pilot
from analysis.audit_rc_curriculum_free_phase_20261112 import RUNS
from analysis.train_rc_curriculum_confirmatory_20261112 import means, validation, compare

ROOT = pilot.ROOT
OUT = ROOT / 'result/rc_recovery_gate_20261112'


def prepare():
    assert not OUT.exists()
    selected = {}
    hashes = {}
    for label, folder in RUNS.items():
        base = ROOT / 'result' / folder
        initial_path = base / 'curriculum/free_epoch000.json'
        if not initial_path.exists():
            initial_path = base / 'initial_train20.json'
        early_path = base / 'curriculum/free_epoch020.json'
        initial = means(pilot.read(initial_path))
        early = means(pilot.read(early_path))
        epoch = 20 if early['unserved'] == 0 and early['reward'] > initial['reward'] else 40
        checkpoint = base / f'curriculum/checkpoint-{epoch}.pt'
        if 'rng' in label:
            config_path = base.parent / 'protocol.json'
        else:
            config_path = base / 'protocol.json'
        cfg = pilot.read(config_path)['config']
        if label in ('model42', 'model44', 'model45'):
            fixed = ROOT / f'result/rc_curriculum_validation_20261112/seed{label[5:]}/curriculum.json'
        else:
            fixed = base / 'curriculum/validation20.json'
        for path in (initial_path, early_path, checkpoint, config_path, fixed):
            hashes[str(path.relative_to(ROOT))] = pilot.digest(path)
        selected[label] = dict(initial=initial, early=early, selected_epoch=epoch,
            checkpoint=str(checkpoint.relative_to(ROOT)), config=cfg,
            fixed40_validation=str(fixed.relative_to(ROOT)), training_epochs_saved=40-epoch)
    for path in (Path(__file__), ROOT / 'analysis/train_rc_k17_curriculum_20261112.py',
                 ROOT / 'analysis/train_rc_curriculum_confirmatory_20261112.py',
                 ROOT / 'result/rc_curriculum_validation_20261112/protocol.json'):
        hashes[str(path.relative_to(ROOT))] = pilot.digest(path)
    OUT.mkdir()
    pilot.write(OUT / 'selection_manifest.json', dict(rule='Select20 iff free Train20 U=0 and mean reward strictly exceeds epoch0; otherwise select40.',
        threshold=0, runs=selected, input_hashes=hashes,
        assessment='All changed selections validation reward >= fixed40 and U=0 required for nondegrading savings.',
        caveat='Exploratory post-hoc gate, five model seeds plus two RNG repeats. Fixed40 conclusions preserved. No Test/API/training.',
        validation_reuse='Reuse existing40; evaluate changed20 only. Never change gate after outcomes.'))
    for label, entry in selected.items():
        print(label, entry['selected_epoch'], entry['initial']['reward'], entry['early']['reward'])


def evaluate():
    torch.set_num_threads(1)
    protocol = pilot.read(OUT / 'selection_manifest.json')
    assert all(pilot.digest(ROOT / p) == h for p, h in protocol['input_hashes'].items())
    results = {}
    for label, entry in protocol['runs'].items():
        fixed = pilot.read(ROOT / entry['fixed40_validation'])
        if entry['selected_epoch'] == 40:
            selected = fixed
        else:
            model = pilot.Policy(**entry['config']['model']).cuda().eval()
            model.load_state_dict(torch.load(ROOT / entry['checkpoint'], map_location='cpu', weights_only=False)['model_state_dict'], strict=True)
            selected = validation(model, entry['config'])
            del model
            torch.cuda.empty_cache()
        pilot.write(OUT / f'{label}_selected_validation.json', selected)
        result = compare(dict(control=fixed, curriculum=selected))
        result['selected_epoch'] = entry['selected_epoch']
        result['nondegrading'] = result['means']['curriculum']['reward'] >= result['means']['control']['reward'] and result['means']['curriculum']['unserved'] == 0
        results[label] = result
        print(label, result['groups']['all'], flush=True)
    changed = [r for r in results.values() if r['selected_epoch'] == 20]
    pilot.write(OUT / 'report.json', dict(results=results,
        nondegrading_savings=all(r['nondegrading'] for r in changed), changed_runs=len(changed),
        total_training_epochs_saved=20*len(changed), caveat=protocol['caveat']))
    pilot.write(OUT / 'status.json', dict(stage='complete', api_calls=0, new_training_epochs=0, test_accessed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    evaluate() if args.evaluate else prepare()
