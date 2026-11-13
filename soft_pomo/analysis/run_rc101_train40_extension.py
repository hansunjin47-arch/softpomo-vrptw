"""Add 20 training events, solve train40 History, then continue Pure1000."""
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_fixed_v2_lkh import write, sha
from generate_rc101_event_pilot import main as generate
from enrich_rc101_observations import enrich
from eval_rc101_pilot_history_split import main as history

OUT = ROOT/'result/rc101_pure_train40_continue2000_20260928'
DATA = ROOT/'data/rc101_train40_test80_20260928'
HISTORY = ROOT/'analysis/rc101_history_train40_20260928'
SOURCE = ROOT/'result/rc101_pilot_discrete_pure1000_20260923_115506'
OLD = ROOT/'data/rc101_event_pilot_discrete_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def signature(case):
    e = case['event']
    return (e['type'], e['trigger_time'], e['duration'], e['multiplier'], tuple(sorted(e['nodes'])))


def main():
    assert not (OUT/'pipeline.json').exists(), 'Do not overwrite an existing run'
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT/'pipeline.json', dict(stage='generation'))
    raw = ROOT/'data/rc101_extra_train20_raw_20260928'
    observed = ROOT/'data/rc101_extra_train20_discrete_20260928'
    generate('rc101', raw, seed=20260928, per_kind=10, case_prefix='X')
    new_manifest = read(raw/'manifest.json')
    new_split = dict(train=[dict(name=c['name'], kind=c['kind'], sha256=c['file_sha256'])
                           for c in new_manifest['cases']], test=[])
    write(raw/'split.json', new_split)
    enrich(raw, raw/'split.json', observed)
    original_manifest = read(OLD/'manifest.json')
    extra_manifest = read(observed/'manifest.json')
    assert not {signature(c) for c in original_manifest['cases']} & {
        signature(c) for c in extra_manifest['cases']}
    original_split = read(OLD/'split.json')
    assert original_split == read(SOURCE/'provenance/split.json')
    DATA.mkdir(exist_ok=False)
    for source, manifest in [(OLD, original_manifest), (observed, extra_manifest)]:
        for c in manifest['cases']:
            for suffix in ['.txt', '.observations.json']:
                path = source/(c['name']+suffix)
                shutil.copy2(path, DATA/path.name)
                assert sha(path) == sha(DATA/path.name)
    shutil.copy2(raw/'RC101_BASE.txt', DATA/'RC101_BASE.txt')
    shutil.copy2(OLD/'observation_rules.audit.json', DATA/'observation_rules.audit.json')
    assert read(OLD/'observation_rules.audit.json') == read(observed/'observation_rules.audit.json')
    combined = dict(original_manifest, cases=original_manifest['cases']+extra_manifest['cases'],
                    added_seed=20260928, added_source=str(observed), original_source=str(OLD))
    write(DATA/'manifest.json', combined)
    split = copy.deepcopy(original_split)
    split['train'] += read(observed/'split.json')['train']
    split.update(data=str(DATA), generator_manifest_hash=sha(DATA/'manifest.json'))
    assert len(split['train'])==40 and len(split['test'])==80
    assert split['test'] == original_split['test']
    assert len({c['name'] for c in split['train']+split['test']}) == 120
    for group in ['train','test']:
        for c in split[group]:
            assert sha(DATA/(c['name']+'.txt')) == c['sha256']
    write(DATA/'split.json', split)
    write(DATA/'validation.json', dict(train=40, test=80, added=20,
        original_instances_and_observations_byte_identical=True,
        original_test_split_unchanged=True, no_duplicate_events=True,
        added_validation=read(observed/'validation.json')))
    write(OUT/'pipeline.json', dict(stage='history', data=str(DATA), history=str(HISTORY)))
    history(DATA, HISTORY, 'rc101', time_limit=600, service_precheck=True,
            split_path=DATA/'split.json')
    result = read(HISTORY/'summary.json')
    if result.get('status') != 'evaluated' or result.get('test_evaluated') != 80:
        write(OUT/'pipeline.json', dict(stage='history_failed_training_not_started', history=result))
        return
    assert result['groups']['ALL']['means']['served'] == 100
    import torch
    import train_vrptw as training
    from train_rc101_event_pilot import PilotTrainer
    config = read(SOURCE/'provenance/config.json')
    checkpoint = SOURCE/'model/RC101/checkpoint-1000.pt'
    tp = dict(config['trainer'], data_dir=str(DATA), epochs=2000,
        train_instances=[c['name'] for c in split['train']],
        test_instances=[c['name'] for c in split['test']], result_dir=str(OUT/'model'),
        model_load=dict(enable=True, path=str(checkpoint.parent), epoch=1000),
        logging=dict(model_save_interval=500, log_interval=10))
    assert tp['train_batch_size']==64 and tp['train_episodes']==2500
    provenance = OUT/'provenance'
    provenance.mkdir(exist_ok=False)
    for name in ['manifest.json','split.json','validation.json','observation_rules.audit.json']:
        shutil.copy2(DATA/name, provenance/name)
    for path in [Path(__file__), ROOT/'train_vrptw.py', ROOT/'vrptw_env.py',
                 ROOT/'event_observation.py', ROOT/'analysis/train_rc101_event_pilot.py']:
        shutil.copy2(path, provenance/path.name)
    write(provenance/'config.json', dict(config, trainer=tp,
        initialized_from=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        schedule='20 events epochs 1..1000; mixed 40 events epochs 1001..2000'))
    torch.set_num_threads(4)
    training._set_seed(42)
    trainer = PilotTrainer(config['environment'], config['model'], config['optimizer'], tp)
    state = torch.load(checkpoint, map_location=trainer.device, weights_only=False)
    assert state['epoch']==1000 and trainer.start_epoch==1001
    trainer.model.load_state_dict(state['model_state_dict'], strict=True)
    trainer.optimizer.load_state_dict(state['optimizer_state_dict'])
    del state
    write(OUT/'pipeline.json', dict(stage='training', start_epoch=1001, end_epoch=2000,
                                   train=40, test=80, history=str(HISTORY)))
    print('START Pure RL continuation 1001..2000, mixed train40, b64, no LLM', flush=True)
    trainer.run()
    write(OUT/'pipeline.json', dict(stage='complete', epoch=2000, history=str(HISTORY)))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        if OUT.exists():
            write(OUT/'error.json', dict(error=repr(exc)))
        raise
