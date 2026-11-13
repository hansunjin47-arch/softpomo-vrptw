"""Nested train80, unchanged test80; History first, then fresh Pure RL1000."""
import copy
from pathlib import Path
import shutil
import subprocess
import sys
from run_rc101_train40_extension import read, signature
from eval_fixed_v2_lkh import ROOT, write, sha
from generate_rc101_event_pilot import main as generate
from enrich_rc101_observations import enrich
from eval_rc101_pilot_history_split import main as history

OLD = ROOT/'data/rc101_train40_test80_20260928'
DATA = ROOT/'data/rc101_train80_test80_20260929'
OUT = ROOT/'result/rc101_pure_train80_fresh1000_20260929'
HISTORY = ROOT/'analysis/rc101_history_train80_20260929'


def main():
    assert not (OUT/'pipeline.json').exists(), 'Run already exists'
    OUT.mkdir(exist_ok=True)
    write(OUT/'pipeline.json',dict(stage='generation'))
    raw = ROOT/'data/rc101_extra_train40_raw_20260929'
    observed = ROOT/'data/rc101_extra_train40_discrete_20260929'
    generate('rc101',raw,seed=20260929,per_kind=20,case_prefix='Y')
    new = read(raw/'manifest.json')
    write(raw/'split.json',dict(train=[dict(name=c['name'],kind=c['kind'],sha256=c['file_sha256'])
                                    for c in new['cases']],test=[]))
    enrich(raw,raw/'split.json',observed)
    original, extra = read(OLD/'manifest.json'),read(observed/'manifest.json')
    assert not {signature(c) for c in original['cases']} & {signature(c) for c in extra['cases']}
    assert read(OLD/'observation_rules.audit.json')==read(observed/'observation_rules.audit.json')
    DATA.mkdir(exist_ok=False)
    for source,manifest in [(OLD,original),(observed,extra)]:
        for case in manifest['cases']:
            for suffix in ['.txt','.observations.json']:
                path=source/(case['name']+suffix)
                shutil.copy2(path,DATA/path.name)
                assert sha(path)==sha(DATA/path.name)
    for name in ['RC101_BASE.txt','observation_rules.audit.json']:
        shutil.copy2(OLD/name,DATA/name)
    write(DATA/'manifest.json',dict(original,cases=original['cases']+extra['cases'],
        added_seed=20260929,added_source=str(observed),original_source=str(OLD)))
    prior=read(OLD/'split.json')
    split=copy.deepcopy(prior)
    split['train']+=read(observed/'split.json')['train']
    split.update(data=str(DATA),generator_manifest_hash=sha(DATA/'manifest.json'))
    assert split['test']==prior['test'] and split['train'][:40]==prior['train']
    assert len(split['train'])==80 and len(split['test'])==80
    assert len({r['name'] for r in split['train']+split['test']})==160
    for group in ['train','test']:
        for kind in ['RAIN','ACCIDENT']:
            assert sum(r['kind']==kind for r in split[group])==40
        for row in split[group]:
            assert sha(DATA/(row['name']+'.txt'))==row['sha256']
    write(DATA/'split.json',split)
    write(DATA/'validation.json',dict(train=80,test=80,added=40,
        original_instances_and_observations_byte_identical=True,
        original_test_split_unchanged=True,no_duplicate_events=True,
        added_validation=read(observed/'validation.json')))
    write(OUT/'pipeline.json',dict(stage='history',history=str(HISTORY)))
    history(DATA,HISTORY,'rc101',time_limit=600,service_precheck=True,split_path=DATA/'split.json')
    result=read(HISTORY/'summary.json')
    if result.get('status')!='evaluated' or result.get('test_evaluated')!=80:
        write(OUT/'pipeline.json',dict(stage='history_failed_training_not_started',history=result))
        return
    assert result['groups']['ALL']['means']['served']==100
    from train_rc101_train40_fresh import main as train
    train(DATA,OUT,HISTORY)
    shutil.copy2(Path(__file__),OUT/'provenance'/Path(__file__).name)
    subprocess.run([sys.executable,str(ROOT/'analysis/summarize_rc101_pilot_final.py'),
                    '--run',str(OUT)],check=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        write(OUT/'error.json',dict(error=repr(exc)))
        raise
