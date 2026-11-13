"""Build nested C/R train40/80 pools and 16 offline training/evaluation jobs."""
import copy
from pathlib import Path
import shutil
import sys
import tarfile

from train_cached import read, write, sha, verify_bundle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'analysis'))
from generate_rc101_event_pilot import main as generate, ge
from enrich_rc101_observations import enrich

OUT = ROOT/'server_export/cr101_bks_train40_80_late1_5_10_20_20261016'
AUDIT = ROOT/'data/cr101_train_expansion_20261016'


def main():
    OUT.mkdir(exist_ok=False)
    AUDIT.mkdir(exist_ok=False)
    cases = []
    for inst, alpha, k0, t0 in [('c101',2,10,828.94), ('r101',.5,19,1650.80)]:
        source = ROOT/'result/bks_normalized_train20_20261013'/inst
        verify_bundle(source)
        raw, observed = AUDIT/(inst+'_raw'), AUDIT/(inst+'_observed')
        generate(inst, raw, seed=20261016, per_kind=30, case_prefix='X', diagnostics=False)
        manifest = read(raw/'manifest.json')
        data = ge.load_raw(manifest['source'])
        for row in manifest['cases']:
            ev = row['event']
            row['original_multiplier'] = ev['multiplier']
            ev['multiplier'] = round(1+alpha*(ev['multiplier']-1),6)
            p=raw/(row['name']+'.txt')
            ge.write_scenario(data,[ev],row['name'],str(p))
            row['file_sha256']=sha(p)
        manifest['strength_alpha']=alpha
        manifest['multiplier']={kind:[1+alpha*(v-1) for v in bounds]
                                for kind,bounds in manifest['multiplier'].items()}
        write(raw/'manifest.json',manifest)
        shutil.copy2(source/'raw_audit/base_reference.json',raw/'base_reference.json')
        write(raw/'split.json',dict(train=[dict(name=r['name'],kind=r['kind'],sha256=r['file_sha256'])
                                          for r in manifest['cases']],test=[]))
        enrich(raw,raw/'split.json',observed)
        old_split=read(source/'split.json')
        added=read(observed/'split.json')['train']
        old_manifest=read(source/'data/manifest.json')
        def signature(row):
            e=row['event']
            return (e['type'],e['trigger_time'],e['duration'],e['multiplier'],tuple(sorted(e['nodes'])))
        old_signatures={signature(r) for r in old_manifest['cases']}
        new_signatures={signature(r) for r in read(observed/'manifest.json')['cases']}
        assert len(new_signatures)==60 and not old_signatures & new_signatures
        previous={r['name'] for r in old_split['train']}
        for size in (40,80):
            extra=[r for kind in ('RAIN','ACCIDENT') for r in [e for e in added if e['kind']==kind][:(size-20)//2]]
            split=copy.deepcopy(old_split)
            split['train']+=extra
            names={r['name'] for r in split['train']}
            assert len(names)==size and previous<=names
            assert not names & {r['name'] for r in split['test']}
            previous=names
            for weight in (1,5,10,20):
                case=f'{inst}_train{size}_late{weight}'
                cases.append(case)
                target=OUT/case
                (target/'server').mkdir(parents=True)
                (target/'data').mkdir()
                for p in source.glob('*.py'):
                    shutil.copy2(p,target/p.name)
                shutil.copy2(source/'server/train_cached.py',target/'server/train_cached.py')
                for row in split['train']+split['test']:
                    origin=observed if row in extra else source/'data'
                    for suffix in ('.txt','.observations.json'):
                        p=origin/(row['name']+suffix)
                        shutil.copy2(p,target/'data'/p.name)
                    assert sha(target/'data'/(row['name']+'.txt'))==row['sha256']
                cfg=read(source/'configs/pure.json')
                cfg['trainer']['train_instances']=[r['name'] for r in split['train']]
                cfg['environment']['late_penalty']=weight
                assert cfg['trainer']['epochs']==1000 and cfg['trainer']['seed']==42
                assert cfg['trainer']['train_episodes']==2500 and cfg['trainer']['train_batch_size']==128
                write(target/'configs/pure.json',cfg)
                write(target/'split.json',split)
                write(target/'provenance.json',dict(source_manifest_sha256=sha(source/'bundle_manifest.json'),
                    added_manifest_sha256=sha(observed/'manifest.json'),generator_seed=20261016,
                    objective=f'K/{k0}+(T+{weight}*L)/{t0}',lambda_late=weight,train_size=size,
                    event_intensity_scale=alpha,nested_train_pools=True,fresh_training=True,
                    test80_unchanged=True,train_episodes_per_epoch=2500))
                files=[p for p in target.rglob('*') if p.is_file()]
                for p in files:
                    assert not any(s in p.read_text(encoding='utf-8') for s in ('sk-or-v1-','sk-ant-api','BEGIN OPENSSH PRIVATE KEY'))
                write(target/'bundle_manifest.json',dict(sha256={p.relative_to(target).as_posix():sha(p) for p in files}))
                verify_bundle(target)
                print('VERIFIED',case,flush=True)
    # Round-robin instances and pool sizes within each lambda; Slurm may reorder tasks.
    cases=[f'{inst}_train{size}_late{w}' for w in (1,5,10,20) for size in (40,80) for inst in ('c101','r101')]
    write(OUT/'cases.json',cases)
    script=(ROOT/'server/cr_bks_max2.slurm').read_text()
    script=script.replace('--job-name=sj_cr_bks','--job-name=sj_cr_pool')
    script=script.replace('--array=0-3%2','--array=0-15%2')
    script=script.replace('CASES=(c101_late5 r101_late5 c101_late20 r101_late20)','CASES=('+' '.join(cases)+')')
    (OUT/'cr_pool_max2.slurm').write_text(script,encoding='utf-8',newline='\n')
    shutil.copy2(ROOT/'server/eval_pure_tt_bundle.py',OUT/'eval_pure_tt_bundle.py')
    archive=OUT.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(OUT,arcname=OUT.name)
    Path(str(archive)+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii',newline='\n')
    print('READY',archive,flush=True)


if __name__=='__main__':
    main()
