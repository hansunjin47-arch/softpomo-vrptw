"""Package five independent sequential RC101 pure-RL runs."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'server_export/rc101_seq_train20_late15_seeds42_46_20261022'
TEMPLATE=ROOT/'server_export/rc101_bks_late5_20_train20_20261014/late5'


def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2),encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=False)
    for seed in range(42,47):
        dest=OUT/f'seed{seed}';dest.mkdir()
        shutil.copytree(TEMPLATE/'data',dest/'data')
        shutil.copy2(TEMPLATE/'split.json',dest/'split.json')
        for p in TEMPLATE.glob('*.py'):
            source=ROOT/p.name
            assert source.exists(),source
            shutil.copy2(source,dest/p.name)
        (dest/'server').mkdir()
        shutil.copy2(ROOT/'server/train_cached.py',dest/'server/train_cached.py')
        cfg=json.loads((TEMPLATE/'configs/pure.json').read_text())
        cfg['trainer'].update(seed=seed,epochs=1000,n_mc_samples=1,pomo_auto=False,eval_top_k=None)
        cfg['environment']['late_penalty']=15
        cfg['model'].update(eval_type='argmax',learn_first_customer=False)
        write(dest/'configs/pure.json',cfg)
        write(dest/'provenance.json',dict(train_seed=seed,eval_seed='42 + test index; all100 starts, greedy, aug8',
            objective='K/14 + (travel_minutes + 15*late_minutes)/1696.95',
            source='current local sequential modules, including depot-to-first leg correction',
            comparison_warning='Do not pool with archived pre-fix runs',first_customer_entropy=False,
            training='fresh1000; checkpoint500 retained; pure RL only; no LLM calls'))
        write(dest/'bundle_manifest.json',dict(sha256={p.relative_to(dest).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in dest.rglob('*') if p.is_file()}))
    shutil.copy2(ROOT/'server/eval_pure_tt_bundle.py',OUT/'eval_pure_tt_bundle.py')
    shutil.copy2(ROOT/'server/seq_late15_seeds.slurm',OUT/'seq_late15_seeds.slurm')
    archive=OUT.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as t:t.add(OUT,arcname=OUT.name)
    Path(str(archive)+'.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n',encoding='ascii')
    print(archive)


if __name__=='__main__':main()
