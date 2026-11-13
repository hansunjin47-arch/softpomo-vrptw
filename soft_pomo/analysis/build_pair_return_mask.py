"""Build isolated C/R server sweep and RC local configurations."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]
NAME='pair_returnmask_cr_v100_200_l5_s42_46_20261031'
OUT=ROOT/'server_export'/NAME


def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2),encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=False)
    files=list(ROOT.glob('*.py'))+list((ROOT/'experiments').rglob('*.py'))
    for p in files:
        target=OUT/p.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target)
    template=ROOT/'result/rc101_learned_first_train20_20261017/late5/config.json'
    target=OUT/template.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(template,target)
    for case,den,horizon in [('c101',828.94,1236),('r101',1650.80,230),('rc101',1696.95,240)]:
        if case=='rc101':
            src=ROOT/'result/rc101_phase2_online_gpt_train20_v2_2_20261018/source'
            cfg=json.loads(template.read_text())
        else:
            src=ROOT/f'server_export/cr101_seq_l5_10_15_20_s42_46_20261023/{case}/late5/seed42'
            original=json.loads((src/'configs/pure.json').read_text())
            cfg=json.loads(template.read_text())
            cfg['environment']=copy.deepcopy(original['environment'])
            cfg['trainer']['train_instances']=original['trainer']['train_instances']
        split=json.loads((src/'split.json').read_text())
        assert len(split['train'])==20 and len(split['test'])==80
        assert abs(horizon*cfg['environment']['TT_max']-den)<1e-6
        assert abs(horizon*cfg['environment']['Lt_max']-den)<1e-6
        write(OUT/f'datasets/{case}/split.json',split)
        for row in split['train']+split['test']:
            p=src/'data'/(row['name']+'.txt')
            assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
            for data in (p,p.with_suffix('.observations.json')):
                dest=OUT/f'datasets/{case}/data'/data.name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(data,dest)
        write(OUT/f'configs/{case}.json',cfg)
    tasks=[f'{case} {weight} {seed} {den}' for seed in range(42,47)
           for case,den in [('c101',828.94),('r101',1650.80)] for weight in (100,200)]
    (OUT/'tasks.txt').write_text('\n'.join(tasks)+'\n',encoding='ascii',newline='\n')
    submit=(ROOT/'analysis/submit_seq_v100_lanes.sh').read_text().replace('-eq 10','-eq 20').replace('seq 0 9','seq 0 19').replace('10 jobs','20 jobs')
    (OUT/'submit.sh').write_text(submit,encoding='ascii',newline='\n')
    (OUT/'task.slurm').write_text((ROOT/'analysis/pair_return_mask_task.slurm').read_text(),encoding='ascii',newline='\n')
    write(OUT/'manifest.json',dict(tasks=tasks,vehicle_weights=[100,200],late=5,seeds=list(range(42,47)),
          return_mask='sole capacity provider; post-dispatch; train and test',
          evaluation='test80 single greedy, no augmentation',
          sha256={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file()}))
    archive=OUT.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar: tar.add(OUT,arcname=NAME)
    Path(str(archive)+'.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n',encoding='ascii',newline='\n')
    print('BUILT',archive,len(tasks),'tasks')


if __name__=='__main__': main()
