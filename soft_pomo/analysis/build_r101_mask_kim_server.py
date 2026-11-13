"""Freeze API-free R101 Kim training jobs; evaluate downloaded models locally."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]
NAME='r101_pair100_l5_mask_kim_s42_46_20261106'
OUT=ROOT/'server_export'/NAME


def main():
    OUT.mkdir(exist_ok=False)
    for p in list(ROOT.glob('*.py'))+list((ROOT/'experiments').rglob('*.py')):
        dest=OUT/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest)
    source=ROOT/'server_export/pair_returnmask_cr_v100_200_l5_s42_46_20261031/datasets/r101'
    shutil.copytree(source,OUT/'dataset')
    artifact=ROOT/'result/r101_pair100_l5_mask_kim_cache_20261106'
    shutil.copytree(artifact,OUT/'artifacts',ignore=shutil.ignore_patterns('calls'))
    assert json.loads((artifact/'complete.json').read_text())['train_count']==20
    default=Path('result/rc101_learned_first_train20_20261017/late5/config.json')
    (OUT/default).parent.mkdir(parents=True)
    shutil.copy2(artifact/'config.json',OUT/default)
    # Only the isolated server copy skips evaluation; local training stays unchanged.
    p=OUT/'experiments/concurrent_fleet_pair/train.py'
    code=p.read_text()
    marker="        status('evaluation'); model.eval(); results=[]"
    assert code.count(marker)==1
    code=code.replace(marker,"        write(folder/'training_complete.json',dict(epoch=tp['epochs'],test_count=0,evaluation_pending=True))\n        write(args.out/'status.json',dict(stage='training_complete',evaluation_pending=True))\n        return\n"+marker)
    p.write_text(code,encoding='utf-8',newline='\n')
    shutil.copy2(ROOT/'analysis/r101_mask_kim_server_train.py',OUT/'run_training.py')
    submit=(ROOT/'analysis/submit_seq_v100_lanes.sh').read_text().replace('-eq 10','-eq 5').replace('seq 0 9','seq 0 4').replace('10 jobs','5 jobs')
    (OUT/'submit.sh').write_text(submit,encoding='ascii',newline='\n')
    (OUT/'tasks.txt').write_text(''.join(f'{s}\n' for s in range(42,47)),encoding='ascii',newline='\n')
    task=(ROOT/'analysis/pair_return_mask_task.slurm').read_text()
    task=task.replace('sj_pair_mask_l5','sj_r_kim_mask_l5').replace('"$index" -lt 20','"$index" -lt 5')
    start=task.index('read -r case weight seed denominator')
    end=task.index('args=()',start)
    task=task[:start]+'seed=$(sed -n "$((index+1))p" tasks.txt)\nout="runs/seed$seed"\n'+task[end:]
    start=task.index(' python -u -m experiments.concurrent_fleet_pair.train')
    end=task.index('\ncode=$?',start)
    task=task[:start]+' python -u run_training.py --out "$out" --seed "$seed" "${args[@]}"'+task[end:]
    start=task.index("    python3 -c '")
    task=task[:start]+'''    python3 -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); c=json.load(open(p/"late5/training_complete.json")); assert c["epoch"]==1000 and c["evaluation_pending"]; assert (p/"late5/checkpoint-1000.pt").is_file()' "$out"
fi
'''
    (OUT/'task.slurm').write_text(task,encoding='ascii',newline='\n')
    hashes={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file()}
    (OUT/'manifest.json').write_text(json.dumps(dict(sha256=hashes,seeds=list(range(42,47)),evaluation='local only; not included in server job',vehicle_weight=100,late=5,mask=True),indent=2),encoding='utf-8')
    archive=OUT.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:tar.add(OUT,arcname=NAME)
    Path(str(archive)+'.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n',encoding='ascii',newline='\n')
    print('BUILT',archive)


if __name__=='__main__':main()
