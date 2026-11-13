"""Sequential GPU training jobs with durable commands, logs and process status."""
import datetime
import json
import pathlib
import subprocess
import sys

ROOT=pathlib.Path(__file__).resolve().parents[1]
run=pathlib.Path(sys.argv[1]).resolve()
run.mkdir(parents=True,exist_ok=True)
status={'jobs':[],'settings':{'epochs':1000,'save_interval':500,'seed':42,
    'batch_size':32,'pomo':100,'start_mode':'random','cluster_mode':'kim2006',
    'automatic_refresh':False,'fewshot':False,'cache_version':'existing v1 including original C109'}}

def save():
    (run/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')

for family in ['c1','rc1','r1']:
    command=[sys.executable,'-u',str(ROOT/'analysis/train_kim_cached_only.py'),
        '--benchmark',family,'--cluster-mode','kim2006','--epochs','1000','--config','E',
        '--batch-size','32','--pomo','100','--start-mode','random','--seed','42',
        '--acc-ratio','0.333333','--experience-refresh-epoch','-1',
        '--llm-cache-dir',str(ROOT/f'result_soft/llm_cache_kim_initial_v1_20260917_214827/{family}'),
        '--result-dir',str(run/family),'--model','gpt-oss:20b']
    job={'family':family,'command':command,'started':datetime.datetime.now().isoformat()}
    status['jobs'].append(job)
    with (run/f'{family}_out.log').open('w',encoding='utf-8') as stdout, (run/f'{family}_err.log').open('w',encoding='utf-8') as stderr:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr)
        job['pid']=process.pid
        save()
        job['exit_code']=process.wait()
        job['finished']=datetime.datetime.now().isoformat()
        save()
        if job['exit_code']:
            break
