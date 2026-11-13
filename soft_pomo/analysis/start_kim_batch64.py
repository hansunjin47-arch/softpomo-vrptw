"""Start fresh parallel batch64 jobs from a stopped batch32 run's commands."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
old=Path(sys.argv[1]).resolve()
prior=json.loads((old/'status.json').read_text(encoding='utf-8'))
stamp=datetime.datetime.now()
run=root/('result_soft_kim_1000_b64_'+stamp.strftime('%Y%m%d_%H%M%S'))
run.mkdir(exist_ok=False)
status={'settings':dict(prior['settings'],batch_size=64,execution='parallel',resume=False),
        'previous_run':str(old),'jobs':[]}
for previous in prior['jobs']:
    family=previous['family']
    command=previous['command'].copy()
    assert '--resume' not in command
    command[command.index('--batch-size')+1]='64'
    command[command.index('--result-dir')+1]=str(run/family)
    with (run/f'{family}_out.log').open('x',encoding='utf-8') as stdout, (run/f'{family}_err.log').open('x',encoding='utf-8') as stderr:
        p=subprocess.Popen(command,cwd=root,stdout=stdout,stderr=stderr,
                           creationflags=subprocess.CREATE_NO_WINDOW)
    status['jobs'].append(dict(family=family,command=command,pid=p.pid,
                              started=datetime.datetime.now().isoformat()))
    (run/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(f'{family}: PID={p.pid}',flush=True)
prior['stopped_at']=stamp.isoformat()
prior['stop_reason']='User requested fresh batch64 parallel restart'
prior['replacement_run']=str(run)
for job in prior['jobs']:
    job['status']='stopped_by_user'
(old/'status.json').write_text(json.dumps(prior,indent=2),encoding='utf-8')
print('OUTPUT',run,flush=True)
