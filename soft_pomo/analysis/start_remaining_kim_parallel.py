"""Start RC1/R1 alongside an existing C1 job after stopping its queue manager."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
run=Path(sys.argv[1]).resolve()
path=run/'status.json'
status=json.loads(path.read_text(encoding='utf-8'))
assert [job['family'] for job in status['jobs']]==['c1'], 'Jobs already started'
template=status['jobs'][0]['command']
status['settings']['execution']='parallel'
status['settings']['queue_manager_stopped']=True
for family in ['rc1','r1']:
    command=template.copy()
    command[command.index('--benchmark')+1]=family
    command[command.index('--llm-cache-dir')+1]=str(root/f'result_soft/llm_cache_kim_initial_v1_20260917_214827/{family}')
    command[command.index('--result-dir')+1]=str(run/family)
    with (run/f'{family}_out.log').open('x',encoding='utf-8') as stdout, (run/f'{family}_err.log').open('x',encoding='utf-8') as stderr:
        process=subprocess.Popen(command,cwd=root,stdout=stdout,stderr=stderr,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
    status['jobs'].append(dict(family=family,command=command,pid=process.pid,
                              started=datetime.datetime.now().isoformat()))
    path.write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(f'{family}: started PID={process.pid}',flush=True)
