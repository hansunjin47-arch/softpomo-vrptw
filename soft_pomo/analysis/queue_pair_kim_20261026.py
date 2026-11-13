"""Launch Kim only after the std-ON run has completed training and test80."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.concurrent_fleet_pair_kim.run import audit
from experiments.concurrent_fleet_bias_guided.support import ROOT, read
from experiments.concurrent_fleet_pair.io import write

PREVIOUS=ROOT/'result/pair300_epsilon005_floor001_std_20261026'
OUT=ROOT/'result/pair300_epsilon005_floor001_std_kim_20261026'
QUEUE=ROOT/'result/queue_pair_kim_20261026'


def main():
    QUEUE.mkdir(exist_ok=False)
    try:
        if OUT.exists():
            raise RuntimeError('Output already exists')
        if not os.environ.get('OPENROUTER_API_KEY'):
            raise RuntimeError('Test-time LLM API key missing')
        audit()
        protocol=read(PREVIOUS/'protocol.json')
        files=dict(protocol['source_hashes'])
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in files.items()), 'Pure source changed'
        for p in (ROOT/'experiments').rglob('*.py'):
            files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        command=[sys.executable,'-u','-m','experiments.concurrent_fleet_pair_kim.run','--out',str(OUT)]
        write(QUEUE/'protocol.json',dict(previous=str(PREVIOUS),out=str(OUT),command=command,source_hashes=files))
        while True:
            if (QUEUE/'STOP').exists():
                write(QUEUE/'status.json',dict(stage='cancelled_before_launch'))
                return
            state=read(PREVIOUS/'status.json')
            if (PREVIOUS/'failure.json').exists() or state['stage'] in ['failed','paused']:
                raise RuntimeError('Predecessor stopped; do not start Kim')
            if state['stage']=='complete':
                complete=read(PREVIOUS/'late10/complete.json')
                assert complete['epoch']==1000 and complete['test_count']==80
                break
            write(QUEUE/'status.json',dict(stage='waiting',previous_epoch=state.get('epoch'),previous_stage=state['stage']))
            time.sleep(30)
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in files.items()), 'Code changed after scheduling'
        with (QUEUE/'training.stdout.log').open('w') as stdout, (QUEUE/'training.stderr.log').open('w') as stderr:
            child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr)
            write(QUEUE/'status.json',dict(stage='running',pid=child.pid,out=str(OUT)))
            code=child.wait()
        assert code==0, f'Kim failed: exit {code}'
        assert read(OUT/'training/status.json')['stage']=='complete'
        write(QUEUE/'status.json',dict(stage='complete',out=str(OUT)))
    except Exception as error:
        write(QUEUE/'status.json',dict(stage='failed',error=repr(error)))
        raise


if __name__=='__main__':
    main()
