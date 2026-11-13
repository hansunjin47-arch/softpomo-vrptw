"""Run local RC return-mask experiments sequentially for selected seeds."""
import argparse
import hashlib
from itertools import product
import os
from pathlib import Path
import subprocess
import sys
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_bias_guided.support import ROOT, read

BUNDLE=ROOT/'server_export/pair_returnmask_cr_v100_200_l5_s42_46_20261031'
QUEUE=ROOT/'result/queue_rc_returnmask_l5_20261031'


def main():
    global QUEUE
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,nargs='+',default=[42])
    parser.add_argument('--tag',default='20261031')
    args=parser.parse_args()
    assert len(set(args.seeds))==len(args.seeds), 'Duplicate seeds'
    assert args.tag.isalnum(), 'Tag must be alphanumeric'
    QUEUE=ROOT/f'result/queue_rc_returnmask_l5_{args.tag}'
    QUEUE.mkdir(exist_ok=False)
    try:
        files=list((ROOT/'experiments').rglob('*.py'))+list(ROOT.glob('*.py'))
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        commands=[]
        for seed,weight in product(args.seeds,(100,200)):
            out=ROOT/f'result/rc_pair_returnmask_v{weight}_l5_s{seed}_{args.tag}'
            assert not out.exists(), 'Duplicate output'
            command=[sys.executable,'-u','-m','experiments.concurrent_fleet_pair.train',
                '--out',str(out),'--weights','5','--vehicle-cost',str(weight),
                '--reward-denominator','1696.95','--seed',str(seed),'--return-mask',
                '--source',str(BUNDLE/'datasets/rc101'),'--config',str(BUNDLE/'configs/rc101.json'),
                '--new-customer-epsilon','.05','--epsilon-final','.01','--epsilon-end-epoch','1000']
            commands.append((out,command))
        write(QUEUE/'protocol.json',dict(commands=[c for _,c in commands],source_hashes=hashes,
              seeds=args.seeds, setting='RC101, lambda5, fresh1000, return mask train/test, stdON, greedy test80'))
        for out,command in commands:
            if (QUEUE/'STOP').exists():
                write(QUEUE/'status.json',dict(stage='cancelled_before_launch')); return
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()), 'Source changed'
            with (QUEUE/(out.name+'.stdout.log')).open('w') as stdout, (QUEUE/(out.name+'.stderr.log')).open('w') as stderr:
                child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr)
                write(QUEUE/'status.json',dict(stage='training_and_test',pid=child.pid,out=str(out)))
                if child.wait()!=0: raise RuntimeError('Run failed: '+str(out))
            done=read(out/'late5/complete.json')
            assert done['epoch']==1000 and done['test_count']==80
            assert len(read(out/'late5/results.json'))==80
        write(QUEUE/'status.json',dict(stage='complete'))
    except Exception as error:
        write(QUEUE/'status.json',dict(stage='failed',error=repr(error)))
        raise


if __name__=='__main__': main()
