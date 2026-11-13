"""Validate complete Kim test results before recording receipt."""
import argparse
import json
import math
from pathlib import Path


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--receipt',type=Path,required=True)
    args=ap.parse_args()
    received=set(read(args.receipt)) if args.receipt.exists() else set()
    additions=[]
    for f in args.input.rglob('eval_1000_test80_gpu/results.json'):
        bundle=f.parent.parent
        evaluation=read(f.parent/'status.json')
        training=read(bundle/'runs/kim/status.json')
        rows=read(f)
        names=[r['name'] for r in read(bundle/'split.json')['test']]
        assert evaluation['stage']=='complete' and evaluation['epoch']==1000 and evaluation['completed']==80
        assert training['status']=='complete' and training['epoch']==1000
        assert len(rows)==len(set(names))==80 and [r['name'] for r in rows]==names
        for row in rows:
            assert sorted(n for route in row['routes'] for n in route if n)==list(range(1,101))
            assert row['served']==100
            assert all(math.isfinite(row[k]) for k in ['reward','cost','travel_time','total_late'])
            assert abs(row['reward']+row['cost'])<1e-4
        name=bundle.relative_to(args.input).as_posix()
        additions.append(name)
        print('VERIFIED',name,'mean reward',sum(r['reward'] for r in rows)/80)
    received.update(additions)
    temp=args.receipt.with_suffix('.tmp')
    temp.write_text(json.dumps(sorted(received),indent=2),encoding='utf-8')
    temp.replace(args.receipt)
    print('Verified runs in this download:',len(additions),'Total received:',len(received))


if __name__=='__main__':main()
