"""Continue KIM500 with confidence beta 5 -> 0 through epoch1000; no API calls."""
import argparse
import hashlib
import shutil
import traceback
from experiments.concurrent_fleet_bias_guided.kim300 import OUT as ORIGINAL, PURE
from experiments.concurrent_fleet_bias_guided.support import ROOT, read
from experiments.concurrent_fleet_pair.io import write

OUT = ROOT / 'result/v9_kim300_anneal500_1000_20261023'
CHECKPOINT = ORIGINAL / 'kim/training/late10/checkpoint-500.pt'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if not args.resume:
        OUT.mkdir(exist_ok=False)
        source=ORIGINAL/'kim/artifacts'
        target=OUT/'kim/artifacts'
        shutil.copytree(source,target)
        for p in source.rglob('*'):
            if p.is_file():
                assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256((target/p.relative_to(source)).read_bytes()).digest()
        write(OUT/'cache_reuse.json',dict(source=str(source),exact_copy=True,paid_calls=0))
    try:
        from experiments.concurrent_fleet_bias_guided import train
        train.OUT=OUT;train.PURE=PURE
        write(OUT/'pipeline_status.json',dict(stage='training_and_evaluation'))
        train.run('kim',resume=args.resume,checkpoint_epochs=list(range(550,1001,50)),anneal_from=CHECKPOINT)
        write(OUT/'pipeline_status.json',dict(stage=read(OUT/'kim/training/status.json')['stage']))
    except Exception:
        write(OUT/'pipeline_status.json',dict(stage='failed',error=traceback.format_exc()));raise


if __name__=='__main__':main()
