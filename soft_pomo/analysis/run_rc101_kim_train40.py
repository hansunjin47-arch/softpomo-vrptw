"""Reuse validated train20 Kim caches, call only new events, train from scratch."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_rc101_kim_conversation as cache
import train_rc101_kim_pilot as training

DATA = ROOT/'data/rc101_train40_test80_20260928'
OLD = ROOT/'analysis/rc101_kim_independent_train20'
CACHE = ROOT/'analysis/rc101_kim_independent_train40_20260928'
OUT = ROOT/'result/rc101_kim_train40_fresh1000_20260928'


def main():
    global DATA, OLD, CACHE, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--reuse-cache', type=Path, default=OLD)
    parser.add_argument('--cache-out', type=Path, default=CACHE)
    parser.add_argument('--out', type=Path, default=OUT)
    opts = parser.parse_args()
    DATA, OLD, CACHE, OUT = (p.resolve() for p in
        [opts.data_dir, opts.reuse_cache, opts.cache_out, opts.out])
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'settings.json').exists(), 'Training already started'
    cache.save(OUT/'pipeline.json', dict(stage='cache_preparation'))
    args = ['run_rc101_kim_conversation.py','--out',str(CACHE),'--data-dir',str(DATA),'--independent']
    sys.argv = args+['--prepare-only']
    cache.main()
    old = json.loads((OLD/'manifest.json').read_text())
    new = json.loads((CACHE/'manifest.json').read_text())
    reused = len(old['settings']['order'])
    total = len(new['settings']['order'])
    assert old['status']=='complete' and (reused,total) in [(20,40),(40,80)]
    for key in ['model','mode','base_sha256','clusters','common_prompt','response_format']:
        assert old['settings'][key]==new['settings'][key], key
    assert new['settings']['order'][:reused]==old['settings']['order']
    for index,name in enumerate(old['settings']['order']):
        assert old['settings']['inputs'][name]==new['settings']['inputs'][name]
        turn=f'{index+1:02}_{name}'
        target=CACHE/'turns'/turn
        if not target.exists():
            shutil.copytree(OLD/'turns'/turn,target)
    cache.save(OUT/'pipeline.json',dict(stage='cache_generation', reused=reused,new_calls=total-reused,total=total))
    print(f'Validated matching prompt, geometry and inputs; reuse{reused}, call only new{total-reused}',flush=True)
    sys.argv=args
    cache.main()
    sys.argv=['train_rc101_kim_pilot.py','--cache-run',str(CACHE),'--out',str(OUT),
              '--data-dir',str(DATA),'--base-guidance','off']
    training.main()


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        cache.save(OUT/'error.json',dict(error=repr(exc)))
        raise
