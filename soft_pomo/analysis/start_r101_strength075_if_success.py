"""Start Pure RL only after the exact 600s history run successfully evaluates 80 cases."""
import json
import argparse
from pathlib import Path
import shutil
import sys
import time

from eval_fixed_v2_lkh import ROOT, write, sha
from enrich_rc101_observations import enrich


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=Path, default=ROOT/'analysis/r101_history_strength075_600s_20260925_100204')
    parser.add_argument('--raw', type=Path, default=ROOT/'data/r101_event_strength075_20260925_093959')
    parser.add_argument('--data', type=Path, default=ROOT/'data/r101_event_strength075_discrete_600s')
    parser.add_argument('--run', type=Path, default=ROOT/'result/r101_strength075_pure1000_600s')
    args = parser.parse_args()
    history, raw, data, run = [p.resolve() for p in (args.history,args.raw,args.data,args.run)]
    run.mkdir(exist_ok=True)
    status = run/'pipeline.json'
    if status.exists():
        raise RuntimeError('Pipeline already exists; do not launch duplicate training')
    write(status, dict(stage='waiting_for_history', prerequisite=str(history)))
    deadline = time.monotonic()+1800
    while not (history/'summary.json').exists():
        if time.monotonic() > deadline:
            write(status, dict(stage='not_started', reason='history result timeout'))
            return
        print('WAITING for 600-second History result', flush=True)
        time.sleep(15)
    summary = json.loads((history/'summary.json').read_text())
    if summary.get('status') != 'evaluated' or summary.get('test_evaluated') != 80:
        write(status, dict(stage='not_started', reason='history unsuccessful', summary=summary))
        print('History failed: NO RL TRAINING', flush=True)
        return
    assert summary['groups']['ALL']['means']['served'] == 100
    if (history/'retry_protocol.json').exists():
        retry = json.loads((history/'retry_protocol.json').read_text())
        assert retry['matrix_unchanged'] and retry['split_unchanged'] and retry['time_limit']==600
    solver = json.loads((history/'solver.json').read_text())
    assert solver['time_limit']==600
    split = json.loads((history/'split.json').read_text())
    assert Path(split['data']).resolve() == raw
    assert split['generator_manifest_hash']==sha(raw/'manifest.json')
    assert len(split['train'])==20 and len(split['test'])==80
    results = json.loads((history/'results.json').read_text())
    assert len(results)==80 and all(r['served']==100 for r in results)
    assert {r['name'] for r in results}=={r['name'] for r in split['test']}
    manifest = json.loads((raw/'manifest.json').read_text())
    original = Path(manifest['source_pilot'])
    assert sha(raw/'R101_BASE.txt') == sha(original/'R101_BASE.txt')
    # The event-free reference remains valid because only event multipliers changed.
    shutil.copy2(original/'base_reference.json', raw/'base_reference.json')
    write(status, dict(stage='preparing_observations'))
    enrich(raw,history/'split.json',data)
    write(status, dict(stage='training', data=str(data),epochs=1000,batch=64,
                       history=str(history),base_in_training=False))
    sys.argv = ['train_rc101_event_pilot.py','--run-dir',str(run),
                '--data-dir',str(data),'--benchmark','r1']
    from train_rc101_event_pilot import main as train
    train()
    write(status, dict(stage='complete',epochs=1000,data=str(data)))


if __name__=='__main__':
    main()
