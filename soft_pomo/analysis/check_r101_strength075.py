"""Paired event-strength pilot, without changing timing or split."""
import copy
import argparse
import datetime
import json
import shutil
import numpy as np
from eval_fixed_v2_lkh import ROOT, write, sha, load_solomon
from generate_rc101_event_pilot import ge
from eval_rc101_pilot_history_split import main as history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=.75)
    parser.add_argument('--time-limit', type=int, default=600)
    parser.add_argument('--benchmark', choices=['r101', 'c101'], default='r101')
    args = parser.parse_args()
    alpha = args.alpha
    assert np.isfinite(alpha) and alpha > 0
    benchmark = args.benchmark
    tag = f'{alpha:.2f}'.replace('.', '')
    source_stamp = '20260923_202358' if benchmark == 'r101' else '20260923_203633'
    source = ROOT/'data'/f'{benchmark}_event_pilot_{source_stamp}'
    old_split_path = ROOT/'analysis'/f'{benchmark}_pilot_history_20_80_{source_stamp}'/'split.json'
    original = json.loads((source/'manifest.json').read_text())
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    data = ROOT/'data'/(benchmark+'_event_strength'+tag+'_'+stamp)
    out = ROOT/'analysis'/(benchmark+'_history_strength'+tag+'_'+stamp)
    data.mkdir(exist_ok=False)
    raw = ge.load_raw(original['source'])
    manifest = copy.deepcopy(original)
    manifest.update(source_pilot=str(source), source_manifest_sha256=sha(source/'manifest.json'),
        strength_alpha=alpha, multiplier_transform=f'1 + {alpha} * (original_multiplier - 1)',
        multiplier={kind:[1+alpha*(v-1) for v in bounds]
                    for kind,bounds in original['multiplier'].items()},
        code_sha256=sha(__import__('pathlib').Path(__file__)),
        observations='UNSET placeholders; solver sensitivity pilot only, not ready for RL/LLM',cases=[])
    shutil.copy2(source/(benchmark.upper()+'_BASE.txt'),data/(benchmark.upper()+'_BASE.txt'))
    for case in original['cases']:
        old_path = source/(case['name']+'.txt')
        assert sha(old_path) == case['file_sha256']
        updated = copy.deepcopy(case)
        event = updated['event']
        old_mu = event['multiplier']
        event['multiplier'] = round(1+alpha*(old_mu-1),6)
        path = data/(case['name']+'.txt')
        ge.write_scenario(raw,[event],case['name'],str(path))
        old,new = load_solomon(str(old_path)),load_solomon(str(path))
        for key in old:
            if key not in ['preset_events']:
                np.testing.assert_equal(np.asarray(old[key]),np.asarray(new[key]),err_msg=key)
        for key in old['preset_events'][0]:
            if key != 'multiplier':
                assert old['preset_events'][0][key] == new['preset_events'][0][key],key
        assert abs(new['preset_events'][0]['multiplier']-(1+alpha*(old_mu-1))) <= 1e-6
        updated.update(file_sha256=sha(path),source_sha256=sha(old_path),original_multiplier=old_mu)
        manifest['cases'].append(updated)
    write(data/'manifest.json',manifest)
    print('GENERATED',data,'100 events; only excess delay scaled by',alpha,flush=True)
    print('History only; no RL training or LLM calls',flush=True)
    history(data,out,benchmark,time_limit=args.time_limit,service_precheck=True)
    old_split=json.loads(old_split_path.read_text())
    new_split=json.loads((out/'split.json').read_text())
    for group in ['train','test']:
        assert [r['name'] for r in old_split[group]] == [r['name'] for r in new_split[group]]
    write(out/'pipeline.json',dict(stage='complete',data=str(data),alpha=alpha,
        split_unchanged=True,training_started=False,observations_ready=False))
    print('FINAL', (out/'summary.json').read_text(),flush=True)


if __name__=='__main__':
    main()
