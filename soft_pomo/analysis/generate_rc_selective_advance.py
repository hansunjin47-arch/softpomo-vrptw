"""Generate isolated RC sparse guidance from existing reference sequences."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import SelectiveAdvanceLLMModule as llm
from extract_rl_routes import routes_to_cluster_json
from train_soft_cluster import SoftClusterTrainer, _get_accidents
from vrptw_env import load_solomon
import train_vrptw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--instances', nargs='+', type=int, default=list(range(101,109)))
    parser.add_argument('--reference-dir', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    source = Path(args.reference_dir).resolve()
    reference_manifest = llm.strict_json((source/'manifest.json').read_text(encoding='utf-8'))
    checkpoint = Path(reference_manifest['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == reference_manifest['checkpoint_sha256']
    assert reference_manifest['epoch'] == 500 and reference_manifest['llm_enabled'] is False
    jobs = []
    hashes = {}
    objective = ('Minimize finite weighted cost, NOT lexicographic priority. '
                 'Serve all customers subject to capacity/vehicle constraints. '
                 'RC1 reward Config E parameters: ' + json.dumps(train_vrptw._REWARD_CONFIGS_RC1['E']) +
                 '. Late-stop count is reported but has zero direct penalty; total lateness is penalized. '
                 'Travel time includes event multipliers.')
    for number in args.instances:
        assert 101 <= number <= 108
        name = f'RC{number}'
        references = {}
        for suffix in ['', '_RAIN2']:
            path = source / f'{name}{suffix}.json'
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            old = llm.strict_json(path.read_text(encoding='utf-8'))
            assert old['name'] == name+suffix
            clusters = old['clusters']
            inst = load_solomon(str(ROOT.parent/'data/Solomon'/f'{name.lower()}{suffix.lower()}.txt'))
            replay = old['reference']
            stats = {int(k): v for k, v in replay['_stats'].items()}
            references[suffix] = (clusters, stats)
            jobs.append((name+suffix, inst, clusters, stats, None))
        acc = load_solomon(str(ROOT.parent/'data/Solomon'/f'{name.lower()}_acc2.txt'))
        clusters, base_stats = references['']
        for wave in sorted({float(f['t_start']) for _, f in _get_accidents(acc)}):
            inst = copy.deepcopy(acc)
            inst['preset_events'] = [e for e in inst['preset_events']
                                     if e['trigger_time']/float(inst['T']) <= wave]
            mult = {}
            for event, features in _get_accidents(inst):
                if features['t_start'] <= wave < features['t_end']:
                    for node in event.affected_nodes:
                        mult[node] = max(mult.get(node, 1.), features['multiplier'])
            stats = SoftClusterTrainer._apply_accident_to_stats(base_stats, mult)
            jobs.append((f'{name}_ACC2_wave_{wave:.6f}', inst, clusters, stats, wave))
    manifest = dict(model='anthropic/claude-sonnet-4.6', source_hashes=hashes,
                    reference_provenance=reference_manifest,
                    acc_policy='BASE reference, visible-event own-leg estimate; no downstream propagation or actual episode state.',
                    test_split='RC101 inference only; never a training example', results={})
    def save():
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save()
    # Validate every input before making any paid request.
    for stem, inst, clusters, stats, wave in jobs:
        prompt = llm.build_advance_prompt(inst, clusters, stats=stats, objective=objective,
                    events_visible_until=wave, stats_is_estimate=wave is not None)
        (out/f'{stem}_preview.txt').write_text(prompt, encoding='utf-8')
    print(f'PREPARED {len(jobs)} requests: {out}', flush=True)
    if args.prepare_only:
        return
    for stem, inst, clusters, stats, wave in jobs:
        success = False
        for attempt in range(1, 4):
            print('CALL', stem, 'attempt', attempt, flush=True)
            folder = out/('evaluation' if stem.startswith('RC101') else 'training')/stem/f'attempt{attempt}'
            try:
                cache = llm.generate_advance(inst, clusters, stats=stats, objective=objective,
                    model=manifest['model'], output_dir=folder,
                    events_visible_until=wave, stats_is_estimate=wave is not None)
                manifest['results'][stem] = dict(success=True, path=str(folder/'cache.json'),
                    selected=len(cache['advance']), attempt=attempt)
                save()
                print('SUCCESS', stem, 'selected', len(cache['advance']), flush=True)
                success = True
                break
            except (ValueError, RuntimeError) as exc:
                manifest['results'][stem] = dict(success=False, attempt=attempt, error_type=type(exc).__name__)
                save()
                print('FAILED', stem, type(exc).__name__, flush=True)
                if isinstance(exc, RuntimeError):
                    raise
                time.sleep(3)
        if not success:
            raise RuntimeError(f'{stem}: validation failed; no fallback')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    print('DONE', len(manifest['results']), flush=True)


if __name__ == '__main__':
    main()
