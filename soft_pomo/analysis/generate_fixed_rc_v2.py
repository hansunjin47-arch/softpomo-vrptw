"""Generate isolated RC fixed-pool v2 events; never substitute missing references."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'original_POMO'))
import generate_events as ge
sys.path.insert(0, str(ROOT))
from vrptw_env import load_solomon


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=['RC', 'C', 'R'], default='RC')
    args = parser.parse_args()
    prefix = {'RC': 'RCH', 'C': 'CH', 'R': 'RH'}[args.family]
    source = ROOT.parent / 'data/Solomon'
    out = ROOT / 'data' / (f'fixed_{args.family.lower()}_v2_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    cfg = ge.GenConfig()
    manifest = dict(seed=42, family=args.family, config=vars(cfg), generator=str(Path(ge.__file__)),
                    generator_hash=digest(Path(ge.__file__)), instances=[], failures=[],
                    reference_policy='Fresh LKH BASE routes; fail if unavailable; no fallback.',
                    acceptance='Existing stage-1 realized reference impact; accident wave 2-3 events.',
                    scope='Data only. No training or LLM calls.')
    print('OUTPUT', out, flush=True)
    fields = ['n_customers', 'vehicle_limit', 'vehicle_capacity', 'T', 'depot_xy',
              'node_xy', 'node_demand', 'node_tw_open', 'node_tw_close', 'node_service',
              'depot_demand', 'depot_tw_open', 'depot_tw_close', 'depot_service']
    def generate_one(idx):
        name = f'{prefix}{idx + 1:02d}'
        src = source / (name + '.txt')
        originals = {str(p): digest(p) for p in source.glob(name + '*.txt')}
        try:
            print('START', name, flush=True)
            base = load_solomon(str(src))
            shutil.copy2(src, out / src.name)
            data = ge.load_raw(str(src))
            sec = ge.sector_of(data, cfg.n_sectors_total)
            rng = np.random.default_rng(42000 + idx)
            arrivals, routes = ge._reference_route_info(str(src), cache_dir=str(out / 'reference_routes'))
            if arrivals is None or routes is None:
                raise RuntimeError('Missing LKH reference; no fallback permitted')
            assert sorted(n for r in routes for n in r if n) == list(range(1, 101))
            ev = ge.draw_until_stage(data, 'RAIN', cfg, rng, 1, sec=sec,
                                     ref_arrivals=arrivals, ref_routes=routes)
            if ev is None:
                raise RuntimeError('RAIN stage-1 generation exhausted')
            wave = ge.generate_accident_wave(data, cfg, rng, sec=sec,
                                             ref_arrivals=arrivals, ref_routes=routes)
            if wave is None or not 2 <= len(wave) <= 3:
                raise RuntimeError('ACC wave generation failed; no single-event substitute')
            record = dict(name=name, seed=42000 + idx, source_hash=digest(src),
                          reference_vehicles=len(routes), scenarios={})
            for suffix, events in [('RAIN2', [ev]), ('ACC2', wave)]:
                target = out / f'{name}_{suffix}.txt'
                ge.write_scenario(data, events, f'{name}_{suffix}', str(target))
                parsed = load_solomon(str(target))
                for field in fields:
                    assert np.array_equal(np.asarray(base[field]), np.asarray(parsed[field])), field
                assert len(parsed['preset_events']) == len(events)
                record['scenarios'][suffix] = dict(file=target.name, sha256=digest(target), events=events)
            for p, h in originals.items():
                assert digest(Path(p)) == h, 'Original file changed'
            print('OK', name, 'RAIN=1 ACC=' + str(len(wave)), flush=True)
            return record, None
        except Exception as exc:
            print('FAILED', name, repr(exc), flush=True)
            return None, dict(name=name, error=repr(exc))
    with ThreadPoolExecutor(max_workers=3) as executor:
        for future in as_completed([executor.submit(generate_one, i) for i in range(15)]):
            record, failure = future.result()
            if record is not None:
                manifest['instances'].append(record)
            if failure is not None:
                manifest['failures'].append(failure)
            (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default), encoding='utf-8')
    print('DONE', len(manifest['instances']), 'failures', len(manifest['failures']), flush=True)
    if manifest['failures']:
        raise SystemExit(1)
    import subprocess
    subprocess.run([sys.executable, str(ROOT / 'analysis/validate_fixed_rc_v2.py'), str(out)], check=True)


if __name__ == '__main__':
    main()
