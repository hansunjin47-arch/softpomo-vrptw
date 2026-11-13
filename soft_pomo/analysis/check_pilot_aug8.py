"""Check fixed-action simulator invariance over all pilot events and 8 views."""
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vrptw_env import VRPTWEnv, load_solomon, make_batch
from train_vrptw import _make_rain_tokens


def main():
    torch.set_num_threads(2)
    data = ROOT / 'data/rc101_event_pilot_discrete_v1'
    manifest = json.loads((data / 'manifest.json').read_text())
    reference = ROOT / 'data/rc101_event_pilot_20260922_220135/base_reference.json'
    routes = json.loads(reference.read_text())['routes']
    actions = [0]
    for route in routes:
        actions.extend([n for n in route if n != 0] + [0])
    for case in manifest['cases']:
        inst = load_solomon(str(data / (case['name'] + '.txt')))
        batch = make_batch(inst, 8, torch.device('cpu'), augment8=True)
        coords = torch.cat([batch['depot_xy'], batch['node_xy']], 1)
        distances = torch.cdist(coords, coords)
        torch.testing.assert_close(distances, distances[:1].expand_as(distances), atol=1e-3, rtol=1e-5)
        for key in ['node_demand', 'node_tw_open', 'node_tw_close', 'node_service', 'tt']:
            torch.testing.assert_close(batch[key], batch[key][:1].expand_as(batch[key]))
        assert batch['events'] == inst['preset_events']
        rain = [e for e in inst['preset_events'] if e['type'] == 'RAIN']
        tokens = _make_rain_tokens(inst, rain, 8, torch.device('cpu'))
        torch.testing.assert_close(tokens, tokens[:1].expand_as(tokens))
        env = VRPTWEnv(problem_size=100, pomo_size=1)
        env.load_problems(batch)
        env.reset()
        for node in actions:
            state, reward, done = env.step(torch.full((8, 1), node, dtype=torch.long))
            for value in [state.current_time, env.total_late, env.n_late_stops, state.load]:
                torch.testing.assert_close(value, value[:1].expand_as(value), atol=0, rtol=0)
            if reward is not None:
                torch.testing.assert_close(reward, reward[:1].expand_as(reward), atol=0, rtol=0)
        assert done
    print('PASS: 100 scenarios x 8 views; same actions give identical simulator times, lateness and rewards.')


if __name__ == '__main__':
    main()
