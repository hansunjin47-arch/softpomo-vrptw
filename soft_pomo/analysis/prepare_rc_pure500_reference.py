"""Reuse saved held-out solutions; extract training references from exact Pure500."""
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import train_soft_cluster as train
import train_vrptw
from extract_rl_routes import routes_to_cluster_json
from vrptw_env import load_solomon


def main():
    out = ROOT/'analysis'/('rc_pure500_reference_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    checkpoint = ROOT/'result/config_E_rc1_final_weightedsum_b32_aug8/rc102-rc108/checkpoint-500.pt'
    evaluation = ROOT/'result_soft_kim_1000_b64_20260918_000534/eval_pure500_20260918_105232'
    prior = json.loads((evaluation/'manifest.json').read_text())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == next(j['checkpoint_sha256'] for j in prior['jobs'] if j['family']=='rc1')
    tp = copy.deepcopy(train.trainer_params)
    tp.update(train_instances=['rc102'], test_instances=['rc101'],
              result_dir=str(out/'evaluation'), model_load=dict(enable=False),
              test_only=True, benchmark='rc1', start_mode='random', pomo_size=100,
              experience_refresh_epoch=-1)
    lp = dict(train.llm_params, enabled=False)
    train._set_seed(42)
    trainer = train.SoftClusterTrainer(dict(train.env_params, **train_vrptw._REWARD_CONFIGS_RC1['E']),
        copy.deepcopy(train.model_params), copy.deepcopy(train.optimizer_params), tp, lp)
    ckpt = torch.load(checkpoint, map_location=trainer.device, weights_only=False)
    assert ckpt['epoch'] == 500
    trainer.model.load_state_dict(ckpt['model_state_dict'], strict=True)
    trainer.model.eval()
    manifest = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest, epoch=500,
                    seed=42, pomo=100, aug=8, llm_enabled=False, results={})
    for n in range(101,109):
        for suffix in ['', '_RAIN2']:
            name = f'RC{n}{suffix}'
            inst = load_solomon(str(ROOT.parent/'data/Solomon'/f'{name.lower()}.txt'))
            if n == 101:
                path = evaluation/'rc1/rc102-rc108/plots'/f'{name}_solution.txt'
                text = path.read_text(encoding='utf-8')
                rows = re.findall(r'^\s*V\d+:\s*(.*?)\s*\[', text, re.M)
                routes = [[int(v.strip()) for v in row.split('->')] for row in rows]
                reward = float(re.search(r'reward=\s*(-?[\d.]+)', text).group(1))
                source = str(path)
            else:
                with torch.no_grad():
                    routes, reward = trainer._best_solution_aug(inst)
                source = 'Fresh inference with exact existing Pure500 checkpoint; no training'
            clusters = [[int(v) for v in route if v] for route in routes if any(route)]
            assert sorted(v for route in clusters for v in route) == list(range(1, inst['n_customers']+1))
            reference = routes_to_cluster_json([[0]+r+[0] for r in clusters], inst)
            result = dict(name=name, clusters=clusters, reference=reference, reward=reward, source=source)
            (out/f'{name}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            manifest['results'][name] = dict(source=source, reward=reward, vehicles=len(clusters))
            (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            print(name, 'vehicles',len(clusters),'reward', reward, flush=True)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
