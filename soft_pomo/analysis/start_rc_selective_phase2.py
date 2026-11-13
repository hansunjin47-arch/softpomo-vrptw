"""Validate isolated sparse caches and resume the existing Pure500 checkpoint."""
import copy
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import train_soft_cluster as train
import train_vrptw
import SelectiveAdvanceLLMModule as guidance
from vrptw_env import load_solomon


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = [ROOT/'result_soft/llm_cache_rc101_selective_advance_cluster_v2_sonnet46',
               ROOT/'result_soft/llm_cache_rc1_selective_advance_cluster_v2_sonnet46_training']
    checkpoint = ROOT/'result/config_E_rc1_final_weightedsum_b32_aug8/rc102-rc108/checkpoint-500.pt'
    reference = json.loads((ROOT/'analysis/rc_pure500_reference_20260920_124625/manifest.json').read_text())
    assert sha(checkpoint) == reference['checkpoint_sha256']
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = ROOT/f'result_soft_rc1_selective_cluster_v2_b32_{stamp}'
    out.mkdir(exist_ok=False)
    cache = out/'runtime_cache'
    cache.mkdir()
    provenance = {}
    for source in sources:
        manifest = json.loads((source/'manifest.json').read_text())
        assert len(manifest['results']) == (5 if source == sources[0] else 35)
        for stem, item in manifest['results'].items():
            assert item['success']
            src = ROOT/item['path']
            data = guidance.strict_json(src.read_text())
            guidance.load_advance_cache(src, instance=data['instance'], customer_ids=range(1,101))
            ref = json.loads((ROOT/'analysis/rc_pure500_reference_20260920_124625'/
                             ((data['instance'].split('_')[0] if '_ACC' in data['instance'] else data['instance'])+'.json')).read_text())
            assert data['clusters'] == ref['clusters']
            target = cache/f'{stem}_advance.json'
            assert not target.exists()
            shutil.copyfile(src, target)
            provenance[target.name] = dict(source=str(src), sha256=sha(src), selected=len(data['advance']))
    assert len(provenance) == 40
    (out/'cache_provenance.json').write_text(json.dumps(provenance, indent=2))
    tp = copy.deepcopy(train.trainer_params)
    tp.update(train_instances=['rc102'], test_instances=['rc101'], result_dir=str(out/'smoke'),
              llm_cache_dir=str(cache), cluster_mode='rl', guidance_mode='selective_advance',
              start_mode='random', pomo_size=100, model_load=dict(enable=False),
              test_only=True, experience_refresh_epoch=-1, benchmark='rc1')
    lp = dict(train.llm_params, enabled=True, bias_strength=5., force_refresh=False)
    train._set_seed(42)
    trainer = train.SoftClusterTrainer(dict(train.env_params, **train_vrptw._REWARD_CONFIGS_RC1['E']),
        copy.deepcopy(train.model_params), copy.deepcopy(train.optimizer_params), tp, lp)
    ckpt = torch.load(checkpoint, map_location=trainer.device, weights_only=False)
    assert ckpt['epoch'] == 500
    trainer.model.load_state_dict(ckpt['model_state_dict'], strict=True)
    trainer.model.eval()
    for n in range(101,109):
        for suffix in ['', '_RAIN2', '_ACC2']:
            inst = load_solomon(str(ROOT.parent/'data/Solomon'/f'rc{n}{suffix.lower()}.txt'))
            trainer._ensure_soft_cluster_cache(inst)
            assert len(trainer._acc_stage_events[inst['name']]) == (3 if suffix == '_ACC2' else 0)
    # Execute full routes through the real decoder for all three scenario types.
    smoke = {}
    for suffix in ['', '_RAIN2', '_ACC2']:
        inst = load_solomon(str(ROOT.parent/'data/Solomon'/f'rc101{suffix.lower()}.txt'))
        with torch.no_grad():
            routes, reward = trainer._best_solution(inst)
        nodes = [int(n) for r in routes for n in r if n]
        assert sorted(nodes) == list(range(1,101))
        smoke[inst['name']] = dict(reward=reward, served=len(nodes))
    (out/'smoke.json').write_text(json.dumps(smoke, indent=2))
    del trainer, ckpt
    torch.cuda.empty_cache()
    command = [str(ROOT.parent/'.venv/Scripts/python.exe'), '-u', str(ROOT/'analysis/train_feedback_cached_strict.py'),
        '--benchmark','rc1','--cluster-mode','rl','--guidance-mode','selective_advance',
        '--resume',str(checkpoint),'--epochs','1000','--batch-size','32','--pomo','100',
        '--start-mode','random','--seed','42','--config','E','--acc-ratio','0.333333',
        '--experience-refresh-epoch','-1','--bias-strength','5',
        '--llm-cache-dir',str(cache),'--result-dir',str(out/'training')]
    with (out/'out.log').open('x') as stdout, (out/'err.log').open('x') as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
    status = dict(pid=process.pid, command=command, checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint), epochs='501-1000', batch_size=32,
        seed=42, bias_strength=5, guidance_mode='selective_advance',
        llm_calls=False, cache_count=40, limitations=[
            'ACC own-leg estimates do not propagate delays or use episode visited sets.',
            'Empty wave replaces prior guidance with zero; intentional no-intervention, not fallback.',
            'Checkpoint resume does not restore RNG state.'])
    (out/'status.json').write_text(json.dumps(status, indent=2))
    print('STARTED', process.pid, out, flush=True)


if __name__ == '__main__':
    main()
