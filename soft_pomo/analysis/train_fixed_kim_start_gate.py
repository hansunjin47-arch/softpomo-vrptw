"""RC ablation: gate an entire rollout by its initial random customer."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import train_soft_cluster as train
import SoftClusterLLMModule as llm
import VRPTWLLMModule as backend


def initial_start_mask(history, eligible):
    nonzero = history != 0
    first_position = nonzero.long().argmax(dim=-1)
    first_node = history.gather(-1, first_position.unsqueeze(-1)).squeeze(-1)
    return nonzero.any(dim=-1) & eligible[first_node]


class StartGatedTrainer(train.SoftClusterTrainer):
    def __init__(self, *args, **kwargs):
        self.start_eligible = {}
        self.gate_counts = dict(decisions=0, enabled=0)
        super().__init__(*args, **kwargs)

    def _acc_conf_now(self, name, *args, **kwargs):
        bias = super()._acc_conf_now(name, *args, **kwargs)
        history = self.env.selected_node_list
        if name not in self.start_eligible:
            # Initial ACC starts use BASE scores, never a future accident wave.
            base_name = name.removesuffix('_ACC2')
            path = Path(self._llm_cache_dir)/f'{base_name}_cluster.json'
            cache = json.loads(path.read_text(encoding='utf-8'))
            eligible = torch.zeros(self.env.problem_size+1, dtype=torch.bool, device=history.device)
            for key, scores in cache.items():
                if key.startswith('_'):
                    continue
                maximum = max(scores.values())
                eligible[[int(n) for n,v in scores.items() if v == maximum]] = True
            self.start_eligible[name] = eligible
        mask = initial_start_mask(history, self.start_eligible[name])
        self.gate_counts['decisions'] += mask.numel()
        self.gate_counts['enabled'] += int(mask.sum().item())
        return bias * mask.unsqueeze(-1)

    def _train_one_epoch(self, epoch):
        result = super()._train_one_epoch(epoch)
        counts = dict(self.gate_counts, epoch=epoch)
        counts['enabled_fraction'] = counts['enabled']/max(1,counts['decisions'])
        path = Path(self.trainer_params['result_dir'])/'start_gate_metrics.jsonl'
        with path.open('a',encoding='utf-8') as f:
            f.write(json.dumps(counts)+'\n')
        self.gate_counts = dict(decisions=0,enabled=0)
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--family', choices=['rc1', 'r1'], default='rc1')
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'settings.json').exists():
        raise RuntimeError('Existing experiment: refusing to restart or overwrite')
    source = ROOT/'result_soft_fixed_kim_sonnet_20260921'/args.family
    settings = json.loads((source/'training_settings.json').read_text())
    tp, lp = settings['trainer'], settings['llm']
    assert tp['model_load']['enable'] is False and tp['start_mode']=='random'
    tp['result_dir'] = str(out/'training')
    settings['gate'] = dict(scope='whole rollout',match='initial customer is highest raw customer score within its cluster',
        ties='all maxima',acc='initial BASE scores; no reclassification at later waves or depot returns',
        original_run=str(source),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'settings.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    def reject(*args,**kwargs):
        raise RuntimeError('Cache-only experiment: LLM calls prohibited')
    llm.query_llm = backend.query_llm = reject
    train._set_seed(42)
    trainer = StartGatedTrainer(dict(train.env_params,**settings['env']),copy.deepcopy(train.model_params),
        copy.deepcopy(train.optimizer_params),tp,lp)
    print(f'TRAINING_START initial-start gate {args.family}, batch64, 1000 epochs, seed42',flush=True)
    trainer.run()
    print('COMPLETE',flush=True)


if __name__=='__main__':
    main()
