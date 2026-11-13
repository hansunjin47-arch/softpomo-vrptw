"""Pure RL on the frozen RC101 event-only 20/80 split."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import train_vrptw as training


class PilotTrainer(training.VRPTWTrainer):
    def _train_one_batch(self, inst, batch_size):
        result = super()._train_one_batch(inst, batch_size)
        if not getattr(self, '_first_batch_logged', False):
            print(f'[First batch OK] instance={inst["name"]} batch={batch_size} '
                  f'reward={result[0]:.6f} loss={result[1]:.6f}', flush=True)
            self._first_batch_logged = True
        return result

    def _train_one_epoch(self, epoch=0):
        start = time.time()
        result = super()._train_one_epoch(epoch)
        print(f'[Epoch {epoch}] reward={result[0]:.6f} loss={result[1]:.6f} '
              f'seconds={time.time()-start:.1f}', flush=True)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/rc101_event_pilot_discrete_v1')
    parser.add_argument('--benchmark', choices=['rc1', 'r1'], default='rc1')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    data = args.data_dir.resolve()
    split = json.loads((data / 'split.json').read_text())
    validation = json.loads((data / 'validation.json').read_text())
    assert validation['event_dynamics_unchanged'] and validation['fixed_route_replay_identical']
    assert len(split['train']) == 20 and len(split['test']) == 80
    assert not ({e['name'] for e in split['train']} & {e['name'] for e in split['test']})
    for entry in split['train'] + split['test']:
        path = data / (entry['name'] + '.txt')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
    assert torch.cuda.is_available(), 'CUDA required for this run'
    torch.set_num_threads(4)
    training._set_seed(42)
    configs = training._REWARD_CONFIGS_RC1 if args.benchmark == 'rc1' else training._REWARD_CONFIGS_R1
    ep = dict(training.env_params, **configs['E'])
    ep['pomo_size'] = 100
    tp = copy.deepcopy(training.trainer_params)
    tp.update(data_dir=str(data), epochs=1000, train_batch_size=64,
        train_instances=[e['name'] for e in split['train']],
        test_instances=[e['name'] for e in split['test']], result_dir=str(run / 'model'),
        use_augmentation=True, train_aug8=False, seed=42, curriculum_epoch=0)
    tp['model_load']['enable'] = False
    run.mkdir(parents=True, exist_ok=True)
    provenance = run / 'provenance'
    provenance.mkdir(exist_ok=False)
    for name in ['split.json', 'manifest.json', 'observation_rules.audit.json', 'validation.json']:
        shutil.copy2(data / name, provenance / name)
    for name in ['train_vrptw.py', 'vrptw_env.py', 'event_observation.py', 'analysis/train_rc101_event_pilot.py']:
        shutil.copy2(ROOT / name, provenance / Path(name).name)
    (provenance / 'config.json').write_text(json.dumps(dict(
        trainer=tp, model=training.model_params, environment=ep,
        optimizer=training.optimizer_params, config='E_'+args.benchmark.upper(),
        initialized_from='random; no checkpoint', base_in_training=False), indent=2))
    trainer = PilotTrainer(ep, training.model_params, training.optimizer_params, tp)
    print(f'[START] CUDA={torch.cuda.get_device_name(0)} result={trainer.result_dir}', flush=True)
    print('[Protocol] train=20 test=80; b64 pomo100 train_aug8=OFF eval_aug8=ON; 2500 episodes/epoch; no LLM', flush=True)
    trainer.run()


if __name__ == '__main__':
    main()
