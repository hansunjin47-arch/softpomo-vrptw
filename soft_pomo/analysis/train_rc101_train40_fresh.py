"""Train Pure RL from scratch on the frozen 40/80 event split."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import train_vrptw as training
from train_rc101_event_pilot import PilotTrainer
from eval_fixed_v2_lkh import sha, write

DATA = ROOT/'data/rc101_train40_test80_20260928'
OUT = ROOT/'result/rc101_pure_train40_fresh1000_20260928'
SOURCE = ROOT/'result/rc101_pilot_discrete_pure1000_20260923_115506/provenance'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(DATA=DATA, OUT=OUT, history_path=None):
    split = read(DATA/'split.json')
    original = read(SOURCE/'split.json')
    count = len(split['train'])
    assert count in (40, 80) and split['test'] == original['test']
    assert split['train'][:20] == original['train']
    assert len({r['name'] for r in split['train']+split['test']}) == count+80
    for row in split['train']+split['test']:
        assert sha(DATA/(row['name']+'.txt')) == row['sha256']
    config = read(SOURCE/'config.json')
    tp = dict(config['trainer'], data_dir=str(DATA), epochs=1000,
        train_instances=[r['name'] for r in split['train']],
        test_instances=[r['name'] for r in split['test']], result_dir=str(OUT/'model'),
        model_load=dict(enable=False, path=None, epoch=None),
        logging=dict(model_save_interval=500, log_interval=10))
    assert tp['train_batch_size']==64 and tp['train_episodes']==2500
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = OUT/'provenance'
    provenance.mkdir(exist_ok=False)
    for name in ['split.json','manifest.json','validation.json','observation_rules.audit.json']:
        shutil.copy2(DATA/name, provenance/name)
    for path in [Path(__file__),ROOT/'train_vrptw.py', ROOT/'vrptw_env.py',
                 ROOT/'event_observation.py', ROOT/'analysis/train_rc101_event_pilot.py']:
        shutil.copy2(path,provenance/path.name)
    write(provenance/'config.json',dict(config,trainer=tp,
        initialized_from='random; no checkpoint or optimizer state loaded',
        history=str(history_path or ROOT/'analysis/rc101_history_train40_20260928')))
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    training._set_seed(42)
    trainer = PilotTrainer(config['environment'],config['model'],config['optimizer'],tp)
    assert trainer.start_epoch==1 and len(trainer.instance_pool)==count and len(trainer.test_pool)==80
    write(OUT/'pipeline.json',dict(stage='training',start_epoch=1,end_epoch=1000,
                                  train=count,test=80,initialized_from='random'))
    print(f'START FRESH Pure RL epoch 1..1000; train{count}/test80; b64; NO checkpoint load',flush=True)
    trainer.run()
    write(OUT/'pipeline.json',dict(stage='complete',epoch=1000,initialized_from='random'))


if __name__=='__main__':
    main()
