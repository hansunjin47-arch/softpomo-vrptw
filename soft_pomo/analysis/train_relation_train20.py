"""Continue original train20 Pure500 with validated relation-augmented caches."""
import argparse
import copy
import math
from pathlib import Path
import shutil
import sys

from audit_event_relations import ROOT, read, sha
from run_rc101_pure500_phase2 import Phase2Trainer
from run_rc101_kim_conversation import decode
from run_fixed_kim_sonnet import train, llm, backend, save
import train_vrptw as pure
import train_vrptw_llm as shared
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    run = ROOT / 'result/rc101_relation_train20_cache_v1'
    protocol = read(run / 'protocol.json')
    assert read(run / 'pipeline.json')['stage'] == 'cache_complete'
    source = Path(protocol['original_reference_run'])
    source_protocol = read(source / 'protocol.json')
    checkpoint = Path(source_protocol['checkpoint'])
    assert sha(checkpoint) == protocol['reference_checkpoint_sha256']
    names = protocol['train_names']
    assert len(names) == 20 and names == source_protocol['train_names']
    for name in names:
        record = run / 'requests' / name
        reference = read(record / 'reference.json')
        _, cache = decode((record / 'response.txt').read_text(encoding='utf-8'), reference['clusters'])
        assert cache == read(run / 'cache' / (name + '_rl_cluster.json'))
    for item in read(run / 'input_hashes.json'):
        assert sha(run / 'requests' / item['name'] / 'request.json') == item['request_sha256']
    settings = copy.deepcopy(read(source / 'settings.json'))
    tp, lp = settings['trainer'], settings['llm']
    tp.update(result_dir=str(run / ('smoke_model' if args.smoke else 'model')),
              llm_cache_dir=str(run / 'cache'), model_load=dict(enable=False),
              require_reference_start=False, test_instances=[], train_instances=names)
    assert tp['epochs'] == 1000 and tp['train_batch_size'] == 64 and lp['bias_strength'] == 5
    if not args.smoke and (run / 'settings.json').exists():
        raise RuntimeError('Training already initialized; do not overwrite')
    sys.path.insert(0, str(ROOT / 'server'))
    from train_cached import block_network
    denied = block_network()
    backend.query_llm = llm.query_llm = denied
    for module in (pure, train, shared):
        module._MLFLOW = False
    torch.set_num_threads(4)
    pure._set_seed(42)
    trainer = Phase2Trainer(settings['environment'], settings['model'], settings['optimizer'], tp, lp)
    state = torch.load(checkpoint, map_location=trainer.device, weights_only=False)
    assert state['epoch'] == 500
    trainer.model.load_state_dict(state['model_state_dict'], strict=True)
    trainer.optimizer.load_state_dict(state['optimizer_state_dict'])
    trainer.start_epoch = 501
    assert len(trainer.instance_pool) == 20 and not trainer.test_pool
    del state
    if args.smoke:
        for kind in ('RAIN', 'ACC'):
            inst = next(i for i in trainer.instance_pool if '_' + kind + '_' in i['name'])
            result = trainer._train_one_batch(inst, 1)
            assert all(math.isfinite(float(x)) for x in result)
            print(f'SMOKE {kind}: {result}', flush=True)
        print('PASS: train20 caches, checkpoint, rain/accident backward, offline guard', flush=True)
        return
    save(run / 'settings.json', settings)
    save(run / 'training_protocol.json', dict(start_epoch=501, end_epoch=1000,
        checkpoint_sha256=sha(checkpoint), cache_protocol_sha256=sha(run / 'protocol.json'),
        cache_sha256={n: sha(run / 'cache' / (n + '_rl_cluster.json')) for n in names},
        seed=42, batch=64, beta=5, first_customer_match=False,
        test_evaluation=False, source_script_sha256=sha(Path(__file__))))
    for path in [Path(__file__), ROOT / 'train_soft_cluster.py', ROOT / 'train_vrptw_llm.py',
                 ROOT / 'train_vrptw.py', ROOT / 'vrptw_env.py', ROOT / 'analysis/run_rc101_pure500_phase2.py']:
        shutil.copy2(path, run / 'provenance' / path.name)
    save(run / 'training_status.json', dict(stage='training', start_epoch=501, end_epoch=1000))
    print('START train20 relation-guided Phase2: 501..1000, batch64, beta5, seed42', flush=True)
    try:
        trainer.run()
    except Exception as exc:
        save(run / 'training_status.json', dict(stage='failed', error=str(exc)))
        raise
    save(run / 'training_status.json', dict(stage='complete', epoch=1000))


if __name__ == '__main__':
    main()
