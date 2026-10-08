"""Offline RC101 training entry point. No cache generation or evaluation."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temp, path)


def validate_cache(path, n=100):
    data = read(path)
    groups = [key for key in data if not key.startswith('_')]
    if set(groups) != {str(k) for k in range(len(groups))} or not groups:
        raise ValueError(f'Non-contiguous group IDs: {path}')
    nodes = []
    for key in groups:
        if not isinstance(data[key], dict) or not data[key]:
            raise ValueError(f'Empty/invalid group: {path}')
        nodes.extend(int(node) for node in data[key])
    if sorted(nodes) != list(range(1, n + 1)):
        raise ValueError(f'Customer coverage/duplicates: {path}')
    if set(data.get('_depot', {})) != set(groups):
        raise ValueError(f'Missing depot scores: {path}')
    scores = [v for k in groups for v in data[k].values()] + list(data['_depot'].values())
    if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in scores):
        raise ValueError(f'Invalid confidence score: {path}')
    return data


def verify_bundle(root):
    manifest = read(root / 'bundle_manifest.json')
    for name, digest in manifest['sha256'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != digest:
            raise ValueError(f'Bundle integrity failure: {name}')
    return sha(root / 'bundle_manifest.json')


def block_network():
    def denied(*args, **kwargs):
        raise RuntimeError('Server runner is offline: network/API calls are disabled')
    socket.create_connection = denied
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    for key in list(os.environ):
        if key.endswith('_API_KEY') or key in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY'):
            os.environ.pop(key, None)
    return denied


def build_trainer(config, root, out, cpu=False):
    denied = block_network()
    import torch
    import train_vrptw as pure
    import VRPTWLLMModule as backend
    backend.query_llm = denied
    import SoftClusterLLMModule as llm
    llm.query_llm = denied
    import train_soft_cluster as guided
    import train_vrptw_llm as shared
    for module in (pure, guided, shared):
        module._MLFLOW = False
    guided.build_bks_fewshot_block = lambda *a, **k: ''

    class CachedTrainer(guided.SoftClusterTrainer):
        def _ensure_soft_cluster_cache(self, inst, force_refresh=False):
            if force_refresh:
                raise RuntimeError('Cache refresh is not allowed on the server')
            if config['method'] == 'kim':
                return super()._ensure_soft_cluster_cache(inst, False)
            name = inst['name']
            if name in self._cluster_conf_cache:
                return
            cached = validate_cache(Path(self._llm_cache_dir) / (name + '_rl_cluster.json'))
            clusters, scores = guided._parse_cluster_cache(cached)
            tensor = self._build_cluster_conf_tensor(clusters, scores, len(clusters),
                inst['n_customers'], guided._parse_cluster_depot(cached))
            self._cluster_assign_cache[name] = clusters
            accidents = [e for e in inst['preset_events'] if e['type'] == 'ACCIDENT']
            if accidents:
                if len(accidents) != 1:
                    raise ValueError('This runner supports one event per scenario')
                self._acc_stage_events[name] = [dict(
                    t_start=float(accidents[0]['trigger_time']) / float(inst['T']), tensor=tensor)]
                self._cluster_conf_cache[name] = torch.zeros_like(tensor)
            else:
                self._cluster_conf_cache[name] = tensor
            self._original_cluster_conf[name] = self._cluster_conf_cache[name].clone()

    tp = copy.deepcopy(config['trainer'])
    tp.update(data_dir=str(root / 'data'), result_dir=str(out / 'model'),
              use_cuda=not cpu, cuda_device_num=0, test_instances=[],
              model_load={'enable': False})
    if config['method'] != 'pure':
        tp['llm_cache_dir'] = str(root / config['cache_dir'])
    pure._set_seed(tp['seed'])
    torch.set_num_threads(4)
    if not cpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; run inside a Slurm GPU allocation with --nv')
    args = (config['environment'], config['model'], config['optimizer'], tp)
    trainer = pure.VRPTWTrainer(*args) if config['method'] == 'pure' else CachedTrainer(*args, config['llm'])
    return trainer


def save_checkpoint(trainer, epoch, path, contract):
    import numpy as np
    import torch
    state = dict(epoch=epoch, model_state_dict=trainer.model.state_dict(),
        optimizer_state_dict=trainer.optimizer.state_dict(), reward_log=trainer.reward_log,
        loss_log=trainer.loss_log, server_contract=contract,
        python_rng=random.getstate(), numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if trainer.device.type == 'cuda' else [],
        advantage_state=vars(trainer._adv_running_std))
    if hasattr(trainer, '_scaler'):
        state['scaler'] = trainer._scaler.state_dict()
    if hasattr(trainer, '_gen_rng'):
        state['generator_rng'] = trainer._gen_rng.bit_generator.state
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    os.replace(temp, path)


def restore(trainer, path, contract=None):
    import numpy as np
    import torch
    # Only load our own trusted checkpoints, never untrusted downloaded pickle files.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if contract is not None and state.get('server_contract') != contract:
        raise ValueError('Resume config, data, caches, or source differ from the saved run')
    trainer.model.load_state_dict(state['model_state_dict'], strict=True)
    trainer.optimizer.load_state_dict(state['optimizer_state_dict'])
    trainer.start_epoch = int(state['epoch']) + 1
    trainer.reward_log = state.get('reward_log', [])
    trainer.loss_log = state.get('loss_log', [])
    if contract is not None:
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        if trainer.device.type == 'cuda' and state['cuda_rng']:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        vars(trainer._adv_running_std).update(state['advantage_state'])
        if 'scaler' in state and hasattr(trainer, '_scaler'):
            trainer._scaler.load_state_dict(state['scaler'])
        if 'generator_rng' in state and hasattr(trainer, '_gen_rng'):
            trainer._gen_rng.bit_generator.state = state['generator_rng']
    return int(state['epoch'])


def validate_resume(previous, current):
    before, after = copy.deepcopy(previous), copy.deepcopy(current)
    old_target = before['config']['trainer'].pop('epochs')
    new_target = after['config']['trainer'].pop('epochs')
    if before != after or new_target < old_target:
        raise ValueError('Resume permits only an unchanged or increased epoch target')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['pure', 'kim', 'phase2'], required=True)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke-batch-size', type=int, default=1)
    ap.add_argument('--cpu', action='store_true', help='Local smoke/initialization checks only')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, help='Absolute final epoch, not additional epochs')
    ap.add_argument('--batch-size', type=int, help='Override training batch size; default is historical 64')
    ap.add_argument('--save-every', type=int, default=10)
    ap.add_argument('--max-hours', type=float, default=11.5)
    args = ap.parse_args()
    if args.cpu and not (args.smoke or args.check_only):
        ap.error('--cpu is only for checks')
    if args.save_every < 1 or args.max_hours <= 0 or args.smoke_batch_size < 1:
        ap.error('Save interval and time budget must be positive')
    if args.batch_size is not None and args.batch_size < 1:
        ap.error('Batch size must be positive')
    digest = verify_bundle(ROOT)
    config = read(ROOT / 'configs' / (args.method + '.json'))
    config['trainer']['seed'] = args.seed
    if args.epochs is not None:
        config['trainer']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['trainer']['train_batch_size'] = args.batch_size
    if args.smoke:
        config['trainer']['train_batch_size'] = args.smoke_batch_size
    out = (args.out or ROOT / 'runs' / (args.method + ('_smoke' if args.smoke else ''))).resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f'Output is not empty; use a new directory or --resume: {out}')
    if args.resume and (args.smoke or args.check_only):
        ap.error('Use a fresh directory for checks')
    contract = dict(bundle_sha256=digest, config=config)
    checkpoint = out / 'checkpoint-last.pt'
    if args.resume and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    previous = contract
    if args.resume:
        previous = read(out / 'settings.json')
        validate_resume(previous, contract)
    trainer = build_trainer(config, ROOT, out, args.cpu)
    import torch
    epoch = 0
    if args.resume:
        epoch = restore(trainer, checkpoint, previous)
    elif args.method == 'phase2':
        epoch = restore(trainer, ROOT / config['initial_checkpoint'])
        if epoch != 500:
            raise ValueError('Phase2 must branch from epoch 500')
        # Historical Phase2 resets the seed and transfers only model + optimizer.
        trainer.reward_log, trainer.loss_log = [], []
    write(out / 'settings.json', contract)
    print(f'Python={sys.version.split()[0]} torch={torch.__version__} CUDA={torch.version.cuda}', flush=True)
    print(f'Device={trainer.device} GPU={torch.cuda.get_device_name(0) if trainer.device.type == "cuda" else "none"}', flush=True)
    if args.check_only:
        print('PASS: imports, data, caches, model initialization and checkpoint loading', flush=True)
        return
    if args.smoke:
        # Both disclosure paths must complete forward, backward and optimizer updates.
        for kind in ('RAIN', 'ACC'):
            inst = next(i for i in trainer.instance_pool if '_' + kind + '_' in i['name'])
            result = trainer._train_one_batch(inst, args.smoke_batch_size)
            if not all(math.isfinite(float(v)) for v in result):
                raise RuntimeError(f'Non-finite smoke result: {result}')
            print(f'SMOKE {kind}: reward={result[0]} loss={result[1]}', flush=True)
        save_checkpoint(trainer, epoch, out / 'smoke-checkpoint.pt', contract)
        restore(trainer, out / 'smoke-checkpoint.pt', contract)
        print('PASS: rain/accident backward steps and checkpoint roundtrip (not a production checkpoint)', flush=True)
        return
    stop = [False]
    def request_stop(*unused):
        stop[0] = True
    for sig in (signal.SIGTERM, getattr(signal, 'SIGUSR1', signal.SIGTERM)):
        signal.signal(sig, request_stop)
    started = time.monotonic()
    last_duration = 0.0
    final = config['trainer']['epochs']
    if final < epoch:
        raise ValueError('Final epoch is below the checkpoint epoch')
    while epoch < final:
        if stop[0] or time.monotonic() - started + last_duration >= args.max_hours * 3600:
            break
        begin = time.monotonic()
        trainer._current_epoch = epoch + 1
        reward, loss = trainer._train_one_epoch(epoch + 1)
        if not math.isfinite(reward) or not math.isfinite(loss):
            raise RuntimeError('Non-finite epoch result; previous checkpoint retained')
        epoch += 1
        trainer.reward_log.append(reward)
        trainer.loss_log.append(loss)
        last_duration = time.monotonic() - begin
        print(f'Epoch {epoch}/{final} reward={reward:.6f} loss={loss:.6f} seconds={last_duration:.1f}', flush=True)
        if epoch % args.save_every == 0 or epoch == final:
            save_checkpoint(trainer, epoch, checkpoint, contract)
        if epoch == 500 or epoch % 1000 == 0:
            save_checkpoint(trainer, epoch, out / f'checkpoint-{epoch}.pt', contract)
        write(out / 'status.json', dict(epoch=epoch, target=final, status='training'))
    save_checkpoint(trainer, epoch, checkpoint, contract)
    write(out / 'status.json', dict(epoch=epoch, target=final,
        status='complete' if epoch == final else 'needs_resume'))
    print(f'Saved epoch {epoch}; complete={epoch == final}', flush=True)


if __name__ == '__main__':
    main()
