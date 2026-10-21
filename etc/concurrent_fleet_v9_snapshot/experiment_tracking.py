"""Provenance and route metrics shared by Pure RL and guided training."""
import hashlib
from pathlib import Path


def sanitize(value):
    if isinstance(value, dict):
        return {str(k): ('[REDACTED]' if any(s in str(k).lower() for s in
                ('api_key', 'apikey', 'authorization', 'password', 'secret', 'token'))
                else sanitize(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, str) and value.startswith(('sk-', 'Bearer ')):
        return '[REDACTED]'
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def dataset_manifest(tp):
    rows = []
    root = Path(tp['data_dir'])
    for split in ('train', 'test'):
        paths = ([Path(tp['train_dataset'])] if split == 'train' and tp.get('train_dataset')
                 else [root / (name + '.txt') for name in tp.get(split + '_instances', [])])
        for path in paths:
            row = dict(split=split, path=str(path.resolve()), exists=path.is_file())
            if path.is_file():
                with path.open('rb') as f:
                    row['sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
            rows.append(row)
    return rows


def log_setup(trainer):
    import mlflow
    configs = {name: getattr(trainer, name, {}) for name in
               ('trainer_params', 'model_params', 'optimizer_params', 'env_params', 'llm_params')}
    mlflow.log_dict(sanitize(configs), 'provenance/config.json')
    mlflow.log_dict({'files': dataset_manifest(trainer.trainer_params)}, 'provenance/dataset.json')
    mlflow.log_text(str(trainer.model), 'provenance/model.txt')
    mlflow.log_dict({k: list(v.shape) for k, v in trainer.model.state_dict().items()},
                    'provenance/model_shapes.json')
    mlflow.set_tags({'trainer_class': type(trainer).__name__,
                     'model_class': type(trainer.model).__name__,
                     'result_dir': str(Path(trainer.result_dir).resolve())})
    mlflow.log_param('model_parameter_count', sum(p.numel() for p in trainer.model.parameters()))


def route_metrics(inst, routes, reward):
    from vrptw_env import _solution_stats
    stats = _solution_stats(inst, routes)
    served = len({int(n) for r in routes for n in r if n})
    return dict(vehicles=sum(any(n != 0 for n in r) for r in routes),
                travel_time=float(stats['total_travel_time']), distance=float(stats['total_dist']),
                late_count=int(stats['late_count']), total_late=float(stats['total_late']),
                served=served, unserved=int(inst['n_customers'])-served, reward=float(reward))


def log_solution(inst, routes, reward, epoch):
    import mlflow
    name = inst['name']
    metrics = route_metrics(inst, routes, reward)
    mlflow.log_metrics({f'test/{name}/{k}': v for k, v in metrics.items()}, step=epoch)
    mlflow.log_dict(dict(instance=name, epoch=epoch, metrics=metrics,
                         routes=[[int(n) for n in r] for r in routes],
                         lateness_rule='simulator _solution_stats, including depot lateness'),
                    f'evaluation/epoch_{epoch}/{name}.json')
