"""Load Phase2 reference groups and apply per-vehicle accident disclosure gating."""
from pathlib import Path


def load_reference_cache(trainer, inst):
    import torch
    from run import validate_cache
    from train_soft_cluster import _parse_cluster_cache, _parse_cluster_depot

    name = inst['name']
    if name in trainer._cluster_conf_cache:
        return
    cached = validate_cache(Path(trainer._llm_cache_dir) / (name + '_rl_cluster.json'))
    clusters, scores = _parse_cluster_cache(cached)
    tensor = trainer._build_cluster_conf_tensor(clusters, scores, len(clusters),
        inst['n_customers'], _parse_cluster_depot(cached))
    trainer._cluster_assign_cache[name] = clusters
    accidents = [e for e in inst['preset_events'] if e['type'] == 'ACCIDENT']
    if accidents:
        if len(accidents) != 1:
            raise ValueError('This runner supports one event per scenario')
        trainer._acc_stage_events[name] = [dict(
            t_start=float(accidents[0]['trigger_time']) / float(inst['T']), tensor=tensor)]
        trainer._cluster_conf_cache[name] = torch.zeros_like(tensor)
    else:
        trainer._cluster_conf_cache[name] = tensor
    trainer._original_cluster_conf[name] = trainer._cluster_conf_cache[name].clone()
