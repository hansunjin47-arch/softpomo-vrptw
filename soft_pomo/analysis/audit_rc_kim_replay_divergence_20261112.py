"""CPU provenance/state audit of original versus replay, not an intervention."""
import json
import copy
import torch
from analysis import train_rc_k17_curriculum_20261112 as pilot
from analysis.prepare_rc_kim_curriculum_20261112 import CurriculumGuidedPolicy
from analysis.train_rc_kim_curriculum_20261112 import OUT as ORIGINAL
from analysis.replay_rc_kim_free_tail_20261112 import OUT as REPLAY


def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def main():
    old={r['update']:r for r in map(json.loads,(ORIGINAL/'curriculum/training.jsonl').read_text().splitlines())}
    new={r['update']:r for r in map(json.loads,(REPLAY/'training.jsonl').read_text().splitlines())}
    p=pilot.read(REPLAY/'protocol.json');cfg=p['config']
    hashes_ok=all(pilot.digest(pilot.ROOT/k)==v for k,v in p['hashes'].items())
    saved=torch.load(ORIGINAL/'curriculum/checkpoint-30.pt',map_location='cpu',weights_only=False)
    model=CurriculumGuidedPolicy(**cfg['model']).cpu();model.load_state_dict(saved['model_state_dict'],strict=True)
    opt=torch.optim.Adam(model.parameters(),**cfg['optimizer']['optimizer'])
    opt.load_state_dict(copy.deepcopy(saved['optimizer_state_dict']))
    checks=dict(source_hashes_match=hashes_ok,model_state_roundtrip_exact=same(model.state_dict(),saved['model_state_dict']),
        adam_state_roundtrip_exact=same(opt.state_dict(),saved['optimizer_state_dict']),
        data_order_exact=all(old[u]['name']==new[u]['name']==p['data_order'][u-1] for u in new),
        amp_scale_at_first_update_matches=old[601]['amp_scale']==new[601]['amp_scale'],
        saved_scaler=saved['scaler'],rng_runtime_exact_assert_recorded=p['restored_rng_exact'])
    diffs=[]
    for u in new:
        changed={k:dict(original_run=old[u].get(k),replayed_run=new[u].get(k)) for k in old[u]
            if k not in ('seconds','peak_gpu_mb','target','fallback_count') and old[u].get(k)!=new[u].get(k)}
        diffs.append(dict(update=u,changed=changed))
    final=pilot.read(REPLAY/'report.json')['final_comparison']['means']
    pilot.write(REPLAY/'divergence_audit.json',dict(checks=checks,
        first_any_difference=next(r for r in diffs if r['changed']),
        first_rollout_difference=next(r for r in diffs if any(k in r['changed'] for k in ('reward','vehicles','total_late'))),
        final_means=dict(original_run=final['control'],replayed_run=final['curriculum']),
        replay_reproduced_collapse=False,original_fixed40_conclusion='FAIL remains; replay is diagnostic, not replacement or selected improvement',
        code_path_review=['Original evaluates at30, restores PythonCPUCUDA RNG then saves checkpoint; replay restores exactly that saved state.',
            'Neither run evaluates between updates601 and800; optional probes omitted.',
            'Both use same CachedEnv/guide/caches/load/mask/epsilon.01/AMP/Adam/train_batch, batch128 except final68.',
            'New object reconstruction, optimizer state loading, allocator/library runtime state differ from uninterrupted original; these are not fully captured by checkpoint.',
            'Original per-update tensors/RNG/optimizer snapshots before779 absent. Equality of scalar601 logs does not prove identical gradients or post601 weights.',
            'No causal proof of GPU nondeterminism; need paired short continuation tensor comparisons before assigning cause.'],
        all_update_differences=diffs))
    print(checks,flush=True)


if __name__=='__main__':main()
