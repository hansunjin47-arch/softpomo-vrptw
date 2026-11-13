"""Offline CPU integration checks for the exact cached Kim server path."""
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'server'))
from travel_time_prompt_contract import VERSION, validate_config
import train_cached as runner


def main(variant):
    run=ROOT/f'result/rc101_pure_train20_b128_s42_config{variant}_travel_time_1000_20261010_v2'
    sys.path.insert(0,str(run/'source'))
    cache=ROOT/f'result/rc101_kim_train20_config{variant}_{VERSION}'
    config=runner.read(run/'config.json')
    validate_config(config,variant)
    config.update(method='kim',cache_dir=str(cache/'cache'),
        llm=runner.read(ROOT/'result/rc101_kim_independent1000/settings.json')['llm'])
    config['llm'].update(reward_config=variant)
    config['trainer'].update(cluster_mode='kim2006',bias_mode='cluster',start_mode='random',pomo_size=100,
        benchmark='rc1',experience_refresh_epoch=-1,rl_recluster_epoch=None,require_reference_start=False)
    out=ROOT/f'analysis/kim_tt_server_checks/{variant}'
    manifest=runner.read(cache/'cache_manifest.json')
    for name,digest in manifest['files'].items():
        assert runner.sha(cache/'cache'/name)==digest
    trainer=runner.build_trainer(config,run/'source',out,cpu=True)
    import torch
    checks=[]
    for inst in trainer.instance_pool:
        name=inst['name']
        saved=runner.validate_cache(cache/'cache'/(name+'_cluster.json'))
        import train_soft_cluster as guided
        groups,scores=guided._parse_cluster_cache(saved)
        expected=trainer._build_cluster_conf_tensor(groups,scores,len(groups),100,guided._parse_cluster_depot(saved))
        assert trainer._cluster_assign_cache[name]==groups
        if '_ACC_' in name:
            assert torch.count_nonzero(trainer._cluster_conf_cache[name])==0
            stage=trainer._acc_stage_events[name][0]
            assert abs(stage['t_start']-inst['preset_events'][0]['trigger_time']/inst['T'])<1e-8
            assert torch.equal(stage['tensor'],expected)
        else:
            assert torch.equal(trainer._cluster_conf_cache[name],expected)
    for kind in ('RAIN','ACC'):
        inst=next(i for i in trainer.instance_pool if '_'+kind+'_' in i['name'])
        reward,loss=trainer._train_one_batch(inst,2)
        assert math.isfinite(reward) and math.isfinite(loss)
        checks.append(dict(kind=kind,reward=reward,loss=loss))
    contract=dict(config=config)
    trainer._current_epoch=0
    runner.save_checkpoint(trainer,0,out/'checkpoint.pt',contract)
    before=torch.get_rng_state().clone()
    torch.rand(10)
    runner.restore(trainer,out/'checkpoint.pt',contract)
    assert torch.equal(before,torch.get_rng_state())
    runner.write(out/'checked_config.json',config)
    runner.write(out/'report.json',dict(variant=variant,cache_count=20,cpu_batch=2,
        train_batch_config=128,api_disabled=True,accident_stages_verified=True,checkpoint_roundtrip=True,
        gpu_batch128_tested=False,checks=checks,server_runner_sha256=runner.sha(ROOT/'server/train_cached.py')))
    print('PASS',variant,'20 caches, accident stages, backward steps, checkpoint/RNG roundtrip',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',required=True,choices=['A','C'])
    main(parser.parse_args().variant)
