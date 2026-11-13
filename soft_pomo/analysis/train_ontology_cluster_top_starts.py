"""Ontology v5: one raw-confidence maximum per group, POMO size K."""
import argparse
import types
from pathlib import Path

import train_ontology_scale10_diagnostic as baseline

runner = baseline.runner
runner.OUT = runner.ROOT/'result/rc101_ontology_v5_b128_s44_cluster_top_starts'
original_builder = runner.build_trainer


def starts_from_cache(cache):
    groups = sorted((k for k in cache if not k.startswith('_')), key=int)
    starts = [min((int(n) for n in cache[k]),
                  key=lambda n: (-float(cache[k][str(n)]), n)) for k in groups]
    assert len(starts)==len(set(starts)) and all(n>0 for n in starts)
    return starts


def build_trainer(config, bundle, folder):
    trainer = original_builder(config, bundle, folder)
    trainer._cluster_top_starts = {}
    for inst in trainer.instance_pool:
        cache = runner.read(Path(config['cache_dir'])/(inst['name']+'_rl_cluster.json'))
        trainer._cluster_top_starts[inst['name']] = starts_from_cache(cache)

    def det_starts(self, inst, pomo_size):
        starts = self._cluster_top_starts[inst['name']]
        assert pomo_size==len(starts)
        return runner.torch.tensor([starts],device=self.device,dtype=runner.torch.long)
    trainer._det_starts = types.MethodType(det_starts, trainer)

    # The frozen trainer reads trainer_params before loading an instance. Set K
    # at each training batch / evaluation rollout, not once globally.
    for method in ('_train_one_batch','_best_solution'):
        original = getattr(trainer,method)
        def wrapped(self, inst, *args, _original=original, **kwargs):
            k=len(self._cluster_top_starts[inst['name']])
            self.trainer_params['pomo_size']=k
            self.env.pomo_size=k
            result=_original(inst,*args,**kwargs)
            history=self.env.selected_node_list
            expected=runner.torch.tensor(self._cluster_top_starts[inst['name']],device=history.device)
            assert history.shape[1]==k
            assert runner.torch.equal(history[:,:,1],expected[None,:].expand(history.shape[0],-1))
            return result
        setattr(trainer,method,types.MethodType(wrapped,trainer))
    return trainer


def setup():
    config,contract,checkpoint=baseline.setup()
    config['trainer']['start_mode']='det_conf'
    inventory={}
    for split,folder in [('train',runner.BUNDLE/'cache'),
                         ('test',runner.OLD/'eval_1000_test80/cache')]:
        inventory[split]={e['name']:starts_from_cache(runner.read(folder/(e['name']+'_rl_cluster.json')))
                          for e in runner.read(runner.TEST/'split.json')[split]}
    contract.update(experiment='One maximum raw-score customer per cluster; dynamic POMO=K; scale10 beta5.',
        wrapper_sha256=runner.sha(Path(__file__)), starts=inventory,
        baseline=str(runner.ROOT/'result/rc101_ontology_v5_b128_s44_scale10_diagnostic'),
        tie_break='Smallest customer ID among tied raw-score maxima; depot excluded.',
        interpretation='Changes start coverage and multi-start baseline size; evaluation uses 8*K candidates, not 800.',
        disclosure_caveat='Offline scenario-specific cache selects initial starts even before accident disclosure. '
                          'Subsequent logit bias disclosure is unchanged. Not an online-only information protocol.')
    return config,contract,checkpoint


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['smoke','train'])
    args=parser.parse_args()
    assert starts_from_cache({'0':{'2':.5,'1':.5},'1':{'3':0.1},'_depot':{'0':.9}})==[1,3]
    runner.build_trainer=build_trainer
    runner.setup=setup
    runner.main(args.mode)
    if args.mode=='smoke':
        config,_,checkpoint=setup()
        config['cache_dir']=str(runner.OLD/'eval_1000_test80/cache')
        entries=runner.read(runner.TEST/'split.json')['test']
        config['trainer']['train_instances']=[next(e['name'] for e in entries if '_'+kind+'_' in e['name'])
                                             for kind in ('RAIN','ACC')]
        trainer=build_trainer(config,runner.TEST,runner.OUT/'smoke_eval')
        runner.restore(trainer,checkpoint)
        trainer.model.eval()
        for inst in trainer.instance_pool:
            with runner.torch.no_grad():
                routes,reward=trainer._best_solution_aug(inst,n_aug=8)
            assert sorted(n for r in routes for n in r if n)==list(range(1,101))
            print('EVAL PASS',inst['name'],len(trainer._cluster_top_starts[inst['name']]),reward,flush=True)
