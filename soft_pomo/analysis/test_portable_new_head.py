"""Portable workflow checks with synthetic LLM replies; no paid API/GPU."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch
import json
from experiments.concurrent_fleet_pair_kim import portable as p

class PortableChecks(unittest.TestCase):
    def args(self,case,root):
        return S(case=case,vehicle_cost=100,artifact=root/'artifact',allow_api=False,
                 out=root/'job',method='newhead',head_beta=5.,seed=42,resume=False,smoke_updates=1)

    def test_all_inputs(self):
        for case in p.DEN:
            _,split,groups,cfg=p.inputs(S(case=case,vehicle_cost=100))
            self.assertEqual((len(split['train']),len(split['test'])),(20,80))
            self.assertEqual(cfg['trainer']['epochs'],1000)

    def test_prepare_and_train_wiring(self):
        for case in ('c101','r101','rc101'):
            with tempfile.TemporaryDirectory() as temp:
                a=self.args(case,Path(temp));calls=[]
                def response(client,request):
                    context=json.loads(request['messages'][1]['content'])
                    if case=='c101':
                        self.assertIn('SEQUENTIAL-routing',request['messages'][0]['content'])
                        self.assertIn('CONCURRENT routing',request['messages'][0]['content'])
                        self.assertEqual(context['rl_reference_example']['routing_mode'],'sequential')
                        self.assertAlmostEqual(context['rl_reference_example']['objective_coefficients']['vehicles'],.1)
                        self.assertNotAlmostEqual(context['objective_coefficients']['vehicles'],.1)
                    calls.append(context)
                    return {'confidence':{str(n):.5 for n in context['selectable_customers']}}
                with patch.object(p.RetryingClient,'__call__',response):p.prepare(a)
                self.assertEqual(len(calls),21)
                self.assertEqual(p.read(a.artifact/'complete.json')['train_count'],20)
                with patch.object(p.trainer,'run') as train:p.train(a)
                self.assertEqual(train.call_args.args[0].seed,42)
                self.assertIs(p.trainer.FleetPolicy,p.ConfidenceHeadGuidedPolicy)
                self.assertEqual(train.call_args.args[0].resume_from,None)
                self.assertEqual(train.call_args.args[0].config,a.out/'config.json')
                # Restarting cache generation reuses validated files, no calls.
                with patch.object(p.RetryingClient,'__call__',side_effect=AssertionError('API called')):p.prepare(a)

    def test_c_pure_remains_optional(self):
        with tempfile.TemporaryDirectory() as temp:
            a=self.args('c101',Path(temp))
            a.method='pure'
            with patch.object(p.trainer,'run') as train:p.train(a)
            self.assertIs(p.trainer.FleetPolicy,p.FleetPolicy)
            self.assertEqual(train.call_args.args[0].reward_denominator,828.94)

    def test_other_reference_setting_guard_unchanged(self):
        source,_,_,cfg=p.inputs(S(case='r101',vehicle_cost=200))
        with self.assertRaises(AssertionError):p.validate_reference(p.read(source/'reference.json'),'r101',cfg)

if __name__=='__main__':unittest.main()
