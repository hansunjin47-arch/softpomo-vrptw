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
        for case in ('r101','rc101'):
            with tempfile.TemporaryDirectory() as temp:
                a=self.args(case,Path(temp));calls=[]
                def response(client,request):
                    context=json.loads(request['messages'][1]['content'])
                    calls.append(context)
                    return {'confidence':{str(n):.5 for n in context['selectable_customers']}}
                with patch.object(p.RetryingClient,'__call__',response):p.prepare(a)
                self.assertEqual(len(calls),21)
                self.assertEqual(p.read(a.artifact/'complete.json')['train_count'],20)
                with patch.object(p.trainer,'run') as train:p.train(a)
                self.assertEqual(train.call_args.args[0].seed,42)
                self.assertIs(p.trainer.FleetPolicy,p.StartHeadGuidedPolicy)
                self.assertEqual(train.call_args.args[0].resume_from,None)
                self.assertEqual(train.call_args.args[0].config,a.out/'config.json')
                # Restarting cache generation reuses validated files, no calls.
                with patch.object(p.RetryingClient,'__call__',side_effect=AssertionError('API called')):p.prepare(a)

    def test_c_requires_independent_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            a=self.args('c101',Path(temp))
            with self.assertRaisesRegex(ValueError,'reference --pure-run'):p.prepare(a)
            a.method='pure'
            with patch.object(p.trainer,'run') as train:p.train(a)
            self.assertIs(p.trainer.FleetPolicy,p.FleetPolicy)
            self.assertEqual(train.call_args.args[0].reward_denominator,828.94)

if __name__=='__main__':unittest.main()
