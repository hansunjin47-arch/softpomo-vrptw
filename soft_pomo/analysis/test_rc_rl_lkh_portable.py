import argparse,contextlib,io,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from experiments.concurrent_fleet_pair_kim import rl_lkh as m

class PortableComparison(unittest.TestCase):
    def args(self,root,late=10):
        return argparse.Namespace(late=late,artifact=root/'cache',out=root/'run',seed=44,allow_api=False,resume=False,validate_only=False)
    def prepare(self,a):
        _,_,groups,_,_,_=m.inputs(a.late)
        requests=[]
        class FakeClient:
            def __init__(self,folder,validate,**kw):self.validate=validate
            def __call__(self,req):
                payload={'confidence':[dict(cluster=k+1,customer=n,score=.5) for k,g in enumerate(groups) for n in g]}
                self.validate(req,payload);requests.append(req);return payload
        with patch.object(m,'RetryingClient',FakeClient),contextlib.redirect_stdout(io.StringIO()):m.prepare(a)
        return requests
    def test_prompts_cache_reuse_and_accident_gate(self):
        import json
        for late in (5,10):
            with tempfile.TemporaryDirectory() as d:
                a=self.args(Path(d),late);requests=self.prepare(a);self.assertEqual(len(requests),21)
                self.assertEqual(self.prepare(a),[])
                for req in requests:
                    ctx=json.loads(req['messages'][1]['content']);c=ctx['objective_coefficients']
                    self.assertAlmostEqual(c['total_lateness_min']/c['driving_time_min'],late)
                    self.assertEqual(ctx['guidance_application']['new_head_beta'],5)
                    self.assertEqual(ctx['rl_lkh_comparison_example']['scenario'],ctx['rl_reference_example']['scenario'])
                    self.assertNotIn('Depot, NEW and STOP have no',req['messages'][0]['content'])
                base=m.read(a.artifact/'base_confidence.json')
                for f in (a.artifact/'train_cache').glob('*ACC*'):
                    cache=m.read(f);self.assertEqual(cache['initial'],base);self.assertIsNotNone(cache['post_disclosure'])
                a.late=5 if late==10 else 10
                with self.assertRaises(AssertionError):self.prepare(a)
    def test_training_configuration_and_cpu_environments(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as d:
            a=self.args(Path(d));self.prepare(a)
            original=m.trainer.train_batch
            try:
                a.validate_only=True
                with contextlib.redirect_stdout(io.StringIO()):m.train(a)
                m.trainer.train_batch=original;a.validate_only=False
                with patch.object(m.trainer,'run') as run:m.train(a)
                cfg=m.read(a.out/'config.json')
                self.assertEqual(cfg['environment']['late_penalty'],10)
                self.assertTrue(cfg['environment']['require_dispatch_capacity'])
                self.assertTrue(cfg['environment']['protect_last_service_vehicle'])
                self.assertTrue(cfg['environment']['normalize_advantage_std'])
                self.assertEqual(run.call_args.args[0].weights,[10])
                self.assertEqual(run.call_args.args[0].seed,44)
            finally:m.trainer.train_batch=original
if __name__=='__main__':unittest.main()
