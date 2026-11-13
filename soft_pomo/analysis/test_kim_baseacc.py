"""Offline regression tests; no API calls or production cache writes."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_kim_baseacc as gen
from phase2_base_event_prompt import make_request as phase2_request


class Tests(unittest.TestCase):
    def test_phase2_feedback_and_base_provenance(self):
        template={'messages':[{'content':'Training contract\n{}'}]}
        base={'fixed_instance':{'groups':[]},'public_event':{},
              'train_execution_feedback':{'scope':'offline_train_only','metrics':{'reward':-2}}}
        event=copy.deepcopy(base)
        event['public_event']={'event_type':'RAIN'}
        event['train_execution_feedback']['metrics']['reward']=-3
        request=phase2_request(template,event,training=True,prior_base_confidence={'0':{}},base_context=base)
        payload=json.loads(request['messages'][0]['content'].splitlines()[-1])
        self.assertEqual(payload['train_execution_feedback']['metrics']['reward'],-3)
        with self.assertRaises(ValueError):phase2_request(template,event,training=True)
        changed=copy.deepcopy(event);changed['fixed_instance']['groups']=[1]
        with self.assertRaises(ValueError):
            phase2_request(template,changed,training=True,prior_base_confidence={},base_context=base)

    def test_phase2_test_rejects_rl_feedback(self):
        folder=gen.OLD/'rc101/test'
        name=gen.io.read(folder/'protocol.json')['names'][0]
        ctx=gen.io.read(folder/'requests'/name/'context.json')
        template=gen.io.read(folder/'requests'/name/'request.json')
        base=copy.deepcopy(ctx);base['scenario']='BASE';base['public_event']={}
        phase2_request(template,ctx,training=False,prior_base_confidence={},base_context=base)
        ctx['train_execution_feedback']={'scope':'offline_train_only'}
        with self.assertRaises(AssertionError):
            phase2_request(template,ctx,training=False,prior_base_confidence={},base_context=base)

    def test_base_has_no_event_or_prior(self):
        request={'model':'test','messages':[{'role':'user','content':'Instructions\n{}'}]}
        context={'scenario':'RC101_ACC_secret','fixed_instance':{'customers':[]},
                 'public_event':{'news_text':'SECRET','start_min':99}}
        original=copy.deepcopy(context)
        req,ctx=gen.make_request(request,context)
        self.assertEqual(ctx,dict(scenario='BASE',fixed_instance={'customers':[]},public_event={}))
        self.assertNotIn('SECRET',json.dumps(req))
        self.assertNotIn('RC101_ACC_secret',json.dumps(req))
        self.assertEqual(context,original)

    def test_acc_contains_prior_without_rl(self):
        req,ctx=gen.make_request({'messages':[{'content':'Instructions\n{}'}]},
            {'scenario':'ACC','fixed_instance':{},'public_event':{'start_min':50}},
            {'0':{'1':.7},'_depot':{'0':0}})
        self.assertEqual(ctx['prior_base_confidence']['0']['1'],.7)
        self.assertIn('YOU previously generated',req['messages'][0]['content'])

    def test_all_families_base_and_event_preparation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(gen,'OUT',Path(tmp)):
            for case in gen.CASES:
                gen.prepare_base(case)
                base=gen.OUT/case/'base'
                context=gen.io.read(base/'requests/BASE/context.json')
                self.assertEqual(context['public_event'],{})
                protocol=gen.io.read(base/'protocol.json')
                # Synthetic valid scores for testing, never used as production guidance.
                data={str(k):{str(n):.5 for n in group} for k,group in enumerate(protocol['clusters'])}
                data['_depot']={str(k):0 for k in range(len(protocol['clusters']))}
                gen.immutable(base/'cache/BASE_cluster.json',data)
                gen.prepare_events(case)
                for section,count in [('train',20),('test',80)]:
                    folder=gen.OUT/case/section
                    hashes=gen.io.read(folder/'request_hashes.json')
                    self.assertEqual(len(hashes),count)
                    for name in hashes:
                        ctx=gen.io.read(folder/'requests'/name/'context.json')
                        self.assertEqual(ctx['prior_base_confidence'],data)
                        self.assertFalse((folder/'requests'/name/'response.txt').exists())


if __name__=='__main__':unittest.main()
