import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import analysis.pair_kim_rl_example as m
from experiments.concurrent_fleet_v9_guided.prompts import COMMON


class CacheTests(unittest.TestCase):
    def test_pair_selection_description_for_all_prompt_stages(self):
        request = {'messages': [dict(role='system', content=COMMON),
                                dict(role='user', content='{"public_events": []}')]}
        original = json.dumps(request)
        for offline in (False, True):
            for base in (None, {'confidence': []}):
                with self.subTest(offline=offline, base=base):
                    enriched = m.enrich(request, base, {'scenario': 'reference'}, offline=offline)
                    text = enriched['messages'][0]['content']
                    self.assertIn('RL chooses NEW(customer)', text)
                    self.assertIn('selecting an existing pair closes initial dispatch', text)
                    self.assertIn('environment designates one ready vehicle', text)
                    self.assertIn('RL cannot choose a different vehicle', text)
                    self.assertNotIn('pair after initial dispatch', text)
                    self.assertIn('Depot return retires that vehicle', text)
                    self.assertIn('does not change a committed destination', text)
        self.assertEqual(json.dumps(request), original)

    def test_train_cache_generation_without_paid_calls(self):
        calls = []
        class Client:
            def __init__(self, *args, **kwargs):
                pass
            def __call__(self, request):
                context = json.loads(request['messages'][1]['content'])
                calls.append(context)
                return {'confidence': {str(n): .4 for n in context['selectable_customers']}}
        with tempfile.TemporaryDirectory() as tmp:
            art = Path(tmp)
            m.write(art/'reference.json', dict(scenario=m.EXAMPLE))
            with patch.object(m, 'ART', art), patch.object(m, 'RetryingClient', Client), patch.object(m.adapter, 'ARTIFACT', art):
                m.prepare()
                self.assertEqual(len(calls), 21)
                self.assertEqual(calls[0]['public_events'], [])
                self.assertNotIn('prior_base_confidence', calls[0])
                self.assertTrue(all(c['rl_reference_example']['scenario'] == m.EXAMPLE for c in calls))
                self.assertTrue(all('prior_base_confidence' in c for c in calls[1:]))
                caches = list((art/'train_cache').glob('*.json'))
                self.assertEqual(len(caches), 20)
                base = m.read(art/'base_confidence.json')
                for path in caches:
                    cache = m.read(path)
                    if '_ACC_' in path.name:
                        self.assertEqual(cache['initial'], base)
                        self.assertIsNotNone(cache['post_disclosure'])
                m.prepare()
                self.assertEqual(len(calls), 21)

    def test_test_prompt_keeps_disclosure_filter(self):
        split = m.read(m.OLD/'split.json')
        row = next(r for r in split['test'] if '_ACC_' in r['name'])
        inst, public = m.load(row)
        req = m.build_test_request(inst, m.read(m.PURE/'protocol.json')['configs']['10']['environment'],
            m.read(m.OLD/'groups.json'), public, template=m.read(m.OLD/'request_template.json'))
        original = json.loads(req['messages'][1]['content'])
        enriched = m.enrich(req, {'confidence': []}, {'scenario': m.EXAMPLE})
        ctx = json.loads(enriched['messages'][1]['content'])
        self.assertEqual(ctx['public_events'], [])
        self.assertEqual(ctx['selectable_customers'], original['selectable_customers'])
        self.assertNotIn('reference_routes', ctx)
        self.assertNotIn('observed_performance', ctx)


if __name__ == '__main__':
    unittest.main()
