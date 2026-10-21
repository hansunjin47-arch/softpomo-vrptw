import json
import unittest
import torch
from experiments.concurrent_fleet_v9_guided import test_guidance as fixtures
from experiments.concurrent_fleet_v9_guided.prompts import build_train_request, build_test_request, parse_test_response


class Tests(unittest.TestCase):
    def setUp(self):
        self.env = fixtures.Tests().env()
        self.event = dict(event_type='ACCIDENT', affected_nodes=[1,3], regions=['E'],
            start_min=30., end_min=60., revealed_at_min=30., vehicles_involved=2, multiplier=3.)
        self.args = (self.env.inst, self.env.params, fixtures.GROUPS, [self.event])
        self.template = dict(model='unchanged-model', temperature=0)

    def test_train_offline_reference(self):
        req = build_train_request(*self.args, template=self.template,
            training_reference={'historical_route':[0,1,3,0]}, reference_id='train500')
        ctx = json.loads(req['messages'][1]['content'])
        self.assertEqual(req['model'], 'unchanged-model')
        self.assertEqual(ctx['reference_routes'], [[1,3]])
        self.assertEqual(len(ctx['public_events']), 1)
        self.assertNotIn('base_travel_time_min', ctx)
        self.assertNotIn('structured_context', ctx)
        self.assertNotIn('fleet', ctx)
        self.assertNotIn('"multiplier":', json.dumps(ctx))
        self.assertTrue(ctx['cache_activation']['requires_disclosure_gate'])
        with self.assertRaises(ValueError):
            build_train_request(*self.args, template=self.template, training_reference={},
                                reference_id='train500', cache_stage='initial')

    def test_test_initial_no_reference_or_future(self):
        req = build_test_request(*self.args, template=self.template)
        ctx = json.loads(req['messages'][1]['content'])
        self.assertEqual(ctx['public_events'], [])
        self.assertNotIn('reference_routes', ctx)
        with self.assertRaises(TypeError):
            build_test_request(*self.args, template=self.template, training_reference={})
        with self.assertRaises(ValueError):
            build_test_request(*self.args, template=self.template, group_source='test_rollout')

    def test_live_refresh_and_output(self):
        self.env.step(torch.zeros(4, dtype=torch.long), True)
        self.env.time[:] = 30
        previous={'confidence':[dict(cluster=1,customer=1,score=.9),
            dict(cluster=2,customer=2,score=.3),dict(cluster=1,customer=3,score=.8)]}
        req = build_test_request(*self.args, template=self.template, env=self.env, previous_confidence=previous)
        ctx = json.loads(req['messages'][1]['content'])
        self.assertEqual(ctx['selectable_customers'], [2,3])
        self.assertEqual(len(ctx['public_events']), 1)
        self.assertNotIn('"multiplier":', json.dumps(ctx))
        self.assertEqual([e['customer'] for e in ctx['previous_confidence']['confidence']], [2,3])
        schema=req['response_format']['json_schema']['schema']['properties']['confidence']
        self.assertEqual(schema['type'], 'object')
        self.assertEqual(schema['required'], ['2','3'])
        self.assertEqual(set(schema['properties']), {'2','3'})
        self.assertFalse(schema['additionalProperties'])
        payload={'confidence':[dict(cluster=2,customer=2,score=.3),dict(cluster=1,customer=3,score=.8)]}
        scores=parse_test_response(payload, fixtures.GROUPS, [2,3])
        self.assertEqual(scores[0][1], 0.)
        payload['confidence'].append(dict(cluster=1,customer=1,score=.5))
        with self.assertRaises(ValueError): parse_test_response(payload, fixtures.GROUPS, [2,3])

    def test_empty_and_invalid_subset(self):
        scores=parse_test_response({'confidence':[]}, fixtures.GROUPS, [])
        self.assertTrue(all(v == 0 for row in scores.values() for v in row.values()))
        for entries in ([dict(cluster=2,customer=3,score=.5)],
                        [dict(cluster=1,customer=3,score=1.1)],
                        [], [dict(cluster=1,customer=3,score=.5)]*2):
            with self.assertRaises(ValueError):
                parse_test_response({'confidence':entries}, fixtures.GROUPS, [3])

    def test_keyed_scores(self):
        scores = parse_test_response({'confidence': {'2': .3, '3': .8}}, fixtures.GROUPS, [2,3])
        self.assertEqual(scores[0][1], 0.)
        self.assertEqual(scores[0][3], .8)
        self.assertEqual(scores[1][2], .3)
        for values in ({'2': .3}, {'2': .3, '3': .8, '1': 0},
                       {'2': True, '3': .8}, {'2': float('nan'), '3': .8},
                       {'2': 1.1, '3': .8}, {'2': '0.3', '3': .8}):
            with self.assertRaises(ValueError):
                parse_test_response({'confidence': values}, fixtures.GROUPS, [2,3])


if __name__ == '__main__': unittest.main()
