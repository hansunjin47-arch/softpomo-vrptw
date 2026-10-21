import copy
import json
import unittest
import torch
from experiments.concurrent_fleet_guided import test_guidance as fixtures
from experiments.concurrent_fleet_guided.prompts import build_request, parse_response, observed_state

GROUPS = fixtures.GROUPS


class Tests(unittest.TestCase):
    def setUp(self):
        self.env = fixtures.Tests().env(count=1)
        self.template = {'model': 'test-model', 'messages': ['obsolete'], 'max_tokens': 2000}
        self.response = {'confidence': [dict(cluster=k+1, customer=n, score=.4)
            for k, g in enumerate(GROUPS) for n in g]}

    def request(self, events=(), **kwargs):
        return build_request(self.env.inst, self.env.params, GROUPS, events,
                             template=self.template, **kwargs)

    def context(self, **kwargs):
        return json.loads(self.request(**kwargs)['messages'][1]['content'])

    def test_initial_contract(self):
        request = self.request()
        context = json.loads(request['messages'][1]['content'])
        self.assertEqual(context['fleet'], {'clock_min': 0.})
        self.assertAlmostEqual(context['objective_coefficients']['vehicles'], 1/14)
        self.assertEqual(request['model'], 'test-model')
        self.assertEqual(self.template['messages'], ['obsolete'])
        self.assertEqual(request['response_format']['json_schema']['schema']['properties']
                         ['confidence']['minItems'], 3)
        self.assertIn('ONE shared clock', request['messages'][0]['content'])

    def test_disclosure_and_private_fields(self):
        events = [dict(event_type='rain', revealed_at_min=0, multiplier=99),
                  dict(event_type='accident', revealed_at_min=10, news_text='incident')]
        context = self.context(events=events)
        self.assertEqual(context['public_events'], [{'event_type':'rain', 'revealed_at_min':0}])
        self.env.inst['preset_events'] = [dict(multiplier=999, secret='future')]
        self.assertEqual(context, self.context(events=events))
        with self.assertRaises(ValueError):
            self.request(events=[dict(event_type='accident')])

    def test_snapshot_and_refresh(self):
        self.env.step(torch.tensor([0]), validate=True)
        state = observed_state(self.env)
        self.assertEqual(state['unassigned_customers'], [2, 3])
        self.assertNotIn('reserved', state)
        self.assertNotIn('served', state)
        self.assertEqual(state['vehicles'][0]['status'], 'moving')
        self.assertEqual(state['vehicles'][0]['anchor_cluster'], 1)
        self.assertNotIn('arrival', state['vehicles'][0])
        self.env.step(torch.tensor([7]), validate=True)
        context = self.context(env=self.env, previous_confidence=self.response,
            events=[dict(event_type='accident', revealed_at_min=10)])
        self.assertEqual(context['fleet']['clock_min'], float(self.env.time[0]))
        self.assertEqual(context['fleet']['unassigned_customers'], [2, 3])
        self.assertEqual(context['fleet']['vehicles'][0]['status'], 'ready')
        self.assertEqual(len(context['public_events']), 1)
        self.assertEqual(context['previous_confidence'], self.response)
        state = observed_state(self.env)
        self.env.arrival[:] = 99999
        self.env.available[:] = 99999
        self.assertEqual(state, observed_state(self.env))

    def test_train_feedback_gate(self):
        with self.assertRaises(ValueError):
            self.request(training_reference={'routes': [[1, 2, 3]]})
        context = self.context(training=True, training_reference={'routes': [[1, 2, 3]]})
        self.assertIn('train_only_reference', context)
        self.assertNotIn('train_only_reference', self.context())

    def test_response_adapter_and_validation(self):
        scores = parse_response(json.dumps(self.response), GROUPS)
        self.env.update_confidence(scores)
        self.assertEqual(scores, {0: {1:.4, 2:.4}, 1: {3:.4}})
        for key, value in [('cluster', 0), ('cluster', 2), ('customer', 0),
                           ('score', float('nan')), ('score', 2), ('score', True)]:
            payload = copy.deepcopy(self.response)
            payload['confidence'][0][key] = value
            with self.assertRaises(ValueError):
                parse_response(payload, GROUPS)
        payload = copy.deepcopy(self.response)
        payload['confidence'][1] = payload['confidence'][0]
        with self.assertRaises(ValueError):
            parse_response(payload, GROUPS)

    def test_env_mismatch(self):
        with self.assertRaises(ValueError):
            self.request(env=self.env, beta=2)


if __name__ == '__main__':
    unittest.main()
