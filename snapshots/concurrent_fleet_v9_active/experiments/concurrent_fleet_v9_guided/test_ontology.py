import json
import unittest
import torch
from experiments.concurrent_fleet_v9_guided.test_guidance import Tests as Fixture, GROUPS
from experiments.concurrent_fleet_v9_guided.ontology import build_context, validate


class Tests(unittest.TestCase):
    def setUp(self):
        self.env = Fixture().env()
        self.event = dict(event_type='ACCIDENT', affected_nodes=[1,3], regions=['E'],
                          revealed_at_min=30., start_min=30., end_min=60.,
                          vehicles_involved=2, multiplier=2.5)

    def test_initial_and_disclosure(self):
        inst = self.env.inst
        ctx = build_context(inst, GROUPS, [self.event])
        self.assertFalse(any(n['type']=='PublicEventReport' for n in ctx['nodes']))
        self.assertNotIn('multiplier', json.dumps(ctx['nodes']))
        self.env.time[:] = 30
        ctx = build_context(inst, GROUPS, [self.event], env=self.env)
        reports = [n for n in ctx['nodes'] if n['type']=='PublicEventReport']
        self.assertEqual(len(reports), 1)
        self.assertNotIn('multiplier', reports[0])

    def test_committed_customer_and_hidden_times(self):
        self.env.step(torch.zeros(4, dtype=torch.long), True)
        ctx = build_context(self.env.inst, GROUPS, [], env=self.env)
        self.assertIn(dict(subject='vehicle:1', predicate='committedTo', object='customer:1'), ctx['relations'])
        self.assertNotIn(dict(subject='customer:1', predicate='unassignedAt', object='snapshot'), ctx['relations'])
        self.env.arrival += 123
        self.env.available += 456
        self.assertEqual(ctx, build_context(self.env.inst, GROUPS, [], env=self.env))

    def test_train_evidence_separate(self):
        with self.assertRaises(ValueError):
            build_context(self.env.inst, GROUPS, [], training_reference={'route': [1,2]})
        ctx = build_context(self.env.inst, GROUPS, [], mode='train',
            training_reference={'route': [1,2]}, reference_id='train-checkpoint-500',
            offline_training_reports=[self.event])
        self.assertFalse(ctx['offline_training_reference']['is_current_state'])
        self.assertFalse(any(n['type']=='PublicEventReport' for n in ctx['nodes']))
        self.assertNotIn('multiplier', ctx['offline_training_reports']['reports'][0])
        ctx['mode'] = 'test'
        with self.assertRaises(ValueError): validate(ctx)

    def test_invalid_groups_and_reports(self):
        with self.assertRaises(ValueError): build_context(self.env.inst, [[1,2]], [])
        with self.assertRaises(ValueError):
            build_context(self.env.inst, GROUPS, [dict(self.event, end_min=0)])


if __name__ == '__main__': unittest.main()
