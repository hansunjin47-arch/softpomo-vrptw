import copy
import math
import unittest
from pathlib import Path
import numpy as np

from audit_event_relations import ROOT, read, load_solomon
from routing_ontology import context, validate, visit_diagnosis, attach_reference_groups


class OntologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = ROOT / 'result/rc101_relation_train20_cache_v1'
        cls.graphs = [read(root/'requests'/name/'relations.json')
                      for name in read(root/'protocol.json')['train_names']]
        protocol = read(ROOT/'result/rc101_pure500_fullconfidence_phase2/protocol.json')
        inst = load_solomon(str(Path(protocol['source_config']['trainer']['data_dir'])/(protocol['train_names'][0]+'.txt')))
        cls.coords = np.vstack([inst['depot_xy'].numpy(),inst['node_xy'].numpy()])*inst['max_coord']

    def test_all_twenty_and_no_mutation(self):
        for graph in self.graphs:
            original = copy.deepcopy(graph)
            payload = context(graph, self.coords)
            self.assertEqual(graph, original)
            self.assertTrue(all('source' in d for d in payload['ontology']['relations'].values()))
            columns = payload['facts']['leg_columns']
            for raw, packed in zip(graph['legs'], payload['facts']['legs']):
                self.assertEqual(len(columns),len(packed))
                row = dict(zip(columns,packed))
                self.assertAlmostEqual(row['base_travel'],raw['baseline']['travel'],places=6)
                self.assertAlmostEqual(row['reference_wait'],raw['reference']['wait'],places=6)

    def test_invalid_domain(self):
        graph = copy.deepcopy(self.graphs[0])
        graph['relations'].append(dict(subject='event', predicate='locatedIn', object='N'))
        with self.assertRaisesRegex(ValueError, 'domain'):
            validate(graph, self.coords)

    def test_missing_exposure(self):
        graph = copy.deepcopy(self.graphs[0])
        graph['relations'] = [e for e in graph['relations'] if e['predicate'] != 'exposedTo']
        with self.assertRaisesRegex(ValueError, 'Missing'):
            validate(graph, self.coords)

    def test_corrupt_delay(self):
        graph = copy.deepcopy(self.graphs[0])
        graph['legs'][0]['incoming_delay'] += 1
        with self.assertRaisesRegex(ValueError, 'derived quantity'):
            validate(graph, self.coords)

    def test_hidden_field(self):
        graph = copy.deepcopy(self.graphs[0])
        graph['nodes'][0]['public_observation']['multiplier'] = 2
        with self.assertRaisesRegex(ValueError, 'Nonpublic'):
            validate(graph, self.coords)

    def test_relation_minutes(self):
        for value in (999999, float('nan'), -1):
            graph = copy.deepcopy(next(g for g in self.graphs if any('minutes' in e for e in g['relations'])))
            next(e for e in graph['relations'] if 'minutes' in e)['minutes'] = value
            with self.assertRaisesRegex(ValueError, 'relation minutes'):
                validate(graph, self.coords)

    def test_wrong_region_even_if_unaffected(self):
        graph = copy.deepcopy(self.graphs[0])
        e = next(e for e in graph['relations'] if e['predicate']=='locatedIn')
        e['object'] = 'NE' if e['object']!='NE' else 'N'
        with self.assertRaisesRegex(ValueError, 'spatial'):
            validate(graph, self.coords)

    def test_duplicate_departure_disagrees(self):
        graph = copy.deepcopy(self.graphs[0])
        next(n for n in graph['nodes'] if n['type']=='ScheduledLeg')['reference_departure'] += 10
        with self.assertRaisesRegex(ValueError, 'timing'):
            validate(graph, self.coords)

    def test_interval_disagrees(self):
        graph = copy.deepcopy(self.graphs[0])
        next(n for n in graph['nodes'] if n['id']=='active')['end'] += 1
        with self.assertRaisesRegex(ValueError, 'Interval'):
            validate(graph, self.coords)

    def test_prompt_semantics(self):
        from prepare_ontology_train20 import INSTRUCTIONS
        self.assertIn('J=0.04*K+T/5760+0.0075*Lt',INSTRUCTIONS)
        self.assertIn('NO mandatory ordering',INSTRUCTIONS)
        self.assertIn('not a certificate',INSTRUCTIONS)
        self.assertIn('cannot express time-conditional',INSTRUCTIONS)
        self.assertNotIn('ANY stop with slack < 0 must',INSTRUCTIONS)

    def test_visit_identity_and_typed_relations(self):
        for graph in self.graphs:
            payload = context(graph,self.coords)
            facts = payload['facts']
            nodes = {n['id']:n['type'] for n in graph['nodes']}
            visits = facts['visits']
            self.assertEqual(len(visits),len({v[0] for v in visits}))
            nodes.update({v[0]:'ServiceVisit' for v in visits})
            for subject,predicate,obj in facts['relations']:
                definition = payload['ontology']['relations'][predicate]
                self.assertIn(nodes[subject],definition['domain'])
                self.assertIn(nodes[obj],definition['range'])
            returns = [v for v in visits if v[3]=='c0']
            self.assertEqual(len(returns),len({v[1] for v in visits}))
            self.assertEqual(len(returns),len({v[0] for v in returns}))

    def test_diagnostic_counterexamples(self):
        def leg(exposed=False, arrival=0, outgoing=0, base_late=0, late=0):
            return dict(reference_exposure={'exposed':exposed}, arrival_delay=arrival,
                outgoing_delay=outgoing, baseline={'late':base_late},
                reference={'late':late}, added_lateness=late-base_late)
        # Exposure can be entirely absorbed, without increasing lateness.
        a = visit_diagnosis(leg(True,10,0))
        self.assertTrue(a['direct_exposure'] and a['fully_absorbed_here'])
        self.assertFalse(a['event_added_lateness'])
        # Unexposed movement can still reach a newly late visit.
        b = visit_diagnosis(leg(False,10,10,0,5))
        self.assertTrue(b['indirect_only'] and b['new_late_visit'])
        # Late before the event does not establish event causation.
        c = visit_diagnosis(leg(False,0,0,5,5))
        self.assertTrue(c['preexisting_lateness'])
        self.assertFalse(c['event_added_lateness'] or c['new_late_visit'])
        # Already late and worsened is different from newly late.
        d = visit_diagnosis(leg(True,3,3,5,8))
        self.assertTrue(d['preexisting_lateness'] and d['event_added_lateness'])
        self.assertFalse(d['new_late_visit'])
        self.assertFalse(any(visit_diagnosis(leg()).values()))
        self.assertFalse(any(k in a for k in ('score','urgent','recommended_action')))

    def test_bias_semantics_not_probability(self):
        def odds(a,b):
            return math.exp(5*a)/math.exp(5*b)
        self.assertAlmostEqual(odds(.5,.5),1)
        self.assertGreater(odds(.5,0),12)
        self.assertAlmostEqual(odds(0,0),1)
        self.assertAlmostEqual(odds(.85,.75),math.exp(.5))

    def test_unknown_event_rejected(self):
        graph = copy.deepcopy(next(g for g in self.graphs if g['nodes'][0]['type']=='AccidentEvent'))
        graph['nodes'][0]['public_observation']['event_type']='UNKNOWN_EVENT'
        with self.assertRaisesRegex(ValueError,'Unknown event'):
            validate(graph,self.coords)

    def test_negative_travel_even_with_unchanged_differences(self):
        graph = copy.deepcopy(self.graphs[0])
        leg = graph['legs'][0]
        shift = max(leg['baseline']['travel'],leg['reference']['travel'])+1
        for schedule in ('baseline','reference'):
            leg[schedule]['travel'] -= shift
        with self.assertRaisesRegex(ValueError,'Negative schedule'):
            validate(graph,self.coords)

    def test_explicit_group_links(self):
        packet = read(ROOT/'result/rc101_ontology_train20_v4/requests/RC101_RAIN_P018/context.json')
        instance = packet.pop('instance')
        attach_reference_groups(packet,instance)
        self.assertEqual(sum(e[1]=='memberOfReferenceGroup' for e in packet['facts']['relations']),100)
        self.assertEqual(sum(e[1]=='groupFromRoute' for e in packet['facts']['relations']),len(instance['groups']))
        broken = copy.deepcopy(instance)
        broken['groups'][0]['reference_order'].reverse()
        with self.assertRaisesRegex(ValueError,'order'):
            attach_reference_groups(copy.deepcopy(packet),broken)


if __name__ == '__main__':
    unittest.main()
