import unittest
import json
from unittest.mock import patch

from generate_kim500_openrouter import parse_response, build_reference, ROOT
from vrptw_env import load_solomon, _solution_stats
from SoftClusterLLMModule import build_all_clusters_prompt
from SoftClusterOntology import SoftClusterOntology


class ResponseTests(unittest.TestCase):
    def test_any_positive_lateness_counts(self):
        from extract_rl_routes import routes_to_cluster_json
        values=[0.0, 1e-9, .19841, .5]
        stops=[dict(node=n+1,late=v,tw_close=10.,arrival=10.+v,travel_time=1.) for n,v in enumerate(values)]
        with patch('extract_rl_routes._solution_stats', return_value={'per_vehicle':[dict(stop_details=stops,travel_time=4.)]}):
            result=routes_to_cluster_json([[0,1,2,3,4,0]],{})
        self.assertEqual(result['_stats']['0']['late_count'],3)

    def test_confidence_only_prompt(self):
        inst=load_solomon(str(ROOT.parent/'data/Solomon/rc102.txt'))
        clusters=[[1,2,3]]
        ref=build_reference(clusters,inst)
        ctx=[SoftClusterOntology(inst).get_cluster_context(clusters[0],0)]
        prompt=build_all_clusters_prompt(inst,clusters,ctx,use_cot=True,
            use_misfit=False,stats={0:ref['_stats']['0']},confidence_only=True)
        self.assertIn('## Output contract',prompt)
        self.assertIn('sole top-level key must be "confidence"',prompt)
        self.assertNotIn('"reason":',prompt)
        self.assertNotIn('then output JSON',prompt)
        self.assertNotIn('Keep reasoning brief',prompt)
        self.assertNotIn('one short line per stop',prompt)
        contract=prompt.split('## Output contract')[1]
        self.assertNotIn('...',contract)
        self.assertIn('"1": ["0", "1", "2", "3"]',contract)

    def test_acc_confidence_only_prompt(self):
        inst=load_solomon(str(ROOT.parent/'data/Solomon/r104_acc2.txt'))
        clusters=[[1,2,3]]
        ctx=[SoftClusterOntology(inst).get_cluster_context(clusters[0],0)]
        prompt=build_all_clusters_prompt(inst,clusters,ctx,use_cot=True,
            use_misfit=False,confidence_only=True,stats_is_estimate=True)
        self.assertNotIn('Keep reasoning brief',prompt)
        self.assertNotIn('one short line per stop',prompt)
        self.assertIn('Return ONLY one valid JSON object',prompt)

    def test_all_saved_reference_timings(self):
        folder = ROOT/'result_soft/llm_cache_kim500_sonnet46_c1_20260918_113254'
        paths = sorted(folder.rglob('*_reference.json'))
        self.assertEqual(len(paths), 18)
        for path in paths:
            with self.subTest(instance=path.stem):
                data = json.loads(path.read_text(encoding='utf-8'))
                name = path.stem.removesuffix('_reference').lower()
                inst = load_solomon(str(ROOT.parent/f'data/Solomon/{name}.txt'))
                clusters = data['routes']
                fixed = build_reference(clusters, inst)
                replay = _solution_stats(inst, [[0]+ns+[0] for ns in clusters])
                self.assertEqual(sum(len(v['node_detail']) for v in fixed['_stats'].values()), 100)
                for k, vehicle in enumerate(replay['per_vehicle']):
                    stats = fixed['_stats'][str(k)]
                    self.assertEqual(list(fixed[str(k)]), list(map(str, clusters[k])))
                    self.assertAlmostEqual(stats['travel_time'], vehicle['travel_time'], delta=.006)
                    for stop in vehicle['stop_details']:
                        detail = stats['node_detail'][str(stop['node'])]
                        self.assertAlmostEqual(detail['slack'], stop['tw_close']-stop['arrival'], delta=.006)
                        self.assertAlmostEqual(detail['travel_time'], stop['travel_time'], delta=.006)

    def test_reference_keeps_first_and_last_customer(self):
        inst = load_solomon(str(ROOT.parent/'data/Solomon/c102.txt'))
        result = build_reference([[1,2,3],[4]], inst)
        self.assertEqual(set(result['_stats']['0']['node_detail']), {'1','2','3'})
        self.assertEqual(set(result['_stats']['1']['node_detail']), {'4'})

    def test_cot_and_json(self):
        response = 'Explanation.\n```json\n{"confidence":{"1":{"0":0,"5":0.7}}}\n```'
        self.assertEqual(parse_response(response, [[5]])['1']['5'], .7)

    def test_ambiguous_blocks(self):
        with self.assertRaises(ValueError):
            parse_response('```json\n{}\n```\n```json\n{}\n```', [[5]])

    def test_duplicate_inside_block(self):
        with self.assertRaises(ValueError):
            parse_response('```json\n{"confidence":{"1":{"0":0,"5":0.7,"5":0.8}}}\n```', [[5]])

    def test_missing_customer(self):
        with self.assertRaises(AssertionError):
            parse_response('```json\n{"confidence":{"1":{"0":0}}}\n```', [[5]])


if __name__ == '__main__':
    unittest.main()
