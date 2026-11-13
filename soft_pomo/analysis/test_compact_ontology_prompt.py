import copy
import json
from pathlib import Path
import unittest
from compact_ontology_prompt import compact, expand, canonical, build_request

ROOT = Path(__file__).resolve().parents[1]


class CompactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = next((ROOT/'result/rc101_ontology_v5_train20_b128_seed43/requests').iterdir())
        cls.payload = json.loads((folder/'context.json').read_text())
        cls.request = json.loads((folder/'request.json').read_text())

    def test_roundtrip_and_nonmutation(self):
        original = copy.deepcopy(self.payload)
        p, s = compact(self.payload)
        self.assertEqual(canonical(expand(p,s)), canonical(original))
        self.assertEqual(self.payload, original)

    def test_false_is_not_unknown(self):
        p, s = compact(self.payload)
        p['facts']['positive_diagnoses'] = {k:[] for k in p['facts']['positive_diagnoses']}
        decoded = expand(p,s)
        self.assertTrue(all(not any(row[1:]) for row in decoded['facts']['diagnoses']))

    def test_bad_visit_fails(self):
        p = copy.deepcopy(self.payload)
        p['facts']['visits'][0][3] = 'c999'
        with self.assertRaises(ValueError):
            compact(p)

    def test_unknown_leg_fails(self):
        p,s = compact(self.payload)
        p['facts']['positive_diagnoses']['direct_exposure'].append('missing')
        with self.assertRaises(ValueError):
            expand(p,s)

    def test_request_contract_and_numbers(self):
        r,p,s = build_request(self.request,self.payload)
        self.assertEqual(r['response_format'],self.request['response_format'])
        self.assertEqual(r['model'],self.request['model'])
        self.assertEqual(p['facts']['legs'],self.payload['facts']['legs'])
        self.assertEqual(p['facts']['event'],self.payload['facts']['event'])
        self.assertLess(len(r['messages'][0]['content']),len(self.request['messages'][0]['content']))


if __name__ == '__main__':
    unittest.main()
