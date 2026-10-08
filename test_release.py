"""Fast input-contract tests: python -m unittest test_release."""
from pathlib import Path
import tempfile
import unittest

from run import read, sha, validate_cache, validate_resume, verify_bundle

ROOT = Path(__file__).resolve().parent


class ReleaseTests(unittest.TestCase):
    # Fast input/resume contract tests, not validation of training performance or a new LKH solve.
    def test_integrity(self):
        verify_bundle(ROOT)

    def test_split_and_caches(self):
        split = read(ROOT / 'data/split.json')
        names = []
        for part in ('train', 'test'):
            self.assertEqual(len(split[part]), 80)
            self.assertEqual(sum(e['kind'] == 'RAIN' for e in split[part]), 40)
            for row in split[part]:
                name = row['name']
                names.append(name)
                self.assertEqual(sha(ROOT / 'data' / (name + '.txt')), row['sha256'])
                for method, suffix in (('kim', '_cluster.json'), ('phase2', '_rl_cluster.json')):
                    validate_cache(ROOT / 'proposed_model/cache' / method / (name + suffix))
                if row['kind'] == 'ACCIDENT':
                    stages = list((ROOT / 'proposed_model/cache/kim').glob(name + '_acc_refresh_*.json'))
                    self.assertEqual(len(stages), 1)
                    validate_cache(stages[0])
        self.assertEqual(len(set(names)), 160)

    def test_resume_rejects_changed_method(self):
        contract = {'config': {'trainer': {'epochs': 1000}, 'method': 'pure'}}
        validate_resume(contract, {'config': {'trainer': {'epochs': 2000}, 'method': 'pure'}})
        with self.assertRaises(ValueError):
            validate_resume(contract, {'config': {'trainer': {'epochs': 1000}, 'method': 'kim'}})

    def test_manifest_detects_changes(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'input.txt').write_text('original')
            (root / 'bundle_manifest.json').write_text(json.dumps({'sha256': {'input.txt': sha(root / 'input.txt')}}))
            verify_bundle(root)
            (root / 'input.txt').write_text('changed')
            with self.assertRaises(ValueError):
                verify_bundle(root)


if __name__ == '__main__':
    unittest.main()
