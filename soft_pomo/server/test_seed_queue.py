"""Offline queue tests; never submit a real job."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from seed_queue import next_task, submit, write
from train_cached import read, verify_bundle


class QueueTests(unittest.TestCase):
    def test_train_resume_evaluate_and_advance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cases = [dict(pool=20, seed=s, bundle='train20') for s in (43, 44)]
            plan = dict(cases=cases)
            self.assertEqual(next_task(root, plan), dict(case=cases[0], stage='train', resume=False))
            run = root / 'train20/runs/pure128_4000_seed43'
            run.mkdir(parents=True)
            write(run / 'status.json', dict(epoch=3700, target=4000, status='needs_resume'))
            with self.assertRaises(RuntimeError):
                next_task(root, plan)
            (run / 'checkpoint-last.pt').touch()
            self.assertTrue(next_task(root, plan)['resume'])
            write(run / 'status.json', dict(epoch=4000, target=4000, status='complete'))
            for epoch in (1000, 2000, 3000, 4000):
                task = next_task(root, plan)
                self.assertEqual((task['stage'], task['epoch']), ('eval', epoch))
                out = run / ('eval_%s_test80' % epoch)
                out.mkdir()
                write(out / 'protocol.json', {})
                self.assertTrue(next_task(root, plan)['resume'])
                write(out / 'status.json', dict(status='complete'))
                write(out / 'summary.json', {})
            self.assertEqual(next_task(root, plan)['case'], cases[1])
            self.assertIsNone(next_task(root, dict(cases=cases[:1])))

    def test_submission_dependency(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task = dict(case=dict(pool=20, seed=43), stage='train')
            with patch('seed_queue.subprocess.check_output', return_value='123;cluster\n') as call:
                submit(root, task, '122')
                self.assertIn('--dependency=afterok:122', call.call_args[0][0])
            self.assertEqual(read(root / 'queue_state.json')['job'], '123')

    def test_export_integrity_and_only_pool_differs(self):
        root = Path(__file__).resolve().parents[1] / 'server_export/soft_pomo_seed_sweep'
        plan = read(root / 'plan.json')
        self.assertEqual([(c['pool'], c['seed']) for c in plan['cases']],
                         [(n, s) for n in (20, 40, 80, 200) for s in (43, 44)])
        configs = []
        for n in (20, 40, 80, 200):
            bundle = root / ('train%s' % n)
            verify_bundle(bundle)
            config = read(bundle / 'configs/pure.json')
            self.assertEqual(len(config['trainer'].pop('train_instances')), n)
            configs.append(config)
        self.assertTrue(all(c == configs[0] for c in configs))


if __name__ == '__main__':
    unittest.main()
