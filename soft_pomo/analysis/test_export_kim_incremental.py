import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


class ExportTest(unittest.TestCase):
    def test_completed_only_and_receipt_for_both_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = 'kim_l5_rc_r_c_20261026/rc101/seed42'
            new = 'kim_rl_example_l5_rc_r_c_20261027/rc101/seed42'
            pending = 'kim_rl_example_l5_rc_r_c_20261027/rc101/seed43'
            names = [dict(name=f'case{i}') for i in range(80)]
            for relative in (old, new, pending):
                data = {
                    'split.json': {'test': names},
                    'configs/kim.json': {},
                    'runs/kim/status.json': {'status': 'complete', 'epoch': 1000},
                    'eval_1000_test80_gpu/results.json': names,
                    'eval_1000_test80_gpu/status.json': {
                        'stage': 'running' if relative == pending else 'complete',
                        'epoch': 1000, 'completed': 80},
                }
                for name, value in data.items():
                    path = root/relative/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(value), encoding='utf-8')
            receipt = root/'receipt.json'
            receipt.write_text(json.dumps([old]), encoding='utf-8')
            archive = root/'export.tar.gz'
            subprocess.run([sys.executable, str(Path(__file__).with_name('export_seq_incremental.py')),
                            '--root', str(root), '--received', str(receipt), '--out', str(archive),
                            '--method', 'kim'], check=True, capture_output=True)
            with tarfile.open(archive) as stream:
                members = stream.getnames()
            self.assertTrue(members)
            self.assertTrue(all(name.startswith(new+'/') for name in members))


if __name__ == '__main__':
    unittest.main()
