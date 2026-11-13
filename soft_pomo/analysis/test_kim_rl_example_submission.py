"""Exercise submission without contacting Slurm or creating real jobs."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = Path.home() / 'AppData/Local/Programs/Git/bin/bash.exe'


class SubmissionTest(unittest.TestCase):
    def test_two_chains_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            shutil.copy2(ROOT / 'analysis/submit_kim_rl_example_lanes.sh', folder / 'submit.sh')
            (folder / 'tasks.txt').write_text(''.join(f'case/seed{i}\n' for i in range(15)))
            env = dict(os.environ, MOCK_DIR=folder.as_posix(), USER='mock_user')
            script = r'''
sbatch() {
    index="${@: -1}"
    printf '%s\n' "$*" >> "$MOCK_DIR/calls.txt"
    echo "$((9000+index))"
}
squeue() { printf '123 sj_kim_l5_all\n456 sj_kim_l5_tail\n789 sikim_other_person\n'; }
scontrol() { printf '%s\n' "$*" >> "$MOCK_DIR/releases.txt"; }
export -f sbatch squeue scontrol
bash "$MOCK_DIR/submit.sh"
'''
            for _ in range(2):
                result = subprocess.run([str(BASH), '-c', script], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = (folder / 'calls.txt').read_text().splitlines()
            self.assertEqual(len(calls), 15)
            for i, call in enumerate(calls):
                self.assertIn('--hold', call)
                dependency = '123:456' if i < 2 else str(9000+i-2)
                self.assertIn('--dependency=afterany:'+dependency, call)
                self.assertNotIn('789', call)
            self.assertEqual(len((folder / 'releases.txt').read_text().splitlines()), 15)
            self.assertTrue((folder / 'submission_released.ok').exists())


if __name__ == '__main__':
    unittest.main()
