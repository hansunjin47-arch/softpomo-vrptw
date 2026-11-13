"""Mock Slurm to check the two-chain limit without submitting any jobs."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = Path.home()/'AppData/Local/Programs/Git/bin/bash.exe'


class SubmissionTests(unittest.TestCase):
    def test_two_lanes_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            shutil.copy2(ROOT/'analysis/submit_seq_v100_lanes.sh', folder/'submit.sh')
            (folder/'tasks.txt').write_text(''.join(f'case/seed{i}\n' for i in range(10)))
            env = dict(os.environ, MOCK_DIR=folder.as_posix(), USER='mock_user')
            script = r'''
sbatch() {
    index="${@: -1}"
    printf '%s\n' "$*" >> "$MOCK_DIR/calls.txt"
    echo "$((9000+index))"
}
squeue() { printf '232570 sj_kim_rl_example\n232571 sj_kim_rl_example\n789 sikim_other_person\n'; }
scontrol() { printf '%s\n' "$*" >> "$MOCK_DIR/releases.txt"; }
export -f sbatch squeue scontrol
bash "$MOCK_DIR/submit.sh"
'''
            for _ in range(2):
                proc = subprocess.run([str(BASH),'-c',script],env=env,capture_output=True,text=True)
                self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
            calls=(folder/'calls.txt').read_text().splitlines()
            self.assertEqual(len(calls),10)
            for index,call in enumerate(calls):
                dep='232570:232571' if index<2 else str(9000+index-2)
                self.assertIn('--dependency=afterany:'+dep,call)
                self.assertIn('--hold',call)
                self.assertNotIn('789',call)
            self.assertEqual(len((folder/'releases.txt').read_text().splitlines()),10)


if __name__=='__main__':
    unittest.main()
