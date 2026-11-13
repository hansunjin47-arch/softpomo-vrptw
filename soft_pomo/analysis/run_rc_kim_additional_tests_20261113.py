"""Evaluate newly completed server runs sequentially on one GPU."""
import subprocess
import sys

if __name__ == '__main__':
    for weight in (100, 200):
        subprocess.run([sys.executable, '-u', '-m',
                        f'analysis.evaluate_rc_kim_v{weight}_additional_20261113'], check=True)
