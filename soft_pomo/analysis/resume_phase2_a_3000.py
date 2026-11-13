"""Resume A Phase2 after C evaluation; preserve the original 1000-epoch run."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'rc101_phase2_train20_b128_s42_config'
SUFFIX = '_ontology_action_semantics_ac_v1'
RUN = ROOT / 'result' / (PREFIX + 'A' + SUFFIX)
OUT = RUN / 'continuation_3000'
DEPENDENCY = ROOT / 'result' / (PREFIX + 'C' + SUFFIX) / 'eval_1000_test80_gpu/status.json'


def main():
    bundle = RUN / 'training_bundle'
    sys.path.insert(0, str(bundle / 'server'))
    import train_cached as io
    io.verify_bundle(bundle)
    checkpoint = RUN / 'training/checkpoint-1000.pt'
    saved = io.read(RUN / 'training/settings.json')
    import torch
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['epoch'] == 1000 and state['server_contract'] == saved
    for key in ('optimizer_state_dict', 'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng', 'advantage_state'):
        assert key in state, key
    assert saved['bundle_sha256'] == io.sha(bundle / 'bundle_manifest.json')
    del state
    if (OUT / 'settings.json').exists():
        raise FileExistsError('Continuation already initialized; inspect before resuming')
    io.write(OUT / 'queue_status.json', dict(stage='waiting_for_C_evaluation', dependency=str(DEPENDENCY), target=3000))
    io.write(OUT / 'provenance.json', dict(source_checkpoint=str(checkpoint), source_sha256=io.sha(checkpoint),
        restore='model, optimizer, Python/NumPy/Torch/CUDA RNG and available scaler/generator state',
        start_epoch=1001, target_epoch=3000, additional_api_calls=False))
    deadline = time.monotonic() + 24*3600
    while not DEPENDENCY.exists() or io.read(DEPENDENCY).get('stage') != 'complete':
        if time.monotonic() > deadline:
            raise TimeoutError('C evaluation did not complete within 24 hours')
        time.sleep(30)
    assert io.read(DEPENDENCY)['completed'] == 80
    time.sleep(10)
    shutil.copy2(checkpoint, OUT / 'checkpoint-last.pt')
    io.write(OUT / 'settings.json', saved)
    io.write(OUT / 'queue_status.json', dict(stage='training', target=3000))
    command = [sys.executable, '-u', str(bundle / 'server/train_cached.py'), '--method', 'phase2',
        '--seed', '42', '--out', str(OUT), '--resume', '--epochs', '3000', '--max-hours', '11.5']
    subprocess.run(command, cwd=bundle, check=True)
    io.write(OUT / 'queue_status.json', io.read(OUT / 'status.json'))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'failure.json').write_text(json.dumps(dict(error=str(exc))), encoding='utf-8')
        raise
