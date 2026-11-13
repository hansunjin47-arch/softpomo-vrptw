"""Exercise the exported runner's extension control flow without GPU training."""
import copy
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    shutil.copytree(ROOT / 'server_export/soft_pomo_rl_train80', root, dirs_exist_ok=True)
    config = json.loads((root / 'configs/pure.json').read_text())
    previous = dict(bundle_sha256=hashlib.sha256((root / 'bundle_manifest.json').read_bytes()).hexdigest(), config=config)
    out = root / 'runs/test'
    out.mkdir(parents=True)
    (out / 'settings.json').write_text(json.dumps(previous))
    (out / 'checkpoint-last.pt').touch()
    subprocess.run([sys.executable, str(ROOT / 'server/patch_extend.py')], cwd=root, check=True)
    namespace = runpy.run_path(str(root / 'server/train_cached.py'))
    main = namespace['main']
    saved = []
    class Trainer:
        device = type('Device', (), {'type': 'cpu'})()
        reward_log = []
        loss_log = []
        def _train_one_epoch(self, epoch):
            return -1.0, 0.0
    def restore(trainer, checkpoint, contract):
        assert contract == previous
        return 3999
    replacements = dict(build_trainer=lambda *a: Trainer(), restore=restore,
                        save_checkpoint=lambda trainer, epoch, path, contract: saved.append((epoch, path.name, contract)))
    with patch.dict(main.__globals__, replacements), patch.object(sys, 'argv',
            ['runner', '--method', 'pure', '--out', str(out), '--resume', '--epochs', '4000']):
        main()
    assert any(epoch == 4000 and name == 'checkpoint-4000.pt' for epoch, name, _ in saved)
    assert saved[-1][2]['config']['trainer']['epochs'] == 4000
    assert previous['config']['trainer']['epochs'] == 1000
    # A new checkpoint/settings pair must still reject changing the batch size.
    changed = copy.deepcopy(config)
    changed['trainer']['train_batch_size'] = 128
    (root / 'configs/pure.json').write_text(json.dumps(changed))
    with patch.dict(main.__globals__, dict(replacements, verify_bundle=lambda root: saved[-1][2]['bundle_sha256'])), patch.object(
            sys, 'argv', ['runner', '--method', 'pure', '--out', str(out), '--resume', '--epochs', '5000']):
        try:
            main()
        except ValueError as exc:
            assert 'Only an unchanged or increased' in str(exc)
        else:
            raise AssertionError('Changed batch size accepted')
    print('PASS: legacy contract restored, extended checkpoint saved, changed batch rejected')
