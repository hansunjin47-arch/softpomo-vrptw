"""One-time in-place patch for the original exported training runner."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path.cwd()
p = root / 'server/train_cached.py'
m = root / 'bundle_manifest.json'
old = p.read_text(encoding='utf-8')
manifest = json.loads(m.read_text())
digest = hashlib.sha256(m.read_bytes()).hexdigest()
assert hashlib.sha256(p.read_bytes()).hexdigest() == manifest['sha256']['server/train_cached.py']
assert not p.with_suffix('.py.before_extend').exists(), 'Already patched or backup exists'
anchor = "    contract = dict(bundle_sha256=digest, config=config)\n"
insert = '''    previous = contract
    if args.resume:
        previous = read(out / 'settings.json')
        if previous['bundle_sha256'] not in (digest, OLD_DIGEST):
            raise ValueError('Unexpected original bundle')
        before, after = copy.deepcopy(previous['config']), copy.deepcopy(config)
        old_target = before['trainer'].pop('epochs')
        new_target = after['trainer'].pop('epochs')
        if before != after or new_target < old_target:
            raise ValueError('Only an unchanged or increased epoch target is allowed')
'''.replace('OLD_DIGEST', repr(digest))
assert old.count(anchor) == 1
assert old.count('epoch = restore(trainer, checkpoint, contract)') == 1
assert old.count('if epoch in (500, 1000):') == 1
new = old.replace(anchor, anchor + insert).replace(
    'epoch = restore(trainer, checkpoint, contract)', 'epoch = restore(trainer, checkpoint, previous)').replace(
    'if epoch in (500, 1000):', 'if epoch == 500 or epoch % 1000 == 0:')
compile(new, str(p), 'exec')
shutil.copy2(p, p.with_suffix('.py.before_extend'))
shutil.copy2(m, m.with_suffix('.json.before_extend'))
p.write_text(new, encoding='utf-8', newline='\n')
manifest['sha256']['server/train_cached.py'] = hashlib.sha256(p.read_bytes()).hexdigest()
m.write_text(json.dumps(manifest, indent=2))
evaluation = root / 'eval_addon/manifest.json'
if evaluation.exists():
    info = json.loads(evaluation.read_text())
    info['required_sources']['server/train_cached.py'] = manifest['sha256']['server/train_cached.py']
    shutil.copy2(evaluation, evaluation.with_suffix('.json.before_extend'))
    evaluation.write_text(json.dumps(info, indent=2))
print('Patched: epoch extension allowed; 1000-epoch snapshots enabled. Checkpoints unchanged.')
