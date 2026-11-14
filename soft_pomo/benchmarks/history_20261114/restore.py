"""List or restore historical experiment files to a separate destination."""
import argparse, gzip, json, hashlib, zipfile
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--match',default='',help='Substring in original path, e.g. rc_pair_returnmask')
p.add_argument('--out',type=Path,help='Separate destination; omit to list only')
p.add_argument('--verify',action='store_true',help='Verify all archive SHA256 values')
a=p.parse_args();base=Path(__file__).resolve().parent
manifest=json.loads((base/'manifest.json').read_text())
assert hashlib.sha256((base/'index.json.gz').read_bytes()).hexdigest()==manifest['index_sha256']
if a.verify:
    for name,digest in manifest['archives'].items():
        assert hashlib.sha256((base/name).read_bytes()).hexdigest()==digest,name
    print('All archive hashes verified')
with gzip.open(base/'index.json.gz','rt',encoding='utf-8') as f:index=json.load(f)
rows=[r for r in index['entries'] if a.match.lower() in r['path'].lower()]
if a.out:
    target=a.out.resolve();target.mkdir(parents=True,exist_ok=True)
    handles={}
    try:
        for r in rows:
            dest=(target/r['path']).resolve()
            if not dest.is_relative_to(target):raise ValueError('Unsafe path')
            if dest.exists():
                if hashlib.sha256(dest.read_bytes()).hexdigest()==r['sha256']:continue
                raise FileExistsError(str(dest)+' differs; choose a fresh output directory')
            if r['archive'] not in handles:handles[r['archive']]=zipfile.ZipFile(base/r['archive'])
            data=handles[r['archive']].read(r['sha256'])
            assert hashlib.sha256(data).hexdigest()==r['sha256']
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    finally:
        for z in handles.values():z.close()
else:
    for r in rows:print(r['path'])
print(str(len(rows))+' matching files')
