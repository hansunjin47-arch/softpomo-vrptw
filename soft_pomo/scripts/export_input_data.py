"""Add local datasets and frozen experiment inputs to a code snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    assert (out/'.git').is_dir(), 'Expected an existing isolated snapshot repository'
    pairs = []
    for source_root in (ROOT.parent/'data', ROOT/'data'):
        for p in source_root.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                pairs.append((p, p.relative_to(ROOT.parent)))
    for name in ('rc101_pair100_200_l5_mask_kim_s42_46_20261111',
                 'r101_pair100_l5_mask_kim_s42_46_20261106'):
        bundle = ROOT/'server_export'/name
        base = Path('soft_pomo/datasets')/name
        for p in (bundle/'dataset').rglob('*'):
            if p.is_file():
                pairs.append((p, base/'dataset'/p.relative_to(bundle/'dataset')))
        for p in (bundle/'artifacts').rglob('config.json'):
            pairs.append((p, base/'configs'/p.relative_to(bundle/'artifacts')))
    hashes = {}
    for source, relative in pairs:
        content = source.read_bytes()
        if source.suffix in {'.json', '.txt', '.py'}:
            text = content.decode('utf-8-sig')
            if re.search(r'sk-or-v1-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}', text):
                raise ValueError('Possible credential: ' + str(relative))
            if source.suffix == '.json':
                json.loads(text)
        target = out/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[relative.as_posix()] = hashlib.sha256(content).hexdigest()
    (out/'input_data_manifest.json').write_text(json.dumps(dict(
        scope='Local datasets plus frozen RC/R dataset splits and training configs; no LLM caches',
        sha256=hashes), indent=2), encoding='utf-8')
    print('Copied and hashed', len(hashes), 'input files')


if __name__ == '__main__':
    main()
