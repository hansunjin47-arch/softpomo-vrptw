"""Clone the frozen late15 experiment with only the late weight changed."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'server_export/rc101_seq_train20_late15_seeds42_46_20261022'
OUT = ROOT / 'server_export/rc101_seq_train20_late10_seeds42_46_20261023'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--late', type=int, choices=[5, 10, 20], default=10)
    args = parser.parse_args()
    late = args.late
    out = OUT.with_name(f'rc101_seq_train20_late{late}_seeds42_46_20261023')
    out.mkdir(exist_ok=False)
    for seed in range(42, 47):
        src = SOURCE / f'seed{seed}'
        dst = out / f'seed{seed}'
        manifest = json.loads((src / 'bundle_manifest.json').read_text())
        for name, expected in manifest['sha256'].items():
            assert digest(src / name) == expected, name
            target = dst / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / name, target)
        cfg = json.loads((dst / 'configs/pure.json').read_text())
        assert cfg['trainer']['seed'] == seed
        assert cfg['trainer']['epochs'] == 1000
        assert cfg['environment']['late_penalty'] == 15
        cfg['environment']['late_penalty'] = late
        write(dst / 'configs/pure.json', cfg)
        provenance = json.loads((dst / 'provenance.json').read_text())
        provenance['objective'] = f'K/14 + (travel_minutes + {late}*late_minutes)/1696.95'
        provenance['source'] = SOURCE.name + '; frozen sources and split unchanged'
        write(dst / 'provenance.json', provenance)
        write(dst / 'bundle_manifest.json', {'sha256': {
            p.relative_to(dst).as_posix(): digest(p)
            for p in dst.rglob('*') if p.is_file()
        }})
    shutil.copy2(SOURCE / 'eval_pure_tt_bundle.py', out / 'eval_pure_tt_bundle.py')
    slurm = (SOURCE / 'seq_late15_seeds.slurm').read_text()
    slurm = slurm.replace('sj_rc_seq_l15', f'sj_rc_seq_l{late}').replace('--array=0-4%1', '--array=0-4%2')
    (out / f'seq_late{late}_seeds.slurm').write_text(slurm, encoding='ascii', newline='\n')
    archive = out.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(out, arcname=out.name)
    Path(str(archive) + '.sha256').write_text(
        digest(archive) + '  ' + archive.name + '\n', encoding='ascii', newline='\n')
    print(archive)


if __name__ == '__main__':
    main()
