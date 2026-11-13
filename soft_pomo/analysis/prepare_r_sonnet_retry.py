"""Preserve the failed run and stage successful artifacts for a versioned retry."""
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'result_soft/llm_cache_kim500_sonnet46_r1_v2_20260918_175218'
    target=ROOT/'result_soft'/('llm_cache_kim500_sonnet46_r1_v3_'+time.strftime('%Y%m%d_%H%M%S'))
    manifest=json.loads((source/'manifest.json').read_text())
    failed=[k for k,v in manifest['results'].items() if not v.get('success')]
    target.mkdir(exist_ok=False)
    for path in source.rglob('*'):
        if not path.is_file():
            continue
        relative=path.relative_to(source)
        if relative.parts[0]=='reference_run' or path.name.startswith(('generation_', 'phase2_')):
            continue
        if any(path.name.startswith(stem+'_') for stem in failed):
            continue
        dest=target/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
    manifest['results']={k:v for k,v in manifest['results'].items() if v.get('success')}
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (target/'retry_provenance.json').write_text(json.dumps(dict(source=str(source),
        retained_successes=list(manifest['results']),retry_cases=failed,
        prompt_change='Replace abbreviated output example with complete per-vehicle required key inventory',
        mixed_prompt_versions=True, failed_attempts_preserved_at_source=True),indent=2),encoding='utf-8')
    print(target)


if __name__=='__main__':
    main()
