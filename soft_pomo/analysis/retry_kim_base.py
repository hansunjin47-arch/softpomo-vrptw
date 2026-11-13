import json
from pathlib import Path
import sys
import time
from generate_kim_initial_cache import save
from kim_cache_retry import regenerate

out=Path(sys.argv[1]).resolve()
manifest=json.loads((out/'manifest.json').read_text())
for name,result in manifest['results'].items():
    if result['success']:
        continue
    family='rc1' if name.startswith('RC') else 'c1' if name.startswith('C') else 'r1'
    folder=out/family
    clusters=json.loads((folder/f'{name}_input.json').read_text())['clusters']
    try:
        conf,history=regenerate((folder/f'{name}_prompt.txt').read_text(encoding='utf-8'),
            clusters,manifest['config'],folder/f'{name}_retry_{time.strftime("%H%M%S")}')
        cache={str(k):{str(n):conf[str(k+1)][str(n)] for n in ns} for k,ns in enumerate(clusters)}
        cache['_depot']={str(k):conf[str(k+1)]['0'] for k in range(len(clusters))}
        save(folder/f'{name}_cluster.json',cache)
        manifest['results'][name]=dict(success=True,initial_failure=result,retry_history=history)
    except Exception as exc:
        manifest['results'][name]=dict(success=False,initial_failure=result,retry_error=str(exc))
    save(out/'manifest.json',manifest)
    print(name,manifest['results'][name]['success'],flush=True)
