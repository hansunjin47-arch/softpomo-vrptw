"""Audit completed Kim BASE/RAIN2/ACC2 files, including cumulative wave invariants."""
import json
import math
from pathlib import Path
import sys
from generate_kim_initial_cache import save

out=Path(sys.argv[1]).resolve()
manifest=json.loads((out/'manifest.json').read_text())
errors=[]
counts={'BASE':0,'RAIN2':0,'ACC2':0}
for name in manifest['names']:
    family='rc1' if name.startswith('RC') else 'c1' if name.startswith('C') else 'r1'
    folder=out/family
    base_path=folder/f'{name}_cluster.json'
    try:
        base=json.loads(base_path.read_text())
        groups={k:list(v) for k,v in base.items() if k.isdigit()}
        flat=[n for ns in groups.values() for n in ns]
        assert len(flat)==100 and set(flat)==set(map(str,range(1,101)))
        paths=[('BASE',base_path),('RAIN2',folder/f'{name}_RAIN2_cluster.json')]
        waves=sorted(folder.glob(f'{name}_ACC2_acc_refresh_*.json'))
        waves=[p for p in waves if p.stem.rsplit('_',1)[-1].replace('.','',1).isdigit()]
        assert len(waves)==3, f'Expected 3 ACC2 waves, found {len(waves)}'
        paths.extend(('ACC2',p) for p in waves)
        previous=base
        for scenario,path in paths:
            data=json.loads(path.read_text())
            assert {k:list(v) for k,v in data.items() if k.isdigit()}==groups,'membership/order'
            assert set(data['_depot'])==set(groups),'depot coverage'
            values=[v for k in groups for v in data[k].values()]+list(data['_depot'].values())
            assert all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in values)
            if scenario=='ACC2':
                inputs=json.loads(path.with_name(path.stem+'_input.json').read_text())
                changed=set(map(str,inputs['original_cluster_indices']))
                for k in set(groups)-changed:
                    assert data[k]==previous[k] and data['_depot'][k]==previous['_depot'][k], 'untouched cluster changed'
                for j,k in enumerate(inputs['original_cluster_indices']):
                    assert inputs['prior'][str(j+1)]==dict(previous[str(k)],**{'0':previous['_depot'][str(k)]}), 'prior not cumulative'
                previous=data
            counts[scenario]+=1
    except Exception as exc:
        errors.append(dict(instance=name,error=f'{type(exc).__name__}: {exc}'))
report=dict(counts=counts,expected={'BASE':29,'RAIN2':29,'ACC2':87},errors=errors)
save(out/'verification.json',report)
print(json.dumps(report,indent=2))
sys.exit(bool(errors))
