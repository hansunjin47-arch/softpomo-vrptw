"""Add RAIN2 and cumulative ACC2 waves to a completed isolated Kim BASE run."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

from generate_kim_initial_cache import save, ROOT
from run_c_fewshot_pilot import strict_response
from SoftClusterLLMModule import build_all_clusters_prompt
from SoftClusterOntology import SoftClusterOntology
from vrptw_env import load_solomon
from kim_cache_retry import regenerate


def main():
    out = Path(sys.argv[1]).resolve()
    base_manifest = json.loads((out/'manifest.json').read_text())
    manifest = dict(config=base_manifest['config'], results={}, scenarios=['RAIN2','ACC2'],
                    fewshot=False, misfit=False, cluster_source='kim',
                    wave_policy='BASE membership; cumulative scores; affected clusters only; no future events',
                    code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for name in base_manifest['names']:
        family = 'rc1' if name.startswith('RC') else 'c1' if name.startswith('C') else 'r1'
        folder = out/family
        if not (folder/f'{name}_cluster.json').exists():
            manifest['results'][name]=dict(success=False,error='BASE cache unavailable; events skipped')
            save(out/'events_manifest.json',manifest)
            continue
        base = json.loads((folder/f'{name}_cluster.json').read_text())
        clusters = [list(map(int, base[str(k)])) for k in range(len(base['_depot']))]
        for scenario in ['RAIN2','ACC2']:
            source = ROOT.parent/'data/Solomon'/f'{name.lower()}_{scenario.lower()}.txt'
            inst = load_solomon(str(source))
            current = copy.deepcopy(base)
            waves = ([None] if scenario == 'RAIN2' else
                     sorted(set(e['trigger_time'] for e in inst['preset_events'] if e['type']=='ACCIDENT')))
            assert scenario != 'ACC2' or len(waves) == 3
            for trigger in waves:
                start = time.time()
                visible = None if trigger is None else trigger/float(inst['T'])
                stem = (f'{name}_RAIN2' if trigger is None else
                        f'{name}_ACC2_acc_refresh_{visible:.6f}')
                cachefile = folder/(stem+'_cluster.json' if trigger is None else stem+'.json')
                assert not cachefile.exists(), 'Refusing to overwrite an event cache'
                nodes = {n for e in inst['preset_events'] if e['trigger_time']==trigger
                         for n in e['nodes']} if trigger is not None else set()
                indices = list(range(len(clusters))) if trigger is None else [
                    k for k, ns in enumerate(clusters) if nodes.intersection(ns)]
                subset = [clusters[k] for k in indices]
                try:
                    ont = SoftClusterOntology(inst)
                    contexts = [ont.get_cluster_context(ns,k,cur_time=visible or 0)
                                for k,ns in enumerate(subset)]
                    prior = {str(j+1): dict(current[str(k)], **{'0':current['_depot'][str(k)]})
                             for j,k in enumerate(indices)}
                    experience = ('Prior confidence before this scenario/wave; reconsider for visible events. '
                        'No rollout-specific visited nodes or route timings are known.\n'+json.dumps(prior))
                    prompt = build_all_clusters_prompt(inst, subset, contexts,
                        use_cot=True,use_ontology=True,reward_config='E',use_misfit=False,
                        cluster_source='kim',events_visible_until=visible,experience_section=experience)
                    (folder/f'{stem}_prompt.txt').write_text(prompt,encoding='utf-8')
                    save(folder/f'{stem}_input.json',dict(original_cluster_indices=indices,
                        clusters=subset,prior=prior,events_visible_until=visible,
                        input_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
                        prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()))
                    print('CALL',stem,'clusters',len(subset),flush=True)
                    conf,history=regenerate(prompt,subset,manifest['config'],folder/stem)
                    for j,k in enumerate(indices):
                        current[str(k)]={str(n):conf[str(j+1)][str(n)] for n in clusters[k]}
                        current['_depot'][str(k)]=conf[str(j+1)]['0']
                    save(cachefile,current)
                    result=dict(success=True,seconds=time.time()-start,attempts=history)
                except Exception as exc:
                    result=dict(success=False,error=f'{type(exc).__name__}: {exc}',seconds=time.time()-start)
                manifest['results'][stem]=result
                save(out/'events_manifest.json',manifest)
                print(stem,json.dumps(result),flush=True)
                if not result['success']:
                    break


if __name__=='__main__':
    main()
