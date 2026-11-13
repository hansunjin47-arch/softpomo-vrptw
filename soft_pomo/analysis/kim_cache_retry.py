"""Same-model full regeneration only; never fill, clamp or fall back."""
import json
import time
import urllib.request
from generate_kim_initial_cache import save
from run_c_fewshot_pilot import strict_response


def regenerate(prompt, clusters, config, prefix, attempts=3):
    history = []
    error = None
    for attempt in range(1, attempts+1):
        request_prompt = prompt
        if attempt > 1:
            request_prompt += ('\nValidation failed on the previous response: '+str(error)+
                '\nRegenerate the ENTIRE response, not a patch. Return one JSON object ONLY, '
                'with no prose or code fences. Include exactly these keys per cluster '
                '(including depot 0), each once. The top-level object MUST have '
                'a "confidence" key mapping cluster IDs to node-score objects, and a '
                '"reason" string. Do not return arrays of IDs as the answer. Required IDs:\n'
                +json.dumps({str(k+1):[0]+ns for k,ns in enumerate(clusters)}))
        path = prefix.parent/(prefix.name+f'_attempt{attempt}')
        path.with_suffix('.prompt.txt').write_text(request_prompt,encoding='utf-8')
        start=time.time()
        print('ATTEMPT',prefix.name,attempt,flush=True)
        try:
            req=urllib.request.Request('http://localhost:11434/api/generate',
                data=json.dumps(dict(config,prompt=request_prompt)).encode(),
                headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=900) as response:
                raw=json.load(response)
            save(path.with_suffix('.raw.json'),raw)
            assert raw.get('done') and raw.get('done_reason')=='stop','incomplete generation'
            conf=strict_response(raw['response'],clusters)
            history.append(dict(attempt=attempt,success=True,seconds=time.time()-start))
            save(prefix.parent/(prefix.name+'_attempts.json'),history)
            return conf,history
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
            history.append(dict(attempt=attempt,success=False,error=error,seconds=time.time()-start))
            save(prefix.parent/(prefix.name+'_attempts.json'),history)
    raise RuntimeError(f'All {attempts} attempts failed: {error}')
