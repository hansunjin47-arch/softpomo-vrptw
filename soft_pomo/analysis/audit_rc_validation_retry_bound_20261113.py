"""Offline HTTP mock: interrupted/failing requests cannot reissue durable slots."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from experiments.concurrent_fleet_bias_guided.llm_retry import RetryingClient
from experiments.concurrent_fleet_pair.io import write
from analysis.evaluate_rc_kim_final100_validation_20261113 import OUT

def main():
    count=0
    def fake_http(*args,**kwargs):
        nonlocal count
        count+=1
        raise TimeoutError('Offline simulated interrupted HTTP')
    request={'model':'offline-test','messages':[]}
    per_restart=[]
    with tempfile.TemporaryDirectory(prefix='retry-bound-') as tmp:
        with patch.dict(os.environ,{'OPENROUTER_API_KEY':'offline-placeholder'}),patch('urllib.request.urlopen',fake_http),patch('time.sleep',lambda _:None):
            for restart in range(3):
                client=RetryingClient(Path(tmp),lambda req,payload:None,allow_paid=True)
                try:client(request)
                except RuntimeError:pass
                per_restart.append(count)
        assert per_restart==[3,3,3]
        assert len(list(Path(tmp).rglob('attempted')))==3
    write(OUT/'retry_bound_audit.json',dict(passed=True,mocked_HTTP_counts_after_restart=per_restart,
        actual_HTTP_calls=0,
        mechanism='OpenRouterClient exclusively creates attempt_N/request_hash/attempted before urlopen. RetryingClient has exactly three fixed attempt directories; restart never removes markers.',
        bound='Durable ledger at most50 hashes times3 fixed exclusive slots implies at most150 HTTP attempts across restarts, provided output/call files are not deleted or moved.',
        caveat='An interrupted attempt without response consumes its slot permanently; evaluator fails closed after three slots. No automatic cache deletion or reset. Single evaluator process only.'))
    print('Offline restart test PASS: mocked HTTP totals 3,3,3; actual HTTP=0')

if __name__=='__main__':main()
