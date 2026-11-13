"""
run_deepseek_zeroshot.py
DeepSeek zero-shot training: C1 → RC1 → R1
기존 Qwen zero-shot(result_soft/)과 동일 조건, LLM만 DeepSeek으로 교체.
캐시는 result_soft_deepseek/llm_cache/ 사용 (사전 생성 완료).
experience-refresh 없음 (순수 zero-shot).
"""
import subprocess, sys, os, time

PYTHON = sys.executable
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_soft_cluster.py')
RESULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result_soft_deepseek')
LOG    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG, exist_ok=True)

QUEUE = ['c1', 'rc1', 'r1']

for bm in QUEUE:
    log_path = os.path.join(LOG, f'deepseek_zeroshot_{bm}.log')
    cmd = [
        PYTHON, '-u', SCRIPT,
        '--benchmark', bm,
        '--result-dir', RESULT,
        '--model', 'deepseek-r1:32b',
        '--experience-refresh-epoch', '999999',  # no refresh = pure zero-shot
        '--config', 'F',
    ]
    print(f'[{bm.upper()}] DeepSeek zero-shot training start...', flush=True)
    t0 = time.time()
    with open(log_path, 'w', encoding='utf-8') as lf:
        ret = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = (time.time() - t0) / 60
    status = 'OK' if ret == 0 else f'FAILED({ret})'
    print(f'[{bm.upper()}] {status}  {elapsed:.1f}min  → {log_path}', flush=True)

print('ALL DONE', flush=True)
