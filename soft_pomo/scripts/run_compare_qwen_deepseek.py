"""
run_compare_qwen_deepseek.py
캐시 생성 완료 대기 → Qwen vs DeepSeek zero-shot 평가 자동 실행

Qwen  : result_soft/llm_cache/
DeepSeek: result_soft_deepseek/llm_cache/
동일 Config F checkpoint, 3 benchmarks (c1, rc1, r1)
"""
import os, subprocess, sys, time

PYTHON = sys.executable
HERE   = os.path.dirname(os.path.abspath(__file__))
ORIG   = os.path.join(HERE, '..', 'original_POMO')
EVAL   = os.path.join(HERE, 'eval_soft_plug.py')
LOG    = os.path.join(HERE, 'logs', 'deepseek_cache_run4.log')

CKPTS = {
    'c1':  os.path.join(ORIG, 'result', 'config_F_pomo100',     'c102-c109',  'checkpoint-last.pt'),
    'rc1': os.path.join(ORIG, 'result', 'config_F_rc1_pomo100', 'rc102-rc108','checkpoint-last.pt'),
    'r1':  os.path.join(ORIG, 'result', 'config_F_r1_pomo100',  'r102-r112',  'checkpoint-last.pt'),
}

RUNS = [
    # (tag,        result_dir,              model)
    # Qwen plug_F_* results already exist in result_soft/ — skip re-run
    ('deepseek','result_soft_deepseek',  'deepseek-r1:32b'),
]

def wait_for_cache():
    print('[AutoRun] Waiting for DeepSeek cache generation to complete...', flush=True)
    while True:
        if os.path.isfile(LOG):
            with open(LOG, encoding='utf-8', errors='replace') as f:
                if 'ALL DONE' in f.read():
                    print('[AutoRun] Cache generation complete!', flush=True)
                    return
        time.sleep(30)

def run_eval(benchmark, tag, result_dir, model):
    ckpt = CKPTS[benchmark]
    out_dir = os.path.join(HERE, f'result_compare_{tag}_{benchmark}')
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(HERE, 'logs', f'compare_{tag}_{benchmark}.log')

    cmd = [
        PYTHON, '-u', EVAL,
        '--benchmark', benchmark,
        '--config', 'F',
        '--resume', ckpt,
        '--model', model,
        '--result-dir', os.path.join(HERE, result_dir),
    ]
    print(f'[AutoRun] {tag.upper()} {benchmark.upper()} start...', flush=True)
    with open(log_path, 'w', encoding='utf-8') as lf:
        ret = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
    status = 'OK' if ret == 0 else f'FAILED(exit={ret})'
    print(f'[AutoRun] {tag.upper()} {benchmark.upper()} {status} → {log_path}', flush=True)
    return ret

def main():
    wait_for_cache()

    print('\n[AutoRun] === Starting Qwen vs DeepSeek evaluation ===\n', flush=True)
    results = []
    for benchmark in ['c1', 'rc1', 'r1']:
        for tag, result_dir, model in RUNS:
            ret = run_eval(benchmark, tag, result_dir, model)
            results.append((benchmark, tag, ret))

    print('\n[AutoRun] === All done ===', flush=True)
    for bm, tag, ret in results:
        status = 'OK' if ret == 0 else f'FAILED({ret})'
        print(f'  {tag:10s} {bm:5s} {status}', flush=True)

if __name__ == '__main__':
    main()
