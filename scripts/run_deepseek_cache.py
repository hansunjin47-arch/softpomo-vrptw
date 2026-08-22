import subprocess, sys, os
benchmarks = ['c1', 'rc1', 'r1']
script = os.path.join(r'C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo', 'generate_llm_cache.py')
for bm in benchmarks:
    print(f'=== [{bm}] DeepSeek cache generation start ===', flush=True)
    ret = subprocess.call([r'C:\Users\hansu\anaconda3\envs\gpils\python.exe', '-u', script,
        '--benchmark', bm,
        '--model', 'deepseek-r1:32b',
        '--result-dir', r'C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo\result_soft_deepseek',
    ])
    print(f'=== [{bm}] done (exit={ret}) ===', flush=True)
print('ALL DONE', flush=True)
