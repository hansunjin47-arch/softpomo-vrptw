# Proposed model (Kim cluster + LLM confidence bias) — init BKS few-shot, Config E, R1
#
# Purpose: clean A/B against pure-RL Config E (result_purerl_e_calibrated/r102-r112).
#   - 4 bias-path bugs fixed (step_max normalisation, acc cache collision, cluster shuffle)
#   - LLM called ONCE at init only  (--experience-refresh-epoch 999999)
#   - All LLM caches pre-supplied  => zero LLM calls during this run
#
# Cache: cache_bks_ranking_fixed/  (copied from result_soft_bks_ranking, ACC files
#        renamed *_cluster.json -> *_acc_refresh.json to match the fixed code paths)

$PYTHON    = "C:\Users\hansu\anaconda3\envs\gpils\python.exe"
$SOFT      = "C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo"
$RESULT    = "result_llm_e_bksfewshot"
$CACHE_DIR = "$SOFT\cache_bks_ranking_fixed"
$LOGDIR    = "$SOFT\logs\$RESULT"

New-Item -ItemType Directory -Force $LOGDIR | Out-Null

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Write-Host $line
    Add-Content -Path "$LOGDIR\run.log" -Value $line
}

Set-Location $SOFT
Log "START  benchmark=r1  config=E  result-dir=$RESULT  cache=$CACHE_DIR"

& $PYTHON -u train_soft_cluster.py `
    --benchmark                r1 `
    --config                   E `
    --epochs                   1000 `
    --result-dir               $RESULT `
    --llm-cache-dir            $CACHE_DIR `
    --experience-refresh-epoch 999999 `
    2>&1 | Tee-Object -Append -FilePath "$LOGDIR\train_r1.log"

if ($LASTEXITCODE -eq 0) { Log "DONE r1" } else { Log "FAILED r1 exit=$LASTEXITCODE" }
