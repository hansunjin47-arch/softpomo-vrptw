# BKS ranking + Config G — reuse existing LLM cache from Config F run
# Benchmark: r1  |  Epochs: 1000  |  --experience-refresh-epoch 999999 (init zeroshot)
# LLM cache: result_soft_bks_ranking/llm_cache (no regeneration needed)

$PYTHON    = "C:\Users\hansu\anaconda3\envs\gpils\python.exe"
$SOFT      = "C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo"
$RESULT    = "result_soft_bks_ranking_g"
$CACHE_DIR = "$SOFT\result_soft_bks_ranking\llm_cache"
$LOGDIR    = "$SOFT\logs\$RESULT"

New-Item -ItemType Directory -Force $LOGDIR | Out-Null

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Write-Host $line
    Add-Content -Path "$LOGDIR\run.log" -Value $line
}

Set-Location $SOFT
Log "START  benchmark=r1  config=G  result-dir=$RESULT  cache=$CACHE_DIR"

& $PYTHON train_soft_cluster.py `
    --benchmark                r1 `
    --config                   G `
    --epochs                   1000 `
    --result-dir               $RESULT `
    --llm-cache-dir            $CACHE_DIR `
    --experience-refresh-epoch 999999 `
    2>&1 | Tee-Object -Append -FilePath "$LOGDIR\train_r1.log"

if ($LASTEXITCODE -eq 0) {
    Log "DONE"
} else {
    Log "FAILED exit=$LASTEXITCODE"
}
