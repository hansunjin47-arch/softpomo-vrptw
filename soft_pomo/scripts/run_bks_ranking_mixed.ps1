# BKS ranking + Config F — benchmark=mixed (C1+RC1+R1)
# Step 1: cache-only (LLM scoring for all mixed instances)
# Step 2: training 1000 epochs with init zeroshot

$PYTHON = "C:\Users\hansu\anaconda3\envs\gpils\python.exe"
$SOFT   = "C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo"
$RESULT = "result_soft_bks_ranking_mixed"
$LOGDIR = "$SOFT\logs\$RESULT"

New-Item -ItemType Directory -Force $LOGDIR | Out-Null

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Write-Host $line
    Add-Content -Path "$LOGDIR\run.log" -Value $line
}

Set-Location $SOFT
Log "START cache-only  benchmark=mixed  config=F"

& $PYTHON train_soft_cluster.py `
    --benchmark  mixed `
    --config     F `
    --result-dir $RESULT `
    --cache-only `
    2>&1 | Tee-Object -Append -FilePath "$LOGDIR\cache.log"

if ($LASTEXITCODE -ne 0) {
    Log "FAILED cache step exit=$LASTEXITCODE"
    exit 1
}

Log "Cache done — starting training"

& $PYTHON train_soft_cluster.py `
    --benchmark                mixed `
    --config                   F `
    --epochs                   1000 `
    --result-dir               $RESULT `
    --experience-refresh-epoch 999999 `
    2>&1 | Tee-Object -Append -FilePath "$LOGDIR\train_mixed.log"

if ($LASTEXITCODE -eq 0) {
    Log "DONE"
} else {
    Log "FAILED train step exit=$LASTEXITCODE"
}
