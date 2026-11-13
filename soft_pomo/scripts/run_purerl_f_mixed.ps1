# Pure RL baseline — Config F, benchmark=mixed (C1+RC1+R1)
# No LLM  |  Epochs: 1000

$PYTHON = "C:\Users\hansu\anaconda3\envs\gpils\python.exe"
$SOFT   = "C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo"
$RESULT = "result_purerl_mixed"
$LOGDIR = "$SOFT\logs\$RESULT"

New-Item -ItemType Directory -Force $LOGDIR | Out-Null

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Write-Host $line
    Add-Content -Path "$LOGDIR\run.log" -Value $line
}

Set-Location $SOFT
Log "START  benchmark=mixed  config=F  result-dir=$RESULT"

& $PYTHON train_soft_cluster.py `
    --benchmark  mixed `
    --config     F `
    --epochs     1000 `
    --result-dir $RESULT `
    --no-llm `
    2>&1 | Tee-Object -Append -FilePath "$LOGDIR\train_mixed.log"

if ($LASTEXITCODE -eq 0) {
    Log "DONE"
} else {
    Log "FAILED exit=$LASTEXITCODE"
}
