# Pure RL baseline — Config G (late_count_penalty=3.0, late_penalty=9.0)
# Benchmark: r1  |  Epochs: 1000  |  No LLM

$PYTHON = "C:\Users\hansu\anaconda3\envs\gpils\python.exe"
$SOFT   = "C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo"
$RESULT = "result_purerl_g"
$LOGDIR = "$SOFT\logs\$RESULT"

New-Item -ItemType Directory -Force $LOGDIR | Out-Null

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Write-Host $line
    Add-Content -Path "$LOGDIR\run.log" -Value $line
}

Set-Location $SOFT
Log "START  benchmark=r1  config=G  result-dir=$RESULT"

& $PYTHON train_soft_cluster.py `
    --benchmark  r1 `
    --config     G `
    --epochs     1000 `
    --result-dir $RESULT `
    --no-llm `
    2>&1 | Tee-Object -Append -FilePath "$LOGDIR\train_r1.log"

if ($LASTEXITCODE -eq 0) {
    Log "DONE"
} else {
    Log "FAILED exit=$LASTEXITCODE"
}
