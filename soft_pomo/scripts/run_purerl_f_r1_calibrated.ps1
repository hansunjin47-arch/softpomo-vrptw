# Pure RL — Config F, R1 calibrated (D_max=15.0, vehicle_penalty=2.0)
$PYTHON = "C:\Users\hansu\anaconda3\envs\gpils\python.exe"
$SOFT   = "C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo"
$LOGDIR = "$SOFT\logs\purerl_calibrated"
New-Item -ItemType Directory -Force $LOGDIR | Out-Null
function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Write-Host $line
    Add-Content -Path "$LOGDIR\run.log" -Value $line
}
Set-Location $SOFT
Log "START r1  config=F  D_max=15.0"
& $PYTHON -u train_soft_cluster.py `
    --benchmark  r1 `
    --config     F `
    --epochs     1000 `
    --result-dir result_purerl_calibrated `
    --no-llm `
    2>&1 | Tee-Object -Append -FilePath "$LOGDIR\train_r1.log"
if ($LASTEXITCODE -eq 0) { Log "DONE r1" } else { Log "FAILED r1 exit=$LASTEXITCODE" }
