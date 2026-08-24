# Pure RL — Config F, benchmark-calibrated reward (R1 then RC1)
# R1:  D_max=15.0, vehicle_penalty=2.0  (vs C1: D_max=3.0, vp=3.0)
# RC1: D_max=12.0, vehicle_penalty=2.0
# Epochs: 1000 each

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

# ── R1 ────────────────────────────────────────────────────────────────────────
Log "START  benchmark=r1  config=F  D_max=15.0 (calibrated)"

& $PYTHON train_soft_cluster.py `
    --benchmark  r1 `
    --config     F `
    --epochs     1000 `
    --result-dir result_purerl_calibrated `
    --no-llm `
    2>&1 | Tee-Object -Append -FilePath "$LOGDIR\train_r1.log"

if ($LASTEXITCODE -ne 0) {
    Log "FAILED r1 exit=$LASTEXITCODE"
    exit 1
}
Log "DONE r1"

# ── RC1 ───────────────────────────────────────────────────────────────────────
Log "START  benchmark=rc1  config=F  D_max=12.0 (calibrated)"

& $PYTHON train_soft_cluster.py `
    --benchmark  rc1 `
    --config     F `
    --epochs     1000 `
    --result-dir result_purerl_calibrated `
    --no-llm `
    2>&1 | Tee-Object -Append -FilePath "$LOGDIR\train_rc1.log"

if ($LASTEXITCODE -eq 0) {
    Log "DONE rc1"
} else {
    Log "FAILED rc1 exit=$LASTEXITCODE"
}
