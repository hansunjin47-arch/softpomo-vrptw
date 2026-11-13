# RC1 Config F 완료 감지 → Config G (R1 + RC1) 동시 실행
$SOFT    = "C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo"
$SCRIPTS = "$SOFT\scripts"
$LOGDIR  = "$SOFT\logs\purerl_g_calibrated"
$RC1F_LOG = "$SOFT\logs\purerl_calibrated\train_rc1.log"
$RC1F_PID = 22980

New-Item -ItemType Directory -Force $LOGDIR | Out-Null

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Write-Host $line
    Add-Content -Path "$LOGDIR\watcher.log" -Value $line
}

Log "Watcher started. Waiting for RC1-F (PID=$RC1F_PID) to finish..."

# PID가 없으면 이미 종료된 것
$proc = Get-Process -Id $RC1F_PID -ErrorAction SilentlyContinue
if ($proc) {
    Wait-Process -Id $RC1F_PID -ErrorAction SilentlyContinue
}

Log "RC1-F training finished. Launching Config G for R1 and RC1 simultaneously..."

$ps = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"

Start-Process $ps -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$SCRIPTS\run_purerl_g_r1_calibrated.ps1`""
Start-Process $ps -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$SCRIPTS\run_purerl_g_rc1_calibrated.ps1`""

Log "Both Config G jobs launched. R1 log: $LOGDIR\train_r1.log  RC1 log: $LOGDIR\train_rc1.log"
