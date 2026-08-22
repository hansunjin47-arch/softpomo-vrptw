$pyPid  = 22020
$psPid  = 33364
$py     = "C:\Users\hansu\anaconda3\envs\gpils\python.exe"
$work   = "C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo"
$ckpt   = "result_soft_rl_deepseek\r102-r112\checkpoint-last.pt"
$log    = "$work\logs\rl_recluster_r1_phase2.log"

Write-Host "[Watcher] Waiting for R1 Phase1 (PID $pyPid) to finish..."

while (Get-Process -Id $pyPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 10
}

Write-Host "[Watcher] R1 Phase1 done. Killing PowerShell PID $psPid..."
Stop-Process -Id $psPid -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

# RC1이 이미 시작됐으면 kill
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)").CommandLine
    if ($cmd -match "benchmark rc1") {
        Write-Host "[Watcher] Killing RC1 Python PID $($_.Id)"
        Stop-Process -Id $_.Id -Force
    }
}

Write-Host "[Watcher] Starting R1 Phase2..."
Set-Location $work
& $py train_soft_cluster.py `
    --benchmark r1 `
    --config F `
    --result-dir result_soft_rl_deepseek `
    --model deepseek-r1:32b `
    --resume $ckpt `
    --epochs 1000 `
    --experience-refresh-epoch 999999 `
    *>> $log

Write-Host "[Watcher] R1 Phase2 complete."
