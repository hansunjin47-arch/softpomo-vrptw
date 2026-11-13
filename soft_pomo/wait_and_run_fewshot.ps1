# wait_and_run_fewshot.ps1
# Waits for zero-shot free-starts training to complete, then launches few-shot free-starts.

$PYTHON  = "C:\Users\hansu\anaconda3\envs\gpils\python.exe"
$SOFT    = "C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo"
$QUEUE   = Join-Path $SOFT "run_free_fewshot_queue.py"
$LOG     = Join-Path $SOFT "logs\free_fewshot\wait.log"

New-Item -ItemType Directory -Force -Path (Split-Path $LOG) | Out-Null

function Write-Log($msg) {
    $line = "$(Get-Date -Format 'HH:mm:ss') $msg"
    Write-Host $line
    Add-Content -Path $LOG -Value $line -Encoding UTF8
}

# PIDs of the running zero-shot free-starts processes
$watchPids = @(22264, 35464, 37432)

Write-Log "=== Waiting for zero-shot free-starts (PIDs: $($watchPids -join ', ')) ==="

while ($true) {
    $alive = $watchPids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }
    if ($alive.Count -eq 0) {
        Write-Log "All zero-shot processes done."
        break
    }
    Write-Log "Still running: $($alive -join ', ') ($($alive.Count)/3)"
    Start-Sleep -Seconds 120
}

# Verify checkpoint-last.pt saved
foreach ($bm in @("c102-c109","rc102-rc108","r102-r112")) {
    $ckpt = Join-Path $SOFT "result_soft_free\$bm\checkpoint-last.pt"
    if (Test-Path $ckpt) {
        Write-Log "OK: $bm checkpoint-last.pt found"
    } else {
        Write-Log "WARN: $bm checkpoint-last.pt NOT found — starting anyway"
    }
}

Write-Log "Launching few-shot free-starts queue..."
& $PYTHON -u $QUEUE 2>&1 | Tee-Object -FilePath (Join-Path $SOFT "logs\free_fewshot\queue_stdout.log") -Append
Write-Log "Queue script exited."
