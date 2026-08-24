# Kim clustering + DeepSeek init zeroshot with BKS few-shot + ranking output
# Benchmark: r1  |  Epochs: 1000  |  Config: F
# LLM called once at init (BKS few-shot injected); cache fixed for all 1000 epochs.

$ErrorActionPreference = "Continue"
$SOFT   = "C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo"
$PYTHON = "C:\Users\hansu\anaconda3\envs\gpils\python.exe"
$RESULT = "result_soft_bks_ranking"
$LOG    = "$SOFT\logs\$RESULT"

New-Item -ItemType Directory -Force $LOG | Out-Null

$out = "$LOG\r1_run.log"
$err = "$LOG\r1_err.log"

"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [START] r1  result-dir=$RESULT" |
    Tee-Object "$LOG\launcher.log" -Append

$proc = Start-Process `
    -FilePath $PYTHON `
    -ArgumentList "`"$SOFT\train_soft_cluster.py`"",
                  "--benchmark",               "r1",
                  "--config",                  "F",
                  "--epochs",                  "1000",
                  "--result-dir",              $RESULT,
                  "--experience-refresh-epoch","999999" `
    -WorkingDirectory $SOFT `
    -RedirectStandardOutput $out `
    -RedirectStandardError  $err `
    -NoNewWindow -PassThru

"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [LAUNCHED] PID=$($proc.Id)" |
    Tee-Object "$LOG\launcher.log" -Append

Write-Host "Log: $out"
Write-Host "Waiting for completion..."
$proc.WaitForExit()

"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [DONE] exit=$($proc.ExitCode)" |
    Tee-Object "$LOG\launcher.log" -Append

Write-Host "Exit code: $($proc.ExitCode)"
