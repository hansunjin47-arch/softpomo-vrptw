# Pure RL on TW-augmented R1 pool  (base instances only, no rain/acc)
#
#   train : r102-r112 (Solomon base, TW width 30-148)
#         + RH/RM/RE 01-10 (generated, TW width 8-70, R1 geometry byte-identical)
#   test  : r101 (TW width 10 — inside the training range for the first time)
#
# Purpose: does covering r101's TW regime fix the base failure (Lc=31)?
# Config E (vp=3.0, Lt>K>D) and F (vp=2.0, Lt>D>K) run in parallel to see
# which controls the vehicle count better.

$PYTHON = "C:\Users\hansu\anaconda3\envs\gpils\python.exe"
$ORIG   = "C:\Users\hansu\PycharmProjects\PythonProject\original_POMO"
$LOGDIR = "$ORIG\logs\twaug_r1"

New-Item -ItemType Directory -Force $LOGDIR | Out-Null

$SOLOMON = 102..112 | ForEach-Object { "r$_" }
$GEN     = @()
foreach ($t in 'H', 'M', 'E') { 1..10 | ForEach-Object { $GEN += ("R{0}{1:00}" -f $t, $_) } }
$TRAIN   = $SOLOMON + $GEN

Write-Host "train pool : $($TRAIN.Count) instances  ($($SOLOMON.Count) Solomon + $($GEN.Count) generated)"
Write-Host "test       : r101"
Write-Host ""

foreach ($cfg in 'E', 'F') {
    $argList = @("-u", "$ORIG\train_vrptw.py",
                 "--benchmark", "r1",
                 "--config", $cfg,
                 "--epochs", "1000",
                 "--pomo", "100",
                 "--tag", "twaug",
                 "--test-instances", "r101",
                 "--train-instances") + $TRAIN

    $proc = Start-Process -FilePath $PYTHON -ArgumentList $argList `
        -WorkingDirectory $ORIG `
        -RedirectStandardOutput "$LOGDIR\config_$cfg.log" `
        -RedirectStandardError  "$LOGDIR\config_$cfg.err" `
        -NoNewWindow -PassThru

    "$(Get-Date -Format 'HH:mm:ss') [LAUNCH] config=$cfg PID=$($proc.Id)" |
        Tee-Object "$LOGDIR\launcher.log" -Append
}

Write-Host "`nLogs: $LOGDIR\config_E.log , config_F.log"
