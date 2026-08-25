# Pure RL on TW-augmented C1/RC1 pools (base instances only, no rain/acc), Config E
#
#   C1  train : c102-c109 (8, Solomon) + CH01-03/CM01-03/CE01-03 (9, generated)  = 17
#   C1  test  : c101
#   RC1 train : rc102-rc108 (7, Solomon) + RCH01-05/RCM01-05/RCE01-05 (15, generated) = 22
#   RC1 test  : rc101
#
# Mirrors the R1 twaug experiment: does covering the test instance's TW regime
# (missing from the Solomon-only training pool) fix the base Lc failure?

$PYTHON = "C:\Users\hansu\anaconda3\envs\gpils\python.exe"
$SOFT   = "C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo"
$LOGDIR = "$SOFT\logs\twaug_c_rc"

New-Item -ItemType Directory -Force $LOGDIR | Out-Null

$C1_SOLOMON  = 102..109 | ForEach-Object { "c$_" }
$C1_GEN      = @('CH01','CH02','CH03','CM01','CM02','CM03','CE01','CE02','CE03')
$C1_TRAIN    = $C1_SOLOMON + $C1_GEN

$RC1_SOLOMON = 102..108 | ForEach-Object { "rc$_" }
$RC1_GEN     = @('RCH01','RCH02','RCH03','RCH04','RCH05',
                 'RCM01','RCM02','RCM03','RCM04','RCM05',
                 'RCE01','RCE02','RCE03','RCE04','RCE05')
$RC1_TRAIN   = $RC1_SOLOMON + $RC1_GEN

Write-Host "C1  train pool : $($C1_TRAIN.Count) instances  ($($C1_SOLOMON.Count) Solomon + $($C1_GEN.Count) generated)  |  test: c101"
Write-Host "RC1 train pool : $($RC1_TRAIN.Count) instances  ($($RC1_SOLOMON.Count) Solomon + $($RC1_GEN.Count) generated)  |  test: rc101"
Write-Host ""

$jobs = @(
    @{ bench = 'c1';  train = $C1_TRAIN;  test = 'c101'  },
    @{ bench = 'rc1'; train = $RC1_TRAIN; test = 'rc101' }
)

foreach ($j in $jobs) {
    $argList = @("-u", "$SOFT\train_vrptw.py",
                 "--benchmark", $j.bench,
                 "--config", "E",
                 "--epochs", "1000",
                 "--pomo", "100",
                 "--tag", "twaug",
                 "--test-instances", $j.test,
                 "--train-instances") + $j.train

    $proc = Start-Process -FilePath $PYTHON -ArgumentList $argList `
        -WorkingDirectory $SOFT `
        -RedirectStandardOutput "$LOGDIR\$($j.bench)_E.log" `
        -RedirectStandardError  "$LOGDIR\$($j.bench)_E.err" `
        -NoNewWindow -PassThru

    "$(Get-Date -Format 'HH:mm:ss') [LAUNCH] benchmark=$($j.bench) config=E PID=$($proc.Id)" |
        Tee-Object "$LOGDIR\launcher.log" -Append
}

Write-Host "`nLogs: $LOGDIR\c1_E.log , rc1_E.log"
