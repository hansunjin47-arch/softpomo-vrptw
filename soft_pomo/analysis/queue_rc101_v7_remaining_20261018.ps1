param([switch]$ValidateOnly)
$ErrorActionPreference = 'Stop'
$root = 'C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo'
$python = 'C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe'
$previous = Join-Path $root 'result/concurrent_fleet_v7_fresh_train20_20261018'
$output = Join-Path $root 'result/concurrent_fleet_v7_fresh_late1_20_train20_20261018'
$stateFile = Join-Path $root 'result/rc101_v7_remaining_queue_20261018.json'
Set-Location -LiteralPath $root
foreach ($weight in @(1,20)) {
    $config = Get-Content -Raw -LiteralPath (Join-Path $root "result/rc101_learned_first_train20_20261017/late$weight/config.json") | ConvertFrom-Json
    if ($config.trainer.epochs -ne 1000 -or $config.trainer.train_instances.Count -ne 20 -or $config.environment.late_penalty -ne $weight) {
        throw "Unexpected configuration for lambda $weight"
    }
}
if (-not (Test-Path -LiteralPath $python)) { throw 'Python not found' }
if (Test-Path -LiteralPath $output) { throw 'Output already exists; refusing duplicate launch' }
if ($ValidateOnly) { Write-Output 'Validated: RC101 train20 lambda1/20, 1000 epochs, fresh output.'; exit 0 }
function Save-State($stage, $detail) {
    @{stage=$stage; detail=$detail; updated=(Get-Date -Format o); queue=@(5,10,1,20); output=$output} |
        ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -LiteralPath $stateFile
}
try {
    Save-State 'waiting' 'Waiting for existing lambda5/10 process and complete evaluation markers.'
    while (@(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(w)?\.exe$' -and
        $_.CommandLine -like '*experiments.concurrent_fleet_v7.train*' -and
        $_.CommandLine -like '*concurrent_fleet_v7_fresh_train20_20261018*'
    }).Count -gt 0) { Start-Sleep -Seconds 15 }
    $status = Get-Content -Raw -LiteralPath (Join-Path $previous 'status.json') | ConvertFrom-Json
    if ($status.stage -ne 'complete') { throw "Predecessor did not finish successfully: $($status.stage)" }
    if (Test-Path -LiteralPath (Join-Path $previous 'failure.json')) { throw 'Predecessor failure marker exists' }
    foreach ($weight in @(5,10)) {
        $done = Get-Content -Raw -LiteralPath (Join-Path $previous "late$weight/complete.json") | ConvertFrom-Json
        if ($done.epoch -ne 1000 -or $done.test_count -ne 80) { throw "Incomplete predecessor lambda $weight" }
    }
    if (Test-Path -LiteralPath $output) { throw 'Output created while waiting; refusing duplicate launch' }
    Save-State 'running' 'Fresh lambda1 then lambda20; one GPU training process.'
    & $python -u -m experiments.concurrent_fleet_v7.train --out $output --weights 1 20 1> "$output.stdout.log" 2> "$output.stderr.log"
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
    $status = Get-Content -Raw -LiteralPath (Join-Path $output 'status.json') | ConvertFrom-Json
    if ($status.stage -ne 'complete') { throw "Training stopped with status $($status.stage)" }
    Save-State 'complete' 'Lambda1 and lambda20 training and evaluation completed.'
} catch {
    Save-State 'blocked' $_.Exception.Message
    throw
}
