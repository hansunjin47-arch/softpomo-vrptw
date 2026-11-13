param(
    [Parameter(Mandatory=$true)][string]$Previous,
    [Parameter(Mandatory=$true)][string]$Next,
    [Parameter(Mandatory=$true)][string]$Python
)
$ErrorActionPreference = 'Stop'
function Set-QueueState([string]$State) {
    @{status=$State; previous=$Previous; next=$Next; time=(Get-Date).ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath "$Next\queue_status.json" -Encoding UTF8
}
try {
    Set-QueueState 'waiting_for_A_training_and_evaluation'
    while ($true) {
        if (Test-Path -LiteralPath "$Previous\failure.json") {
            throw 'Config A reported failure; Config C will not start automatically.'
        }
        $status = Get-Content -LiteralPath "$Previous\status.json" -Raw | ConvertFrom-Json
        if ($status.status -eq 'complete') { break }
        $active = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
            Where-Object { $_.CommandLine -and $_.CommandLine.Contains("$Previous\runner.py") })
        if ($active.Count -eq 0) { throw 'Config A process stopped before completion.' }
        Start-Sleep -Seconds 30
    }
    $active = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -and $_.CommandLine.Contains("$Previous\runner.py") })
    foreach ($process in $active) {
        Wait-Process -Id $process.ProcessId -ErrorAction SilentlyContinue
    }
    Set-QueueState 'checking_config_C'
    & $Python -u "$Next\runner.py" smoke --run $Next
    if ($LASTEXITCODE -ne 0) { throw 'Config C smoke test failed.' }
    Set-QueueState 'running_config_C'
    & $Python -u "$Next\runner.py" train --run $Next
    if ($LASTEXITCODE -ne 0) { throw 'Config C training or evaluation failed.' }
    Set-QueueState 'complete'
} catch {
    Set-QueueState 'failed'
    Write-Error $_
    exit 1
}
