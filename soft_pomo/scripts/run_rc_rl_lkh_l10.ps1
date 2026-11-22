param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
& $Python -m experiments.concurrent_fleet_pair_kim.rl_lkh validate --late 10
if ($LASTEXITCODE -ne 0) { throw 'Input validation failed.' }
if (-not $env:OPENROUTER_API_KEY) {
    $secret = Read-Host 'OpenRouter API key (hidden)' -AsSecureString
    $env:OPENROUTER_API_KEY = ([PSCredential]::new('unused', $secret)).GetNetworkCredential().Password
}
& $Python -u -m experiments.concurrent_fleet_pair_kim.rl_lkh prepare --late 10 --allow-api
if ($LASTEXITCODE -ne 0) { throw 'Cache generation failed. Completed responses are retained; inspect error before retrying.' }
foreach ($seed in 44,45,46) {
    & $Python -m experiments.concurrent_fleet_pair_kim.rl_lkh train --late 10 --seed $seed --validate-only
    if ($LASTEXITCODE -ne 0) { throw "Seed $seed preflight failed." }
}
foreach ($seed in 44,45,46) {
    $out = Join-Path $root "result/rc_rl_lkh_v100_l10_s$seed"
    $marker = Join-Path $out 'run/late10/training_complete.json'
    if (Test-Path -LiteralPath $marker) {
        $done = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
        if ($done.epoch -eq 1000) { Write-Host "Seed $seed already complete"; continue }
    }
    $resumeArgs = @()
    if (Test-Path -LiteralPath (Join-Path $out 'run/late10/checkpoint-last.pt')) { $resumeArgs = @('--resume') }
    Write-Host "Training RC101 RL+LKH lambda10 seed $seed"
    & $Python -u -m experiments.concurrent_fleet_pair_kim.rl_lkh train --late 10 --seed $seed @resumeArgs
    if ($LASTEXITCODE -ne 0) { throw "Seed $seed stopped; later seeds were not started." }
    if (-not (Test-Path -LiteralPath $marker)) { throw "Seed $seed did not finish 1000 epochs; queue stopped." }
}
Write-Host 'All three training runs complete. Causal Test80 evaluation is still required.'
