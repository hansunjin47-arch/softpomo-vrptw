param([switch]$Diagnostics, [switch]$Mask)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$fetchArgs = @('analysis/fetch_pair_completed.py')
if ($Diagnostics) { $fetchArgs += '--diagnostics' }
if ($Mask) { $fetchArgs += '--mask' }
& "$root/../.venv/Scripts/python.exe" @fetchArgs
if ($LASTEXITCODE -ne 0) { throw 'Concurrent result download or validation failed; inspect output.' }
