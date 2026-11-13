$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
& "$root/../.venv/Scripts/python.exe" -m analysis.fetch_rc101_mask_kim
if ($LASTEXITCODE -ne 0) { throw 'Download or verification failed; inspect output.' }
