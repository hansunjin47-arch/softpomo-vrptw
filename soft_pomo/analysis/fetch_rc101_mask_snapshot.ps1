$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
& "$root/../.venv/Scripts/python.exe" -m analysis.fetch_rc101_mask_snapshot
if ($LASTEXITCODE -ne 0) { throw 'Snapshot download failed; inspect output. Server jobs were not stopped.' }
