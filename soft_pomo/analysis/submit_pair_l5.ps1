$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
& "$root/../.venv/Scripts/python.exe" 'analysis/submit_pair_l5.py'
if ($LASTEXITCODE -ne 0) { throw 'Submission incomplete; inspect output. Do not release held jobs manually.' }
