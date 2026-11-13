$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
& "$root/../.venv/Scripts/python.exe" -c "import analysis.submit_pair_l5 as s; s.NAME='pair_returnmask_cr_v100_200_l5_s42_46_20261031'; s.main()"
if ($LASTEXITCODE -ne 0) { throw 'Submission incomplete; inspect output before retrying.' }
