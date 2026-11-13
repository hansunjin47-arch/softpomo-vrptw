$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
& "$root/../.venv/Scripts/python.exe" -c "from analysis import submit_pair_l5 as s; s.NAME='rc101_pair100_200_l5_mask_kim_s42_46_20261111'; s.main()"
if ($LASTEXITCODE -ne 0) { throw 'Submission incomplete; inspect output. Do not release held jobs manually.' }
