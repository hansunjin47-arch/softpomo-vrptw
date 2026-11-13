$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
& "$root/../.venv/Scripts/python.exe" -m analysis.backup_and_stop_rc_kim
if ($LASTEXITCODE -ne 0) { throw 'Backup/stop incomplete. Inspect output and server_export/rc_kim_stop_receipt_*.json before retrying.' }
