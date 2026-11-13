param(
    [ValidatePattern('^[a-zA-Z0-9_]+$')]
    [string]$BundlePrefix,
    [switch]$IncludeKim
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$server = 'byk_postech@210.125.69.5'
$remote = '/scratch/byk_postech/sunjinhan'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$name = "seq_incremental_$stamp.tar.gz"
$receipt = "seq_received_$stamp.json"
$exporter = "export_seq_$stamp.py"
$transferArgs = @('analysis/transfer_seq_single_session.py', '--stamp', $stamp)
if ($BundlePrefix) { $transferArgs += @('--bundle-prefix', $BundlePrefix) }
if ($IncludeKim) { $transferArgs += '--include-kim' }
& "$root/../.venv/Scripts/python.exe" @transferArgs
if ($LASTEXITCODE -ne 0) { throw 'Transfer failed; receipt not updated' }
$archive = Join-Path $root "server_export/$name"
$expected = ((Get-Content -LiteralPath "$archive.sha256" -Raw).Trim() -split '\s+')[0]
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expected) { throw 'SHA256 mismatch' }
$dest = Join-Path $root "server_export/seq_completed_received_$stamp"
New-Item -ItemType Directory -Path $dest | Out-Null
tar -xzf $archive -C $dest
if ($LASTEXITCODE -ne 0) { throw 'Extraction failed' }
& "$root/../.venv/Scripts/python.exe" 'analysis/summarize_seq_received_20261024.py' --input $dest --output "analysis/seq_received_summary_$stamp.json"
if ($LASTEXITCODE -ne 0) { throw 'Validation failed; receipt was not updated' }
Write-Host "Verified new results: $dest"
if ($IncludeKim) {
    $kimArchive = Join-Path $root "server_export/kim_incremental_$stamp.tar.gz"
    $expected = ((Get-Content -LiteralPath "$kimArchive.sha256" -Raw).Trim() -split '\s+')[0]
    if ((Get-FileHash -LiteralPath $kimArchive -Algorithm SHA256).Hash -ne $expected) { throw 'Kim SHA256 mismatch' }
    $kimDest = Join-Path $root "server_export/kim_completed_received_$stamp"
    New-Item -ItemType Directory -Path $kimDest | Out-Null
    tar -xzf $kimArchive -C $kimDest
    if ($LASTEXITCODE -ne 0) { throw 'Kim extraction failed' }
    & "$root/../.venv/Scripts/python.exe" 'analysis/validate_kim_received.py' --input $kimDest --receipt 'server_export/kim_received_runs.json'
    if ($LASTEXITCODE -ne 0) { throw 'Kim validation failed' }
    Write-Host "Verified Kim results: $kimDest"
}
