param(
    [ValidatePattern('^kim_incremental_[0-9]{8}_[0-9]{6}\.tar\.gz$')]
    [string]$ResumeArchive
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$server = 'byk_postech@210.125.69.5'
$remote = '/scratch/byk_postech/sunjinhan'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$name = "kim_incremental_$stamp.tar.gz"
if ($ResumeArchive) {
    $name = $ResumeArchive
    $stamp = $name.Substring('kim_incremental_'.Length, 15)
}
$receipt = 'server_export/kim_received_runs.json'
if (-not (Test-Path -LiteralPath $receipt)) {
    Set-Content -LiteralPath $receipt -Value '[]' -Encoding UTF8
}
if (-not $ResumeArchive) {
    $remoteReceipt = "kim_received_$stamp.json"
    $exporter = "export_kim_$stamp.py"
    scp -P 51014 $receipt "${server}:$remote/$remoteReceipt"
    if ($LASTEXITCODE -ne 0) { throw 'Receipt upload failed' }
    scp -P 51014 'analysis/export_seq_incremental.py' "${server}:$remote/$exporter"
    if ($LASTEXITCODE -ne 0) { throw 'Exporter upload failed' }
    ssh -p 51014 $server "python3 $remote/$exporter --method kim --root $remote --received $remote/$remoteReceipt --out $remote/$name"
    if ($LASTEXITCODE -ne 0) { throw 'Export failed' }
}
Write-Host "Downloading/verifying existing export: $name"
$archive = Join-Path $root "server_export/$name"
scp -P 51014 "${server}:$remote/$name.sha256" "$archive.sha256"
if ($LASTEXITCODE -ne 0) { throw "Checksum download failed. Retry with -ResumeArchive $name; inspect the scp error above." }
$expected = ((Get-Content -LiteralPath "$archive.sha256" -Raw).Trim() -split '\s+')[0]
if ($expected -notmatch '^[0-9a-fA-F]{64}$') { throw 'Invalid SHA256 file' }
$validLocal = (Test-Path -LiteralPath $archive) -and ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -eq $expected)
if (-not $validLocal) {
    scp -P 51014 "${server}:$remote/$name" $archive
    if ($LASTEXITCODE -ne 0) { throw "Archive download failed. Retry with -ResumeArchive $name; inspect the scp error above." }
} else {
    Write-Host 'Local archive checksum matches; skipping duplicate archive download.'
}
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expected) { throw 'SHA256 mismatch' }
$dest = Join-Path $root "server_export/kim_completed_received_$stamp"
if (-not (Test-Path -LiteralPath $dest)) { New-Item -ItemType Directory -Path $dest | Out-Null }
tar -xzf $archive -C $dest
if ($LASTEXITCODE -ne 0) { throw 'Extraction failed' }
& "$root/../.venv/Scripts/python.exe" 'analysis/validate_kim_received.py' --input $dest --receipt $receipt
if ($LASTEXITCODE -ne 0) { throw 'Validation failed; receipt not updated' }
Write-Host "Verified results: $dest"
