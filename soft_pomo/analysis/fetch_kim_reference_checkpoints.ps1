$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$server = 'byk_postech@210.125.69.5'
$bundle = 'cr101_seq_l5_10_15_20_s42_46_20261023'
$remoteRoot = '/scratch/byk_postech/sunjinhan'
foreach ($case in @('r101', 'c101')) {
    $relative = "$bundle/$case/late5/seed42/runs/pure/checkpoint-500.pt"
    $destination = Join-Path $root "server_export/$relative"
    $remote = "$remoteRoot/$relative"
    $hashOutput = & ssh -p 51014 $server "sha256sum '$remote'"
    if ($LASTEXITCODE -ne 0) { throw "Cannot verify remote checkpoint: $case" }
    $expected = (($hashOutput -join "`n").Trim() -split '\s+')[0]
    if ($expected -notmatch '^[a-fA-F0-9]{64}$') { throw "Invalid remote hash: $case" }
    if (Test-Path -LiteralPath $destination) {
        if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -ne $expected) {
            throw "Existing checkpoint differs; refusing to overwrite: $destination"
        }
        Write-Host "Already verified: $case"
        continue
    }
    New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force | Out-Null
    $partial = "$destination.download"
    & scp -P 51014 "${server}:$remote" $partial
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $case" }
    if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $expected) {
        throw "Checkpoint hash mismatch: $case"
    }
    Move-Item -LiteralPath $partial -Destination $destination
    Write-Host "Downloaded and verified: $case"
}
