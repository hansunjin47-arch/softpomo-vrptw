$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$server = 'byk_postech@210.125.69.5'
$bundle = 'rc101_seq_train20_late5_seeds42_46_20261023'
$remote = "/scratch/byk_postech/sunjinhan/$bundle"
$hashLines = @(ssh -p 51014 $server "cd $remote && sha256sum seed42/runs/pure/checkpoint-500.pt seed43/runs/pure/checkpoint-500.pt seed44/runs/pure/checkpoint-500.pt seed45/runs/pure/checkpoint-500.pt seed46/runs/pure/checkpoint-500.pt")
if ($LASTEXITCODE -ne 0) { throw 'Remote checkpoint verification failed' }
$hashes = @{}
foreach ($line in $hashLines) {
    if ($line -match '^([a-fA-F0-9]{64})\s+\*?(seed\d+/runs/pure/checkpoint-500.pt)$') {
        $hashes[$Matches[2]] = $Matches[1]
    }
}
if ($hashes.Count -ne 5) { throw 'Expected five remote SHA256 hashes' }
foreach ($seed in 42..46) {
    $relative = "seed$seed/runs/pure/checkpoint-500.pt"
    $dest = Join-Path $root "server_export/$bundle/$relative"
    New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
    if ((Test-Path -LiteralPath $dest) -and ((Get-FileHash -LiteralPath $dest -Algorithm SHA256).Hash -eq $hashes[$relative])) {
        Write-Host "Already verified: seed$seed"
        continue
    }
    scp -P 51014 "${server}:$remote/$relative" "$dest.download"
    if ($LASTEXITCODE -ne 0) { throw "Download failed: seed$seed" }
    if ((Get-FileHash -LiteralPath "$dest.download" -Algorithm SHA256).Hash -ne $hashes[$relative]) {
        throw "Checksum mismatch: seed$seed"
    }
    Move-Item -LiteralPath "$dest.download" -Destination $dest -Force
    Write-Host "Verified: seed$seed"
}
