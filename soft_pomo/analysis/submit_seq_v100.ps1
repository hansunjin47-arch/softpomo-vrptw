$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$name = 'seq_pure_v100_l5_rc_r_s42_46_20261028'
$archive = Join-Path $root "server_export/$name.tar.gz"
$remote = 'byk_postech@210.125.69.5'
$base = '/scratch/byk_postech/sunjinhan'
$bootstrap = Join-Path $PSScriptRoot 'submit_seq_v100_remote.sh'
if (!(Test-Path -LiteralPath $archive)) { throw 'Build the bundle first' }
$expected = ((Get-Content -LiteralPath "$archive.sha256" -Raw).Trim() -split '\s+')[0]
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expected) { throw 'Local archive hash mismatch' }
scp -P 51014 $archive "$archive.sha256" $bootstrap "${remote}:$base/"
if ($LASTEXITCODE -ne 0) { throw 'Upload failed; no submission attempted' }
ssh -p 51014 $remote "bash $base/submit_seq_v100_remote.sh"
if ($LASTEXITCODE -ne 0) { throw 'Submission incomplete; inspect output before retrying. Do not release held jobs manually.' }
