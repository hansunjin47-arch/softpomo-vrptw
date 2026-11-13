$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$name = 'kim_l5_rc_r_c_s42_46_20261026'
$archive = Join-Path $root "server_export/$name.tar.gz"
$remote = 'byk_postech@210.125.69.5'
$base = '/scratch/byk_postech/sunjinhan'
if (!(Test-Path -LiteralPath $archive)) { throw 'Build the bundle first' }
$bootstrap = Join-Path $PSScriptRoot 'submit_kim_l5_remote.sh'
scp -P 51014 $archive "$archive.sha256" $bootstrap "${remote}:$base/"
if ($LASTEXITCODE -ne 0) { throw 'Upload failed; no submission attempted' }
ssh -p 51014 $remote "bash $base/submit_kim_l5_remote.sh"
if ($LASTEXITCODE -ne 0) { throw 'Remote submission failed; inspect output before retrying' }
