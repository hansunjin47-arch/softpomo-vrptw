$ErrorActionPreference = 'Stop'
$remote = 'byk_postech@210.125.69.5'
$dest = '/scratch/byk_postech/sunjinhan/repair_kim_two_lanes.sh'
scp -P 51014 (Join-Path $PSScriptRoot 'repair_kim_two_lanes.sh') "${remote}:$dest"
if ($LASTEXITCODE -ne 0) { throw 'Repair upload failed' }
ssh -p 51014 $remote "bash $dest 231538"
if ($LASTEXITCODE -ne 0) { throw 'Repair incomplete. Do not release held jobs manually; inspect output.' }
