# R1 benchmark LLM cache generation
# Kim clustering + DeepSeek-R1:32b + BKS few-shot (C102 Route2, RC102 Route6, R102 Route1)
# ranking output format  |  Config: F  |  --cache-only (no training)

$PYTHON   = "C:\Users\hansu\anaconda3\envs\gpils\python.exe"
$SOFT     = "C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo"
$RESULT   = "result_soft_bks_ranking"
$LOG      = "$SOFT\logs\$RESULT\cache_r1.log"

New-Item -ItemType Directory -Force "$SOFT\logs\$RESULT" | Out-Null

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Write-Host $line
    Add-Content -Path $LOG -Value $line
}

Log "START  benchmark=r1  result-dir=$RESULT"
Set-Location $SOFT

& $PYTHON train_soft_cluster.py `
    --benchmark   r1 `
    --config      F `
    --result-dir  $RESULT `
    --cache-only `
    2>&1 | Tee-Object -Append -FilePath $LOG

if ($LASTEXITCODE -eq 0) {
    $n = (Get-ChildItem "$SOFT\$RESULT\llm_cache\*_cluster.json" -ErrorAction SilentlyContinue).Count
    Log "DONE  $n cache files in $RESULT\llm_cache\"
} else {
    Log "FAILED  exit=$LASTEXITCODE"
}
