# Full pipeline: LLM cache generation -> 1000-epoch training
# Kim clustering + DeepSeek init zeroshot with BKS few-shot + ranking output
# Benchmark: r1  |  Config: F  |  --experience-refresh-epoch 999999

$PYTHON = "C:\Users\hansu\anaconda3\envs\gpils\python.exe"
$SOFT   = "C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo"
$RESULT = "result_soft_bks_ranking"
$LOGDIR = "$SOFT\logs\$RESULT"

New-Item -ItemType Directory -Force $LOGDIR | Out-Null

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Write-Host $line
    Add-Content -Path "$LOGDIR\pipeline.log" -Value $line
}

Set-Location $SOFT

# ── Step 1: LLM cache generation ──────────────────────────────────────────────
Log "=== STEP 1: LLM cache generation (--cache-only) ==="

& $PYTHON train_soft_cluster.py `
    --benchmark   r1 `
    --config      F `
    --result-dir  $RESULT `
    --cache-only `
    2>&1 | Tee-Object -Append -FilePath "$LOGDIR\cache_r1.log"

if ($LASTEXITCODE -ne 0) {
    Log "FAILED (exit=$LASTEXITCODE). Aborting training."
    exit 1
}

$n = (Get-ChildItem "$SOFT\$RESULT\llm_cache\*_cluster.json" -ErrorAction SilentlyContinue).Count
Log "Cache generation done. $n cache files in $RESULT\llm_cache\"

# ── Step 2: 1000-epoch training ───────────────────────────────────────────────
Log "=== STEP 2: Training 1000 epochs (--experience-refresh-epoch 999999) ==="

& $PYTHON train_soft_cluster.py `
    --benchmark               r1 `
    --config                  F `
    --epochs                  1000 `
    --result-dir              $RESULT `
    --experience-refresh-epoch 999999 `
    2>&1 | Tee-Object -Append -FilePath "$LOGDIR\train_r1.log"

if ($LASTEXITCODE -eq 0) {
    Log "=== DONE: training complete ==="
} else {
    Log "=== Training exited with code $LASTEXITCODE ==="
}
