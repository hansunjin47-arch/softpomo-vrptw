# run_cache_c1_deepseek.ps1
# DeepSeek-R1:32b로 C1 benchmark LLM 클러스터 캐시 초기 생성
# --cache-only: LLM 캐시만 생성하고 종료 (RL 학습 없음)
# --refresh-cache: 기존 캐시 무시하고 LLM 새로 호출

$PYTHON   = "C:\Users\hansu\anaconda3\envs\gpils\python.exe"
$SOFT_DIR = "C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo"
$CACHE_DIR = "$SOFT_DIR\result_soft\llm_cache"
$LOG      = "$SOFT_DIR\result_soft\cache_c1_deepseek.log"

New-Item -ItemType Directory -Path "$SOFT_DIR\result_soft" -Force | Out-Null
New-Item -ItemType Directory -Path $CACHE_DIR -Force | Out-Null

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $LOG -Value $line
}

# ── Qwen 캐시 백업 (C1 관련 파일만) ──────────────────────────────────────────
$BACKUP_DIR = "$SOFT_DIR\result_soft\llm_cache_qwen3_backup"
New-Item -ItemType Directory -Path $BACKUP_DIR -Force | Out-Null

$c1_files = Get-ChildItem "$CACHE_DIR\C1*", "$CACHE_DIR\c1*", "$CACHE_DIR\C10*", "$CACHE_DIR\c10*" -ErrorAction SilentlyContinue
if ($c1_files.Count -gt 0) {
    Log "Moving $($c1_files.Count) old C1 cache files to backup..."
    $c1_files | Move-Item -Destination $BACKUP_DIR -Force
} else {
    Log "No existing C1 cache files found."
}

# ── Cache 생성 ────────────────────────────────────────────────────────────────
Log "Starting DeepSeek-R1:32b cache generation for C1..."
Set-Location $SOFT_DIR

& $PYTHON train_soft_cluster.py `
    --benchmark  c1 `
    --model      "deepseek-r1:32b" `
    --cache-only `
    --no-cot `
    2>&1 | Tee-Object -Append -FilePath $LOG

if ($LASTEXITCODE -eq 0) {
    $n = (Get-ChildItem "$CACHE_DIR\C1*" -ErrorAction SilentlyContinue).Count
    Log "Done. $n cache files generated in $CACHE_DIR"
} else {
    Log "FAILED. Check log: $LOG"
}
