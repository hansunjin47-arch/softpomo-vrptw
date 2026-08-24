# run_eval_deepseek.ps1
# DeepSeek-R1:32b로 LLM-guided 평가 (Config A-F × c1/r1/rc1)
# 기존 Qwen3 캐시 백업 후 DeepSeek로 재생성

$PYTHON   = "C:\Users\hansu\anaconda3\envs\gpils\python.exe"
$SOFT_DIR = "C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo"
$CKPT_ROOT = "C:\Users\hansu\PycharmProjects\PythonProject\original_POMO\result"
$CACHE_DIR = "$SOFT_DIR\result_soft\llm_cache"
$LOG      = "$SOFT_DIR\result_soft\eval_deepseek.log"

New-Item -ItemType Directory -Path "$SOFT_DIR\result_soft" -Force | Out-Null

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $LOG -Value $line
}

# ── 1. Qwen 캐시 백업 ─────────────────────────────────────────────────────────
$CACHE_BACKUP = "$SOFT_DIR\result_soft\llm_cache_qwen3_backup"
if (Test-Path $CACHE_DIR) {
    if (-not (Test-Path $CACHE_BACKUP)) {
        Log "Backing up Qwen3 cache → llm_cache_qwen3_backup"
        Rename-Item -Path $CACHE_DIR -NewName "llm_cache_qwen3_backup"
    } else {
        Log "Backup already exists. Clearing current cache."
        Remove-Item -Recurse -Force $CACHE_DIR
    }
}
New-Item -ItemType Directory -Path $CACHE_DIR -Force | Out-Null
Log "Fresh cache dir created: $CACHE_DIR"

# ── 2. Config × Benchmark 정의 ────────────────────────────────────────────────
$RUNS = @(
    # config, benchmark, checkpoint_path
    @("A", "c1",  "$CKPT_ROOT\config_A_pomo100\c102-c109\checkpoint-last.pt"),
    @("A", "rc1", "$CKPT_ROOT\config_A_rc1_pomo100\rc102-rc108\checkpoint-last.pt"),
    @("A", "r1",  "$CKPT_ROOT\config_A_r1_pomo100\r102-r112\checkpoint-last.pt"),
    @("B", "c1",  "$CKPT_ROOT\config_B_pomo100\c102-c109\checkpoint-last.pt"),
    @("B", "rc1", "$CKPT_ROOT\config_B_rc1_pomo100\rc102-rc108\checkpoint-last.pt"),
    @("B", "r1",  "$CKPT_ROOT\config_B_r1_pomo100\r102-r112\checkpoint-last.pt"),
    @("C", "c1",  "$CKPT_ROOT\config_C_pomo100\c102-c109\checkpoint-last.pt"),
    @("C", "rc1", "$CKPT_ROOT\config_C_rc1_pomo100\rc102-rc108\checkpoint-last.pt"),
    @("C", "r1",  "$CKPT_ROOT\config_C_r1_pomo100\r102-r112\checkpoint-last.pt"),
    @("D", "c1",  "$CKPT_ROOT\config_D_pomo100\c102-c109\checkpoint-last.pt"),
    @("D", "rc1", "$CKPT_ROOT\config_D_rc1_pomo100\rc102-rc108\checkpoint-last.pt"),
    @("D", "r1",  "$CKPT_ROOT\config_D_r1_pomo100\r102-r112\checkpoint-last.pt"),
    @("E", "c1",  "$CKPT_ROOT\config_E_pomo100\c102-c109\checkpoint-last.pt"),
    @("E", "rc1", "$CKPT_ROOT\config_E_rc1_pomo100\rc102-rc108\checkpoint-last.pt"),
    @("E", "r1",  "$CKPT_ROOT\config_E_r1_pomo100\r102-r112\checkpoint-last.pt"),
    @("F", "c1",  "$CKPT_ROOT\config_F_pomo100\c102-c109\checkpoint-last.pt"),
    @("F", "rc1", "$CKPT_ROOT\config_F_rc1_pomo100\rc102-rc108\checkpoint-last.pt"),
    @("F", "r1",  "$CKPT_ROOT\config_F_r1_pomo100\r102-r112\checkpoint-last.pt")
)

# ── 3. 실행 ───────────────────────────────────────────────────────────────────
Log "Starting DeepSeek-R1:32b eval — 18 runs (Config A-F × c1/rc1/r1)"
$ok = 0; $fail = 0

Set-Location $SOFT_DIR

foreach ($run in $RUNS) {
    $cfg, $bm, $ckpt = $run

    if (-not (Test-Path $ckpt)) {
        Log "SKIP ${cfg}_${bm}: checkpoint not found — $ckpt"
        $fail++
        continue
    }

    Log "START Config $cfg / $bm"
    $t0 = Get-Date

    & $PYTHON eval_soft_plug.py `
        --config   $cfg `
        --benchmark $bm `
        --resume   $ckpt `
        --model    "deepseek-r1:32b" | Tee-Object -Append -FilePath $LOG

    if ($LASTEXITCODE -eq 0) {
        $elapsed = [int]((Get-Date) - $t0).TotalSeconds
        Log "DONE  Config $cfg / $bm  (${elapsed}s)"
        $ok++
    } else {
        Log "FAILED Config $cfg / $bm"
        $fail++
    }
}

Log "========================================"
Log "Finished: $ok OK  /  $fail FAILED  / 18 total"
Log "Results: $SOFT_DIR\result_soft\plug_*\summary.txt"
Log "========================================"
