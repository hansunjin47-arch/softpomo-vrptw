
# RL cluster cache pipeline: r1 -> rc1 -> c1
# 각 benchmark: 500ep RL routes 추출 -> generate_llm_cache (zero-shot 로직 그대로, clustering만 RL로)

$python   = "C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe"
$work     = "C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo"
$rlResult = "C:\Users\hansu\PycharmProjects\PythonProject\original_POMO\result"

$tasks = @(
    @{ bench='r1';  ckpt="$rlResult\config_F_r1_p2_pomo100\r102-r112\checkpoint-500.pt" },
    @{ bench='rc1'; ckpt="$rlResult\config_F_rc1_p2_pomo100\rc102-rc108\checkpoint-500.pt" },
    @{ bench='c1';  ckpt="$rlResult\config_F_p2_pomo100\c102-c109\checkpoint-500.pt" }
)

foreach ($t in $tasks) {
    $bench = $t.bench
    $ckpt  = $t.ckpt

    Write-Host ""
    Write-Host "========== [$bench] START ==========" -ForegroundColor Cyan
    Write-Host (Get-Date)

    # --- Step 1: RL routes 추출 (500ep checkpoint) ---
    Write-Host "  [1/2] extract_rl_routes.py ..."
    & $python "$work\extract_rl_routes.py" --benchmark $bench --config F --ckpt "$ckpt"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [ERROR] extract_rl_routes failed for $bench" -ForegroundColor Red
        continue
    }

    # --- Step 2: generate_llm_cache (zero-shot 디테일 그대로, clustering만 RL routes) ---
    Write-Host "  [2/2] generate_llm_cache.py ..."
    & $python "$work\generate_llm_cache.py" --benchmark $bench --config F --refresh-cache
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [ERROR] generate_llm_cache failed for $bench" -ForegroundColor Red
        continue
    }

    Write-Host "  [$bench] DONE" -ForegroundColor Green
    Write-Host (Get-Date)
}

Write-Host ""
Write-Host "========== ALL DONE ==========" -ForegroundColor Green
