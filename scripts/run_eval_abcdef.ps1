# run_eval_abcdef.ps1 — Config A~F × C1/RC1/R1 LLM plug-in inference
# Training: RL only (기존 checkpoint)
# Inference: LLM soft-clustering bias (기존 llm_cache 활용, LLM 호출 없음)
# 결과: result_soft/plug_{config}_{benchmark}/summary.txt

$ROOT   = "C:\Users\hansu\PycharmProjects\PythonProject"
$ORIG   = "$ROOT\original_POMO\result"
$SCRIPT = "$ROOT\soft_pomo\eval_soft_plug.py"

# checkpoint 경로 패턴
#   C1  → config_{X}_pomo100/c102-c109/checkpoint-last.pt
#   RC1 → config_{X}_rc1_pomo100/rc102-rc108/checkpoint-last.pt
#   R1  → config_{X}_r1_pomo100/r102-r112/checkpoint-last.pt

$runs = @()
foreach ($cfg in @("A","B","C","D","E","F")) {
    $runs += @{ config=$cfg; benchmark="c1";  ckpt="$ORIG\config_${cfg}_pomo100\c102-c109\checkpoint-last.pt"         }
    $runs += @{ config=$cfg; benchmark="rc1"; ckpt="$ORIG\config_${cfg}_rc1_pomo100\rc102-rc108\checkpoint-last.pt"   }
    $runs += @{ config=$cfg; benchmark="r1";  ckpt="$ORIG\config_${cfg}_r1_pomo100\r102-r112\checkpoint-last.pt"      }
}

$total  = $runs.Count
$passed = 0
$failed = @()
$i = 1

foreach ($r in $runs) {
    Write-Host ""
    Write-Host "===== [$i/$total] Config $($r.config) / $($r.benchmark.ToUpper()) =====" -ForegroundColor Cyan

    if (-not (Test-Path $r.ckpt)) {
        Write-Host "  [SKIP] checkpoint not found: $($r.ckpt)" -ForegroundColor Yellow
        $failed += "Config $($r.config) $($r.benchmark) (missing ckpt)"
        $i++; continue
    }

    & "C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe" $SCRIPT `
        --benchmark $r.benchmark `
        --config    $r.config `
        --resume    $r.ckpt

    if ($LASTEXITCODE -eq 0) {
        Write-Host "[DONE] Config $($r.config) $($r.benchmark)" -ForegroundColor Green
        $passed++
    } else {
        Write-Host "[FAILED] Config $($r.config) $($r.benchmark)" -ForegroundColor Red
        $failed += "Config $($r.config) $($r.benchmark)"
    }
    $i++
}

Write-Host ""
Write-Host "===== Summary: $passed/$total passed =====" -ForegroundColor Cyan
if ($failed.Count -gt 0) {
    Write-Host "Failed:" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
}
