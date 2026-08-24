# run_eval_ef.ps1 — Config E & F LLM plug-in inference (C1 → RC1 → R1)
# 기존 LLM 캐시 활용 (LLM 호출 없음)

$ROOT   = "C:\Users\hansu\PycharmProjects\PythonProject"
$ORIG   = "$ROOT\original_POMO\result"
$SCRIPT = "$ROOT\soft_pomo\eval_soft_plug.py"

$runs = @(
    @{ config="E"; benchmark="c1";  ckpt="$ORIG\config_E_pomo100\checkpoint-last.pt"     },
    @{ config="E"; benchmark="rc1"; ckpt="$ORIG\config_E_rc1_pomo100\checkpoint-last.pt" },
    @{ config="E"; benchmark="r1";  ckpt="$ORIG\config_E_r1_pomo100\checkpoint-last.pt"  },
    @{ config="F"; benchmark="c1";  ckpt="$ORIG\config_F_pomo100\checkpoint-last.pt"     },
    @{ config="F"; benchmark="rc1"; ckpt="$ORIG\config_F_rc1_pomo100\checkpoint-last.pt" },
    @{ config="F"; benchmark="r1";  ckpt="$ORIG\config_F_r1_pomo100\checkpoint-last.pt"  }
)

$total = $runs.Count
$i = 1
foreach ($r in $runs) {
    Write-Host ""
    Write-Host "===== [$i/$total] Config $($r.config) / $($r.benchmark.ToUpper()) =====" -ForegroundColor Cyan
    Write-Host "  checkpoint: $($r.ckpt)"
    python $SCRIPT `
        --benchmark $r.benchmark `
        --config    $r.config `
        --resume    $r.ckpt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAILED] Config $($r.config) $($r.benchmark)" -ForegroundColor Red
        exit 1
    }
    Write-Host "[DONE] Config $($r.config) $($r.benchmark)" -ForegroundColor Green
    $i++
}

Write-Host ""
Write-Host "===== All done =====" -ForegroundColor Green
