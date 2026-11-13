# RL recluster + RL few-shot — Phase 2 training for R1
#
# Step 1: gen_test_rl_cluster.py
#   → aug×8 RL inference with checkpoint-500.pt
#   → LLM scores RL clusters → saves {name}_rl_cluster.json
# Step 2: Phase 2 training (epochs 501-1000)
#   → resumes from checkpoint-500.pt
#   → uses RL clusters (auto-loaded via _rl_cluster.json priority)
#   → RL few-shot example from r106_rl_cluster.json

$PYTHON  = "C:\Users\hansu\anaconda3\envs\gpils\python.exe"
$SOFT    = "C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo"
$CKPT    = "$SOFT\result_soft_pure_rl\r102-r112\checkpoint-500.pt"
$RESULT  = "result_soft_rl_fewshot"
$LOGDIR  = "$SOFT\logs\$RESULT"

New-Item -ItemType Directory -Force $LOGDIR | Out-Null

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Write-Host $line
    Add-Content -Path "$LOGDIR\run.log" -Value $line
}

Set-Location $SOFT

# ── Step 1: RL recluster (gen_test_rl_cluster.py) ─────────────────────────────
Log "START  Step1: gen_test_rl_cluster  benchmark=r1  aug=8"

& $PYTHON gen_test_rl_cluster.py `
    --benchmark  r1 `
    --resume     $CKPT `
    --result-dir $RESULT `
    --config     F `
    --aug        8 `
    --rl-fewshot `
    2>&1 | Tee-Object -Append -FilePath "$LOGDIR\recluster.log"

if ($LASTEXITCODE -ne 0) {
    Log "FAILED Step1 exit=$LASTEXITCODE"
    exit 1
}
Log "Step1 done — rl_cluster.json files generated"

# ── Step 2: Phase 2 training (501–1000 epochs) ────────────────────────────────
Log "START  Step2: Phase2 training  epochs=501-1000  config=F  rl-fewshot"

& $PYTHON train_soft_cluster.py `
    --benchmark                r1 `
    --config                   F `
    --epochs                   1000 `
    --result-dir               $RESULT `
    --resume                   $CKPT `
    --rl-fewshot `
    --experience-refresh-epoch 999999 `
    2>&1 | Tee-Object -Append -FilePath "$LOGDIR\train_r1.log"

if ($LASTEXITCODE -eq 0) {
    Log "DONE"
} else {
    Log "FAILED Step2 exit=$LASTEXITCODE"
}
