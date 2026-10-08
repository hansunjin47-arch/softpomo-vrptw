# RC101: four offline experiment workflows

## 새 PC에서 시작하기

이 폴더는 기존 PC의 절대경로, 환경변수 API 키, 실험 결과 폴더 없이 실행하도록 구성되어 있습니다.
아래 명령은 Windows PowerShell 예시입니다. Linux/WSL에서는 가상환경 Python 경로를
`.venv/bin/python`으로 바꾸면 됩니다.

```powershell
git clone https://github.com/hansunjin47-arch/softpomo-vrptw.git
cd softpomo-vrptw/experiments/rc101
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
```

먼저 [PyTorch 설치 안내](https://pytorch.org/get-started/locally/)에서 **자신의 GPU에 맞는**
설치 명령을 선택하고, 그 명령의 `pip` 대신 `.venv\Scripts\python.exe -m pip`를 사용하세요.
그다음 아래를 실행합니다. 전체 학습은 CUDA GPU가 필요합니다.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest test_release
.venv\Scripts\python.exe server/train_cached.py --method phase2 --check-only --cpu --out runs/install_check
.venv\Scripts\python.exe server/train_cached.py --method pure --batch-size 128 --epochs 1000 --out runs/pure128
.venv\Scripts\python.exe evaluate.py --run runs/pure128
```

마지막 평가는 `status.json`이 `complete`이고 1000 epoch 체크포인트가 만들어진 뒤 실행합니다.
시간 예산으로 중단되었다면 학습 명령에 `--resume`을 붙여 완료시킵니다.
Kim/Phase2는 `--method kim` 또는 `--method phase2`로 바꾸고 출력 폴더도 별도로 지정합니다.
Pure/Kim은 무작위 초기화부터, Phase2는 포함된 Pure500에서 501~1000 epoch를 학습합니다.
**새 PC에서 실행한다는 것과 Phase1까지 새로 학습한다는 것은 다릅니다.**
Phase1도 새로 학습하려면 해당 모델에 맞는 기준 경로·confidence 캐시를 다시 준비해야 합니다.

### 파일이 전부 필수인가요?

모든 파일이 매 실행에 사용되는 것은 아닙니다. 현재 배포본은 네 방법의 실행과 재현 검증을
하나의 묶음으로 검사하므로, 사용자가 파일을 임의로 지우면 manifest 검사가 실패합니다.

| 파일/폴더 | 역할과 필요 범위 |
| --- | --- |
| `server/train_cached.py`, `configs/` | 세 RL 방법의 공통 실행 진입점과 학습 설정 |
| `train_vrptw.py`, `vrptw_env.py`, `event_observation.py`, `advantage.py` | 정책·환경·관측·학습 계산 |
| `train_vrptw_llm.py`, `train_soft_cluster.py` | Kim/Phase2 모델과 학습기; 부모 클래스도 포함 |
| `*LLMModule.py`, `*Ontology.py`, `generate_vrptw.py` | 현재 코드의 import 의존성. API 생성/무작위 고객 생성은 지원 명령에서 실행하지 않음 |
| `data/` | train80/test80 사건 파일과 공개 관측; `split.json`이 학습/평가 용도를 구분 |
| `cache/kim`, `cache/phase2` | 각 방법의 기존 LLM 출력. Pure RL 자체에는 필요 없음 |
| `checkpoints/pure500.pt` | Phase2의 초기 모델/optimizer. Pure/Kim 초기 학습에는 필요 없음 |
| `evaluate.py` | 학습된 모델의 test 평가와 공통 결과 집계 |
| `history.py`, `lkh_io.py` | LKH 계획 행렬 작성·외부 실행·저장 경로 재평가 |
| `reference/history/` | 기존 LKH 경로 재현과 행렬 검증용 자료. 신경망 학습 입력이 아님 |
| `bundle_manifest.json`, `test_release.py`, `refresh_manifest.py` | 무결성 검사·테스트·의도적인 변경의 버전 갱신 |
| `README.md`, `requirements.txt`, Git 설정 파일 | 새 환경 설치와 실행 안내 |

코드 읽는 순서는 `server/train_cached.py` → `configs/` → 해당 trainer → `vrptw_env.py` →
`evaluate.py`가 좋습니다. 핵심 분기에는 한국어 주석을 추가했습니다. LLM/ontology 파일의
이름만 보고 API 호출 또는 최신 ontology 실험이 실행된다고 해석하지 마세요.
기존 코드에는 공유 모듈 안의 미사용 분기도 남아 있습니다. **파일 단위 정리는 완료했지만,
모든 함수가 네 실행 경로에 필요한 최소 코드로 재작성된 것은 아닙니다.**

Run all commands **from this directory**. Python 3.12 is recommended. These are
the original train80/test80 four-method settings, not the later ontology ablations.

## Installation

Create and activate a virtual environment. Install a CUDA-enabled PyTorch build
appropriate for your GPU using https://pytorch.org/get-started/locally/, then:

```sh
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Full training/evaluation requires a CUDA GPU. `--cpu` supports small checks only.
For an RTX 5090, use a PyTorch/CUDA build supporting that GPU; do not reuse an old
CUDA wheel. No API key, OpenRouter, Ollama, MLflow server, or external parent
repository imports are required by the supported commands. Network/API calls are
blocked in the RL runner, and missing confidence caches are errors.

## Quick verification (no full training)

```sh
python server/train_cached.py --method pure --check-only --cpu --out runs/check_pure
python server/train_cached.py --method kim --check-only --cpu --out runs/check_kim
python server/train_cached.py --method phase2 --check-only --cpu --out runs/check_phase2
python history.py --replay-reference --out runs/history_reference
python -m unittest test_release
```

For a forward/backward/checkpoint smoke test use `--smoke` instead of
`--check-only`. It runs one rain and one accident batch, not an epoch.
Always use a fresh output directory for checks. The included SHA-256 manifest
checks source, configuration, input data, caches, and the supplied checkpoint.

## Train and evaluate

```sh
python server/train_cached.py --method pure --out runs/pure --epochs 1000
python evaluate.py --run runs/pure

python server/train_cached.py --method kim --out runs/kim --epochs 1000
python evaluate.py --run runs/kim

python server/train_cached.py --method phase2 --out runs/phase2 --epochs 1000
python evaluate.py --run runs/phase2
```

Run these sequentially on a shared GPU. Defaults are seed 42 and batch 64,
preserving the historical experiment configuration. To test batch 128, add
`--batch-size 128` to the training command. This is a new setting, not an exact
reproduction of the historical batch-64 result. `--seed 43` changes training seed.
For Phase2 this changes continuation randomness, **not** the supplied Phase1 seed.

Pure and Kim train epochs 1-1000. Phase2 loads `checkpoints/pure500.pt` and its
optimizer, then trains epochs 501-1000. Its supplied confidence caches were
generated from that exact reference model. Do not substitute another Pure500
checkpoint without regenerating reference routes and matching caches. Cache/API
generation is deliberately outside this offline release.

The default runner stops gracefully after approximately 11.5 hours at an epoch
boundary; this is **not a scheduler-enforced 12-hour limit**. Resume with the same
command and `--resume` (repeat batch/seed overrides). Check `status.json` for
`complete` versus `needs_resume`. It does not auto-submit another job.
`--max-hours` controls this budget. `checkpoint-last.pt` is saved periodically;
epoch 500 and multiples of 1000 also receive numbered checkpoints.

Evaluation writes `results.json` (per-scenario routes/metrics), `summary.json`
(mean and sample standard deviation), and `protocol.json`. It uses the checkpoint
and settings from `--run`; it never trains on test data. Defaults: 100 starts,
8 augmentations, greedy decoding, best candidate by simulated reward. For an
interrupted or non-1000 run use `--checkpoint checkpoint-last.pt`. A fresh
`--out` avoids overwriting prior evaluations. `--smoke` evaluates two scenarios.

## LKH-3 Full History

The included reference plan can be replayed without installing LKH-3. To solve
again, install LKH-3 separately and pass its executable:

```sh
python history.py --lkh /path/to/LKH --out runs/history_new --time-limit 600
```

On Windows, use a native executable or run the entire command in WSL with its
own Python environment and Linux LKH executable. No fixed WSL distribution or
root account is assumed. LKH binaries/source are not redistributed here.

The matrix averages full event-weighted distances over train80 only. Event onset
and duration do not dilute planning weights. Rounding (`numpy.rint`) occurs once
after averaging. One static route plan is replayed across all test80 cases using
floating-point, departure-time-dependent travel. Solver settings: seed 42,
one run, 10000 maximum trials, 600-second limit. No infeasibility relaxation is
silently applied. Time-limited solving may not reproduce the exact reference plan.

## Protocol and interpretation

- Fixed RC101: 100 customers, capacity 200, maximum 25 vehicles, horizon 240.
- Train80 and disjoint test80 each contain 40 rain and 40 accident scenarios.
  These are synthetic event variations, not 80 independent customer networks.
- Full-service cost is `J = 0.04*K + T/5760 + 0.0075*L`; reward is `-J`.
  `L` and late stops include depot returns. Travel excludes waiting/service.
- Kim uses fixed heuristic groups; Phase2 uses scenario-specific Pure500 route
  groups and offline simulated feedback. Confidence is cached, not refreshed
  during training. This is not a public-information-only online evaluation.
- Guidance uses the actual first customer's reference group until depot return;
  all 100 starts remain available. Groups do not constrain feasible actions.
- Bias strength is 5. In this implementation, a customer score below the group's
  depot score becomes an effective score of -1. This is a soft negative bias,
  not a feasibility mask. Otherwise the customer confidence supplies the bias;
  out-of-group customers receive zero. Depot scores are retained.
- Accident observations/bias are gated at onset on each vehicle clock, including
  reset to zero for a new route. Rain is known at dispatch. Offline reference
  feedback and candidate selection still use the simulator.
- LKH solves its static constrained formulation, not the RL weighted objective.
  Replay compares outcomes; it does not make the optimization problems identical.

## Files, provenance, and modifications

`configs/` defines the three RL methods. `data/split.json` records membership and
scenario hashes. `cache/{kim,phase2}` includes both train and test caches.
`checkpoints/pure500.pt` is the approximately 15 MB seed-42 train80 reference
checkpoint, included to keep Phase2 runnable with the matching caches.
Only load trusted checkpoints: PyTorch checkpoint loading uses pickle.
`reference/history` contains the original static plan, matrix, and summary.

Source workflows: `rc101_pure_train80_fresh1000_20260929`,
`rc101_kim_train80_fresh1000_20260929`, and
`rc101_pure500_train80_phase2_20260930`. Test caches come from each guided run's
`eval_1000_test80/cache`. Runtime logs, API requests/credentials, and later
experimental outputs are not included. Solomon instance data is retained in
the scenario files; cite Solomon (1987) and respect upstream dataset terms.

Use the documented entry points, not the legacy modules' standalone main blocks.
The legacy core is preserved to avoid changing the numerical algorithm while
packaging. CLI batch/seed/epoch overrides are recorded in run settings. For
intentional source/config changes, regenerate the manifest with
`python refresh_manifest.py`; this creates a **different experiment version** and
will intentionally prevent resuming an old run under the new code. The manifest
checks consistency, not authenticity; use Git history for provenance.

## Release validation

Packaging was checked with Python 3.12.6, PyTorch 2.9.1+cu128, and CPU execution:
four contract tests; rain/accident backward steps and checkpoint roundtrips for
Pure, Kim, and Phase2; two-scenario evaluation for all three using historical
trained weights; and all 80 scenarios replayed for the LKH reference plan.
Reference replay reproduces mean K=19, T=1785.70642065, L=11.15164798.
An untrained smoke checkpoint can fail full-service evaluation; use trained
weights for performance evaluation. A fresh 1000-epoch run and a fresh native
LKH solve were not part of this packaging check.
