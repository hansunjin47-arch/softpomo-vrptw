# 라우팅 구현 문제와 수정 이력

기준일: 2026-10-18

현재 본 학습 버전은 **v5 full-graph**다. 아래 v1-v4 설명은 수정 이력이며, v4의 step별 gradient 재계산 방식은 중단했다. 최신 변경 범위는 문서 마지막의 v5 대조표를 기준으로 한다.

## 핵심 요약

연구 목표는 과거 이벤트로 학습한 정책이 새로운 시나리오에서 그때까지 공개된 정보만 사용해 여러 차량의 운행을 동적으로 조정하는 것이다. 기존 코드는 차량별 경로를 순차 구성하므로, 실제 동시 운행의 온라인 평가로 해석하기에는 시간 및 고객 배정의 인과관계 문제가 있었다.

문제는 세 범주로 구분해야 한다.

1. 기존 학습 코드의 구체적인 구현 오류: 최초 depot 출발 이동 누락 및 강제 시작 처리 불일치.
2. 기존 방법의 운영 및 평가 가정: 차량별 시간 초기화, 실현된 결과를 이용한 다중 후보 선택. 정적 또는 오프라인 계획에서는 사용할 수 있지만, 우리가 목표로 하는 온라인 평가와는 구분해야 한다.
3. 새 동시 운행 초안에 추가된 문제: 차량 상한 누락, 학습 가능한 전역 ADVANCE, 기존과 다른 신규 차량 출발 규칙, 일부 학습 설정 차이. 이것들은 원래 코드의 문제로 돌리면 안 된다.

기존 결과를 삭제할 필요는 없지만, 아래 버전과 평가 조건을 표시하지 않은 채 새 결과와 직접적인 성능 향상/저하로 해석하면 안 된다.

## 버전별 구분

| 구분 | 구현 및 실행 위치 | 의미 |
|---|---|---|
| 기존 순차 방식 | `result/bks_normalized_late10_train20_20261013/rc101/` 등 보존된 실험 | 강제 시작 학습, 다중 후보 평가. 아래 오류의 적용 범위는 실제 사용한 소스별로 확인 필요 |
| Learned-first 순차 방식 | `result/rc101_learned_first_train20_20261017/` | depot에서 시작하고 첫 고객도 학습. one-shot 평가지만 차량별 시간 초기화 구조는 유지 |
| Phase2 온라인 v2.2 | `result/rc101_phase2_online_gpt_train20_v2_2_20261018/` | train 프롬프트 복원, 차량별 온라인 응답 시각 보호. 동시 차량 환경은 아님 |
| 동시 운행 v1 | `experiments/concurrent_fleet_v1/` | 전역 ADVANCE 포함. 최초 초안에는 차량 상한 누락. 이후 capped 실행에서 복구 |
| 동시 운행 v2 | `experiments/concurrent_fleet_v2/` | 전역 ADVANCE 제거, 차량별 depot 대기 선택. 양수 시각에도 신규 차량 투입 가능 |
| 동시 운행 v3 | `experiments/concurrent_fleet_v3/` | t=0 신규 투입, 기존 차량의 다음 의사결정 선택 후 사건 처리. t>0 신규 투입 금지 및 환경의 차량 지정 |

v1 디렉터리의 현재 소스에는 상한 수정이 들어 있다. 최초 무상한 실행과 구분할 때는 각 결과 폴더의 `source/` 스냅샷 및 `protocol.json`을 기준으로 한다.

## 기존 순차 방식의 문제

### 1 최초 depot 출발 이동이 학습에서 누락

**분류: 확인된 구현 오류.**

기존 `_train_one_batch`는 환경 reset 직후 depot을 먼저 선택하지 않고 고객을 바로 선택했다. 환경의 `step()`은 `previous_node is None`이면 이동시간을 0으로 처리한다. 이 조합 때문에 학습 중 최초 depot에서 첫 고객까지의 이동시간이 누락될 수 있었다.

- 실제보다 이른 도착 및 이후 시간 진행으로 학습하게 된다.
- 최초 구간의 이동 비용과 그로 인한 지각도 잘못 계산될 수 있다.
- 평가가 depot부터 정상 시작했다면, 학습 오류가 있었다는 이유만으로 평가표의 이동시간까지 모두 잘못 집계됐다고 할 수는 없다. 학습과 평가를 나눠 봐야 한다.

**수정:** learned-first 학습에서는 먼저 depot을 선택한 뒤 첫 고객을 정책이 선택한다. 이미 학습된 옛 체크포인트의 학습 과정이 소급 수정되는 것은 아니다.

근거: [기존 학습 소스](../result/bks_normalized_late10_train20_20261013/rc101/train_vrptw.py), [보존된 환경 소스](../result/rc101_learned_first_train20_20261017/source/vrptw_env.py), [수정된 학습 소스](../result/rc101_learned_first_train20_20261017/source/train_vrptw.py).

### 2 학습 루프와 모델 양쪽에 강제 시작 처리 존재

**분류: 확인된 호출 순서 불일치.**

기존 학습 루프가 먼저 시작 고객을 강제로 선택했는데, 모델에도 `selected_count == 1`에서 시작 고객을 강제로 지정하는 분기가 있었다. 따라서 depot을 기준으로 설계된 모델의 첫 고객 분기가, 학습에서는 이미 고객을 선택한 뒤에도 실행될 수 있었다.

강제 시작 자체가 항상 잘못된 것은 아니다. 문제는 학습 루프와 모델이 서로 다른 초기 상태를 가정했다는 점이다.

**수정:** `learn_first_customer=True`에서는 모델의 강제 시작 분기를 우회하며, 실제 첫 고객 선택의 확률을 loss에 포함한다.

### 3 실현된 미래 결과를 이용한 test 후보 선택

**분류: 온라인 평가 목표와의 불일치.**

기존 평가의 여러 시작점 및 augmentation 후보를 동일 test 시나리오의 실제 reward로 끝까지 평가한 뒤 best를 고르면, 의사결정 당시 알 수 없었던 결과가 최종 선택 기준에 들어간다.

- 오프라인 시나리오 최적화나 oracle 기준으로는 별도로 보고할 수 있다.
- 예측 모델로 가능한 미래를 샘플링하는 온라인 lookahead까지 모두 금지된다는 뜻은 아니다.
- 문제는 실제 test의 미래 실현값으로 후보를 비교해 선택하고 이를 인과적인 온라인 정책 성능이라고 부르는 것이다.

**수정:** learned-first 실험에는 단일 greedy rollout 평가를 도입했다. 단, 이것만으로 아래 차량 간 시간 문제가 해결되지는 않는다.

근거: 학습 소스의 `_best_solution`, `analysis/run_learned_first_train20.py`, 각 평가 폴더의 contract 및 summary.

### 4 한 차량의 미래 결과가 다른 차량의 과거 결정에 반영

**분류: 동시 운행 온라인 모델과의 구조적 불일치.**

기존 환경은 차량 A의 경로를 끝까지 만든 뒤 depot 복귀 시 `current_time`을 0으로 초기화한다. 차량 B는 다시 0분부터 시작하지만, A가 방문한 고객은 이미 방문 마스크에서 제외돼 있다.

예를 들어 A가 80분 사고를 반영해 방문 고객을 바꾼 후, B가 0분부터 남은 고객으로 경로를 구성하면 A의 사고 이후 선택이 B의 사고 이전 고객 선택에 간접 반영된다. B에게 사고 feature를 직접 주지 않아도 남은 고객 집합이 영향을 전달할 수 있다.

이 문제는 LLM이 없는 Pure RL에도 해당한다. 여러 차량이 동시에 운행한다는 해석에서의 문제이며, 순차적으로 전체 경로를 구성하는 오프라인 알고리즘 자체가 항상 잘못됐다는 뜻은 아니다.

**수정 방향:** 공통 시간축에서 사건을 처리하고, 아직 서비스가 끝나지 않은 고객은 방문 완료가 아니라 예약 상태로 구분한다. v3는 이를 위한 별도 실험 구현이다.

### 5 Training reward와 test reward의 의미 차이

**분류: 로그 해석상의 주의.**

기존 학습 로그는 `_train_one_batch`에서 `reward.max(dim=1)`을 취한 rollout 최대 reward의 평균이다. 모든 rollout의 평균 reward나 one-shot greedy reward가 아니다.

따라서 training curve가 좋아졌다는 사실만으로 one-shot 성능이 향상됐다고 볼 수 없다. 그래프에는 집계 방식을 명시하고, 별도의 고정 사례 평가와 함께 해석해야 한다.

## Phase2와 LLM 응답의 문제

### 6 응답 생성 상태와 적용 상태의 시간 불일치

**분류: 기존 온라인 v2의 응답 재사용 범위 오류.**

한 차량의 80분 상태를 입력해 만든 응답을 다른 차량이 71분에 사용하는 식의 재사용은, 적용 시점보다 뒤의 상태를 참조하게 된다. 차량별 시계가 다시 시작하는 환경에서 시나리오당 응답 하나를 무조건 공유하면 이런 문제가 생길 수 있다.

**수정:** 온라인 v2.1 이후 차량 ID 및 응답 생성 시각에 따라 재사용 범위를 제한했다. 이 수정은 미래 상태 응답의 재사용을 막는 것이며, 전체 차량을 공통 시계로 바꾸는 수정은 아니었다.

Train의 사전 guidance는 실제 test 중간 상태를 입력하는 온라인 응답과 목적이 다르다. 따라서 이 문제를 근거로 train-only RL 피드백 캐시 자체를 모두 잘못됐다고 할 수 없다.

근거: [Phase2 runtime](phase2_online_v2_runtime.py), [시간 범위 회귀 테스트](test_phase2_online_v2.py).

### 7 Train 프롬프트의 불필요한 변경

**분류: 새 구현에서의 비교 조건 변경.**

초기 온라인 v2에서 train 프롬프트까지 변경했으나, 사용자가 요청한 것은 기존 train 흐름을 유지하면서 test의 정보 사용과 시간 처리를 수정하는 것이었다.

**수정:** v2.2는 기존 per-lambda 요청의 프롬프트·ontology·schema를 복원하고 새 Pure500 reference group으로 교체한다. 원래 group을 넣었을 때 80개 기존 요청과 정확히 일치하는 회귀 검증을 통과했다. 모든 새로운 응답의 품질이 보장된다는 의미는 아니다.

근거: [요청 복원 테스트](test_phase2_train_prompt_restore.py), [검증 결과](../result/rc101_phase2_online_gpt_train20_v2_2_20261018/train_prompt_verification.json).

### 8 잘못된 group 출력과 실행 상태 기록

LLM의 customer/group 쌍이 요구된 구조와 다르면 검증에서 중단하도록 돼 있다. 잘못된 응답을 조용히 다른 group으로 옮겨 학습하는 방식은 사용하지 않는다. 다만 초기 queue는 실패 후 상태 파일이 running으로 남을 수 있었으므로 프로세스·로그·체크포인트를 함께 확인해야 했다.

Phase2 v2.2의 기존 학습은 사용자 요청으로 중단했고, lambda5의 epoch 689 체크포인트를 확인한 뒤 paused로 기록했다. lambda10 학습이 완료됐다는 뜻은 아니다.

## 새 동시 운행 초안에 생긴 문제

### 9 Solomon 차량 상한 누락

**분류: 동시 운행 v1 최초 초안에서 추가된 오류. 원래 코드에는 상한이 있었다.**

기존 `VRPTWEnv`는 `vehicle_limit`을 읽고 마지막 차량 복귀 후 미방문 고객을 미서비스 페널티로 처리한다. RC101 사용 데이터의 차량 상한은 25대이다.

새 v1 최초 초안은 그 상한을 적용하지 않아 100대까지 사용하는 결과가 나왔다. 이는 기존 알고리즘의 성능 문제가 아니라 초안의 제약 누락이다.

**수정:** 신규 차량 행동에 상한 마스크를 적용하고, 사용 가능한 차량으로 고객을 모두 처리하지 못하면 미서비스 수를 기록하며 기존 계수의 페널티를 부과한다. 최초 무상한 결과는 성능 비교에서 제외한다.

### 10 전역 ADVANCE를 학습 행동에 추가

**분류: 동시 운행 v1에 추가한 부적절한 행동 설계.**

v1에서는 차량 출발이 도착 사건을 예약할 뿐이고, 정책이 ADVANCE를 선택해야 시간이 다음 사건으로 이동했다. 이 때문에 차량 투입과 전역 시간 진행이 같은 categorical action 공간에서 경쟁했다.

사용자가 요구한 배차 담당자의 행동은 차량과 목적지에 대한 운영 결정이며, 환경 시간을 직접 진행시키는 선택이 아니었다.

**수정:** v2에서 전역 ADVANCE를 제거했다. 다만 v2는 미투입 차량의 depot 대기를 시각별로 결정하는 별도 규칙을 사용했으므로, 이후 확정한 v3와도 다르다. ADVANCE를 제거했다는 사실만으로 모든 버전의 정책이 같아지는 것은 아니다.

### 11 기존과 다른 지연 신규 배차

**분류: 운영 조건 차이.**

기존 순차 방식에서는 각 차량이 0분부터 시작했다. v1/v2는 환경 시각이 양수일 때도 신규 차량을 그 현재 시각에 출발시켰다. 과거로 출발시킨 것은 아니지만, 기존과 동일한 출발 조건은 아니었다.

**수정:** v3는 t=0에서만 신규 투입을 허용하며 t>0에서는 이를 마스킹한다. 시간창 대기는 허용하지만 서비스 종료 후 임의 대기는 허용하지 않는다. 모든 차량을 25대로 고정하거나 수요 하한만큼 강제 출발시키는 규칙은 추가하지 않았다.

### 12 행동 후보 개수 및 학습 설정 차이

v1/v2는 새 차량 목적지 후보가 고객 수만큼 존재하는 반면 ADVANCE 또는 depot 선택은 적은 수의 후보였다. 하나의 softmax에서 개별 logit이 같아도 행동 종류별 총 확률은 후보 수의 영향을 받는다. 이 구조가 차량 과다 투입의 유일한 원인이라고 확인된 것은 아니다.

또한 최초 초안은 작은 encoder와 표준편차 정규화 없는 REINFORCE advantage를 사용했다. 이후 capped 실행에서 encoder 크기를 복원했고, v3에서는 기존의 평균 baseline 및 advantage 표준편차 정규화를 복원했다.

다만 새로운 fleet/action decoder는 기존 decoder와 다르고, 현재 진단은 rollout 수·update 수를 줄여 수행했다. 따라서 v3를 포함한 짧은 실행은 기존 1,000 epoch와 동등한 학습 예산의 비교 실험이 아니다.

### 13 실행 중단과 메모리 사용

한 capped 진단 프로세스가 정상 완료 전에 종료됐고 Python 오류 로그는 남지 않았다. OOM으로 확정할 증거는 없다. 이후 여러 rollout의 계산 그래프를 한꺼번에 유지하지 않고, 먼저 sampling한 trajectory를 동일 RNG로 하나씩 재실행해 gradient를 누적하도록 변경했다.

각 replay의 reward와 사건 trace가 원본 sampling과 일치하는지 검사한다. checkpoint는 매 update 저장한다. 이 변경은 메모리 보관 방식의 수정이지 reward 가중치 변경이 아니다.

## 현재 v3의 명시적인 규칙과 검증 범위

1. t=0에서 새 차량의 첫 고객을 선택하면 도착 사건을 예약하고 t=0을 유지한다.
2. 기존 차량의 다음 의사결정을 선택하면, 도착·시간창 대기·서비스 완료·사고 공개를 순서대로 처리한다. 첫 전환의 기존 차량 후보는 가장 이른 서비스 완료 차량으로 제한해 다른 차량을 임의로 기다리게 하지 않는다.
3. 목적지는 갱신된 시점의 상태에서 다시 선택한다. 미래 시점의 목적지를 사고 공개 전 상태로 확정하지 않는다.
4. t>0에서는 새 차량 추가와 차량 선택을 마스킹한다. 환경이 가장 이른 서비스 완료 차량을 지정하며, 동시 완료는 차량 ID 순서로 동일 시각에 처리한다.
5. 모든 고객을 처리하기 전에 출발시킨 차량들이 전부 복귀하면 종료하고 미서비스를 명시한다. 상한 이내라는 것만으로 feasible이라고 표시하지 않는다.
6. 실제 이동시간은 기존처럼 구간 출발 시 확정한다. 이동 도중 새 사고로 이미 진입한 구간의 잔여시간을 재계산하는 모델은 아니다.

검증 코드: [v3 테스트](../experiments/concurrent_fleet_v3/test_t0_dispatch.py).

자동 테스트 7개에서 t=0 동시 배차, t>0 신규 투입 차단, 서비스 종료 순서, 목적지 선택 전 사고 공개, 동시 완료 처리, 미서비스 페널티, 시간 단조 증가 및 backward를 확인했다. lambda5/10 각 5 update의 실제 데이터 실행도 완료했다.

아직 짧은 진단이며 초기 정책은 고객을 충분히 서비스하지 못한다. 학습 수렴, 전체 test80 성능, LLM 연결, LKH-3의 동시 운행 평가 일치까지 검증된 상태는 아니다.

## 결과 해석 원칙

- 기존 best-of-many 평가와 one-shot 평가를 같은 프로토콜로 취급하지 않는다.
- 기존 학습 버그 수정, 첫 고객 학습, 동시 운행, 초기 배차 규칙, decoder 변경의 효과를 한 원인으로 단정하지 않는다.
- lambda가 다르면 목적함수가 달라지므로 raw reward만으로 lambda를 선택하지 않는다.
- 미서비스가 있는 결과는 별도 표시하고 차량 수·이동시간 감소를 성능 개선으로 해석하지 않는다.
- 경로만 저장해서 각 차량을 다시 0분 출발로 계산하면 지연 출발 버전의 시간을 재현할 수 없다. 버전별 출발 시각과 사건 trace를 사용한다.
- 시뮬레이터는 실제 이동시간과 이벤트 스케줄을 알아야 하지만, 정책과 LLM에는 공개 시점 이전의 이벤트 정보를 주지 않는다. 단위 테스트만으로 모든 정보 누출 경로가 검증됐다고 주장하지 않는다.
- 현재까지 반복 확인한 test80은 탐색적 평가 자료로 구분하고 최종 설정 선택의 validation과 최종 test를 분리한다.

## GPU 본 학습 구조 정정: v4

v3 진단용 CPU 시뮬레이터를 그대로 본 학습에 확장한 것은 구현상의 잘못이었다. 이후 모델만 CUDA로 옮긴 하이브리드 버전도 환경을 Python 반복문으로 처리하고 100 rollout씩 gradient를 누적했으므로 기존 GPU 배치 처리 성능을 유지하지 못했다. 두 실행은 체크포인트를 보존한 채 중단했다.

v4에서는 차량 상태, 고객 예약/서비스 마스크, 전역 시간, 차량 선택, 실제 이벤트 이동시간, reward 계산을 CUDA 텐서로 처리한다. 같은 인스턴스의 batch 128 x rollout 100을 함께 처리하며, 마지막 batch는 기존처럼 68이다. 루프 제어 및 로그 저장은 CPU에서 하지만 개별 차량/rollout의 상태를 CPU로 가져와 시뮬레이션하지 않는다.

또한 v1-v3에서 추가했던 fleet MLP/scorer를 제거하고 기존 `VRPTWModel`의 encoder, attention decoder, slack head를 그대로 재사용했다. 파라미터 이름과 크기는 기존 모델과 동일하다. 단, 새로운 배차/기존 차량 행동을 점수화하는 행동 매핑은 필요하므로 정책 함수 자체까지 기존 순차 라우팅과 동일하다고 주장하지 않는다. 기존 event token을 공개 시각에 따라 선택하며, 미래 사고 인코딩이 현재 행동에 사용되지 않는지 검사한다.

학습 설정은 기존 train20의 config를 사용한다: 1000 epoch, epoch당 2500 episode, episode당 100 rollout, Adam 및 기존 learning rate, AMP/GradScaler, rollout별 mean/std baseline, gradient clipping 1. 상태를 같은 정책으로 재계산해 decoder gradient를 누적하고 encoder에 한 번 전달하는 메모리 관리 경로는 full-graph gradient와 수치 허용오차 내 일치함을 검사했다. AMP overflow에 따른 optimizer step 생략은 `optimizer_step_skipped`로 기록한다.

- 구현: [GPU 환경](../experiments/concurrent_fleet_v4/env.py), [기존 모델 재사용](../experiments/concurrent_fleet_v4/policy.py), [학습 실행](../experiments/concurrent_fleet_v4/train.py).
- 검증: [테스트 6개](../experiments/concurrent_fleet_v4/test_tensor.py). CPU/CUDA 환경 일치, 실제 RC101 rain/accident 일치, 기존 decoder 일치, gradient 일치, 사고 공개 전 입력 차단을 포함한다.
- 전체 batch 8회 검증에서 정상 optimizer 업데이트를 확인했다. 초기 batch 계산은 약 2~3.5초였으며 전체 학습/평가 시간의 보장은 아니다.
- 본 학습: `result/concurrent_fleet_v4_cuda_train20_20261018`, lambda5 이후 lambda10 순차 실행. 각 1000 epoch 후 test80 one-shot 평가. 초반 20 update는 매번, 이후에는 20 update마다 고정 train probe 및 배차 이력을 기록한다.

## v5: 불필요한 학습 구조 변경 철회

사용자의 요구는 동시 운행 의사결정으로 바꾸되 나머지는 기존 학습 구조를 유지하는 것이었다. v4의 상태 재계산, step별 backward, encoder 결과 공유는 필수 변경이 아니었으며 임의로 추가한 것이다. v4는 epoch 3, 56 batch 처리 시점에서 checkpoint를 보존하고 중단했다. 그 결과를 기존 방식과 동일한 조건의 결과로 취급하지 않는다.

| 항목 | v5 처리 |
|---|---|
| 계산 그래프 / backward | 전체 경로의 그래프 유지, batch당 단일 backward. 재계산 및 분리된 encoder backward 제거 |
| Encoder batch | 실제 batch 128개를 인코딩. 단일 instance encoding 공유 제거 |
| Decoder | 기존 attention / slack head 파라미터 재사용. fleet MLP 추가 없음 |
| AMP | 전체 decoding loop를 autocast로 감싸고 기존 GradScaler 사용. FP32 mask/softmax 승격 유지 |
| 환경 수치형 | 기존처럼 FP32. v3 CPU double 기준과 비교할 때 FP32 허용오차 적용; 고객 상태 및 행동 마스크는 일치 검사 |
| 샘플링 / loss | softmax 후 multinomial, 선택 확률의 log 합, episode별 rollout 평균/std baseline, batch 전체 mean loss |
| Optimizer | 기존 Adam 설정, clip 1, batch당 한 번 step. AMP overflow 시 skip 기록 |
| 학습량 / seed | train20, 1000 epoch, 2500 episode/epoch, 128 batch(마지막 68), 100 rollout, 기존 seed helper |
| 기록 / checkpoint | epoch reward는 기존처럼 rollout 최대 reward의 episode 가중 평균. loss도 episode 가중 평균. 기존 `model_state_dict`, `optimizer_state_dict`, `epoch`, `reward_log`, `loss_log` 키 복원 |

남기는 변경은 동시 차량 상태, 전역 시계, t=0 신규 배차와 기존 차량 선택의 행동 매핑, 예약 고객 마스킹, 그리고 이를 위한 GPU 환경이다. 전역 시각에 맞춰 공개된 이벤트의 encoding을 선택한다. 단, 경로 구성 방식과 행동 공간이 달라지므로 기존 순차 모델과 정책 함수까지 동일하다고 주장하지 않는다. 무효한 신규 배차 분기가 batch 전체에서 사라지면 그 분기만 계산하지 않으며, 유효 행동의 그래프를 버리거나 재계산하지 않는다.

추가 로그, atomic checkpoint와 중간 재시작 상태는 실험 추적용으로 유지한다. 학습 split과 lambda별 reward config를 변경하지 않았다. Test는 이전에 합의한 one-shot을 유지하며, 옛 best-of-many 평가로 되돌리지 않는다. 기존 경로의 최초 depot 이동 누락도 재도입하지 않는다.

검증: [v5 테스트](../experiments/concurrent_fleet_v5/test_tensor.py) 6개 통과. 단일 backward 호출과 실제 encoder batch 크기를 검사하고, 실제 RC101 rain/accident에서 시간 및 reward의 FP32 수치 일치와 고객 상태/행동 마스크 일치를 검사했다. 전체 batch 8회 실행 및 checkpoint 재로딩도 확인했다. AMP는 초반 scale 조정으로 5회 skip 후 정상 업데이트했다. 검증 중 최대 allocated GPU memory는 약 33,178 MiB였으므로 사고 사례에서는 물리 VRAM 용량 근처/초과에 따른 느려짐이 생길 수 있다. 임의로 batch를 낮추거나 재계산 방식으로 대체하지 않았다.

본 학습 경로: `result/concurrent_fleet_v5_fullgraph_train20_20261018`. seed 42에서 새로 시작하며 lambda5 이후 lambda10을 순차 실행한다. 이전 v4 checkpoint를 혼합하지 않는다. 구현은 [학습](../experiments/concurrent_fleet_v5/train.py), [모델 어댑터](../experiments/concurrent_fleet_v5/policy.py), [환경](../experiments/concurrent_fleet_v5/env.py)에 분리 보존했다.
