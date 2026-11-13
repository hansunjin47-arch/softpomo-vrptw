# Concurrent Fleet: MDP Definition and Implementation Changes

기준일: 2026-10-18. 대상: 로컬 `experiments/concurrent_fleet_v5/` 구현.

이 문서는 동시 차량 운행으로 전환한 episode의 정의와 기존 학습에서 유지하는 항목을 기록한다. 실험 성능 보고서나 완료된 논문 검증을 의미하지 않는다. 이번 Git 변경은 문서만 포함하며, 아래 구현 경로는 로컬 작업 트리 기준이다. 관련 코드 및 실험 결과가 이 문서와 함께 원격에 게시됐다는 뜻은 아니다.

## 1. 변경 범위

변경 목적은 차량별로 경로 전체를 구성하고 다음 차량의 시계를 0으로 되돌리는 방식 대신, 여러 차량이 공통 시계 위에서 동시에 운행하게 만드는 것이다.

- 변경: 환경 상태, 의사결정 시점, 차량 투입/목적지 행동, 전이, 행동 마스크, 모델과 환경 사이의 행동 매핑.
- 유지: 기존 encoder/attention decoder/slack head 파라미터 구조, REINFORCE, multi-rollout, reward 계수, optimizer, batch 구성, 전체 경로 단일 backward.
- 정책 함수 자체가 동일한 것은 아니다. 새 행동 공간에 기존 decoder 점수를 매핑하는 어댑터는 달라진다.
- 기존 최초 depot 이동 누락을 되살리거나, 합의한 one-shot 평가를 옛 best-of-many 방식으로 되돌리는 변경은 아니다.

## 2. 문제와 가정

고객 집합을 V={1,...,N}, depot을 0, 차량 상한을 K_max, 용량을 Q로 둔다. 차량 상한과 용량은 Solomon instance에서 읽는다. RC101은 K_max=25이다.

- 고객 위치, 수요, time window, 서비스 시간은 사전에 알려져 있다.
- 용량은 hard constraint이며 고객 수요는 분할하지 않는다.
- time window는 soft constraint다. 일찍 도착하면 기다리고, 늦게 도착하면 지각 비용을 부과한다.
- 각 투입 차량은 single trip으로 운행하며 depot 복귀 후 퇴역한다. 재출발하지 않는다.
- 신규 차량은 공통 시각 t=0에서만 투입할 수 있다. 최소 차량 수를 강제하지 않는다.
- 구간 이동시간은 구간 출발 시점의 실제 이벤트 효과로 확정한다. 이미 이동 중인 구간의 잔여 이동시간을 후속 사고로 다시 계산하지 않는다.

## 3. 상태와 관측

Simulator state X_k에는 다음을 포함한다.

| 구성 | 내용 |
|---|---|
| 공통 시계 | 현재 물리 시각 t_k |
| 고객 상태 | 미배정 0 / 예약됨 1 / 서비스 완료 2 |
| 차량 상태 | 미투입, 이동 중, 서비스/대기 중, 다음 행동 가능, depot 복귀 후 퇴역 |
| 차량별 값 | 출발 노드, 목적지, 잔여 용량, 출발 시각, 실제 도착 시각, 서비스 완료 예정 시각 |
| 차량 수 | 현재까지 투입한 K, instance 상한 K_max |
| 누적 비용 | 실제 이동시간, 거리, 지각 시간, 지각 횟수 |
| 외생 이벤트 | 실제 교통 이벤트 스케줄과 해당 효과, 공개 시각 |

Simulator는 실제 이동시간과 미래 이벤트 스케줄을 갖지만, 정책에 이를 그대로 전달하지 않는다. 정책 관측 O_k는 해당 시각까지 공개된 정보로 제한한다. Rain forecast 등 시작 전에 알려진 정보와 사고 발생 후 알려지는 정보는 공개 시각으로 구분한다.

현재 adapter가 사용하는 입력은 기존 node/event encoding, 선택 대상 차량의 현재 노드와 잔여 용량, 공통 시각, 기존 slack feature, 유효 행동 마스크다. 전체 fleet state를 별도 MLP로 입력하는 구조는 사용하지 않는다. Simulator가 갖는 모든 상태가 정책 입력이라는 뜻은 아니다.

**형식적 주의:** 숨겨진 이벤트를 포함한 충분한 simulator state에서 전이를 정의할 수 있지만, 정책 관측이 Markov sufficient하다는 것은 별도 문제다. 미래 사고가 숨겨지고 정책이 일부 상태만 받으므로, 정책 관점에서는 부분관측 문제로 설명하는 것이 정확하다. 관측만으로 완전관측 MDP라고 단정하지 않는다.

## 4. 행동 공간

### t=0

- `NEW(j)`: 미투입 차량 한 대를 활성화하고 미배정 고객 j로 즉시 출발시킨다.
- `SELECT_EXISTING(i*)`: 다음 의사결정 대상 기존 차량을 선택한다. i*는 환경이 가장 이른 서비스 완료 차량으로 지정하며, 임의의 모든 차량 중 하나를 고르는 구조가 아니다.
- 현재 시각에 이미 ready 차량이 있으면 그 차량의 고객/depot 목적지 선택도 가능하다. 영거리 이동이나 서비스 시간 0의 경계 사례를 포함한다.

`NEW(j)` 후에도 t=0에서 가능한 의사결정이 남아 있으면 시계가 즉시 미래로 넘어가지 않는다. 따라서 차량 A와 B를 같은 시각에 출발시킬 수 있다.

`SELECT_EXISTING`은 아직 도착하지 않은 차량의 다음 목적지를 미리 확정하지 않는다. 환경이 해당 의사결정 시각까지 전이한 뒤, 갱신된 관측에서 목적지를 다시 선택한다. 독립적인 `ADVANCE`, 대기 시간 길이 선택, 별도 초기 배차 종료 버튼은 없다.

### t>0

- 신규 차량 추가는 금지한다.
- 환경이 가장 이른 서비스 완료 차량을 지정하고, 정책은 그 차량의 다음 고객 또는 depot을 선택한다.
- 서비스 완료 직후 출발한다. 같은 시각에 여러 차량이 ready이면 차량 ID 순서로 결정하며 시계는 같은 값을 유지한다.

### 구현상의 action slot

0-based slot을 사용한다.

| Slot | 의미 |
|---|---|
| 0 ... N-1 | NEW(customer=slot+1) |
| N ... 2N | 지정된 기존 차량의 destination=slot-N; 0은 depot |
| 2N+1 | SELECT_EXISTING |

총 2N+2 slot 중 해당 상태에서 유효한 행동만 활성화한다. N=100이면 202개이며, 모든 slot이 항상 선택 가능한 것은 아니다.

## 5. 고객 예약과 마스킹

고객은 **차량의 목적지로 확정되는 순간** 상태 0에서 1로 전이한다. 도착 전에도 모든 차량의 후보에서 제외한다. 서비스 완료 시 1에서 2가 된다.

- 미배정이고 해당 차량의 잔여 용량으로 서비스 가능한 고객만 선택 가능하다.
- 예약 고객과 완료 고객은 다른 차량 및 신규 차량에서 모두 마스킹한다.
- 신규 차량은 t=0, K<K_max인 경우에만 허용한다.
- 잔여 용량은 고객을 예약할 때 차감한다.
- 고객 상태와 마스크는 rollout별 독립 상태다. 한 rollout의 예약이 다른 rollout을 제한하지 않는다.
- 이번 버전에는 예약 취소/고객 재배정 행동이 없다.

## 6. 시간 전이와 의사결정 시점

의사결정 번호 k는 고정 길이의 물리 time step이 아니다. 여러 행동이 t=0에 일어날 수 있고, 다음 행동까지 흐르는 시간도 상태에 따라 달라진다. 따라서 event-driven 의사결정으로 설명하며, 엄밀한 시간 모델링에서는 SMDP 관점도 구분할 수 있다.

출발 시각 t에 노드 i에서 j로 이동하면:

```text
arrival = t + base_travel_time(i,j) * multiplier(t,i,j)
service_completion = max(arrival, opening(j)) + service_time(j)
```

활성 이벤트 중 구간의 양 끝 노드 중 하나라도 영향 지역에 속하면 해당 multiplier를 적용하고, 겹치는 경우 최대 multiplier를 사용한다. 이 실제 multiplier는 simulator 전이용이며 정책 입력으로 직접 제공하지 않는다.

도착 시 실제 이동시간/거리 및 지각을 누적하고, 서비스 완료 시 고객을 완료 처리한다. Depot 도착은 차량 퇴역이다. 다음 의사결정까지의 도착과 서비스 완료를 반영한 뒤 정책을 호출한다.

중간에 사고 공개 시각을 지났다면 다음 목적지 결정에서는 그 정보가 반영된다. CUDA 환경은 정책 행동이 없는 중간 공개 이벤트마다 별도 Python callback을 실행하지 않고, 현재 시각으로 event encoding의 공개 단계를 선택한다.

시계는 뒤로 가지 않는다. 배차 시각을 소급해 t=0으로 설정하거나, 한 차량의 미래 경로 전체를 완료한 뒤 다른 차량을 과거 시각에서 처리하지 않는다.

## 7. Reward와 종료

새로운 차량 투입별 추가 reward shaping은 넣지 않았다. 기존 terminal objective를 유지한다. 현재 RC101 BKS-normalized 설정은 다음과 같다.

```text
J = K / 14 + (T_drive + lambda * L) / 1696.95 + 500 * U / N
R_terminal = -J
```

- K: 실제 투입 차량 수.
- T_drive: depot 복귀를 포함한 실제 운전시간. 대기 및 서비스 시간은 제외한다.
- L: 고객 및 depot 도착 지각 시간의 합.
- U: 서비스하지 못한 고객 수. 실행 종료 시 예약 고객이 남는 정상 경로는 허용하지 않는다.
- 위 식은 해당 RC101 config의 단순화다. 일반 구현은 기존 config의 거리/지각 횟수 등 계수를 읽으며, 이 설정에서는 해당 계수가 0이다.

학습은 terminal reward를 사용하며 중간 행동마다 별도의 차량 투입 penalty를 추가하지 않는다. 기존 차량 비용은 최종 K 항으로 반영된다.

모든 투입 차량이 depot으로 복귀하고 더 진행할 수 없으면 종료한다. 일부 고객이 미서비스인 채 종료될 수 있으며 이때 U penalty를 부과한다. `feasible`은 U=0을 뜻하며, soft-TW 지각이 반드시 0이라는 뜻은 아니다. 차량 수가 적더라도 미서비스가 있으면 좋은 결과로 해석하지 않는다.

## 8. 그대로 유지하는 학습 구조

현재 train20 실험의 기준 설정:

| 항목 | 설정 |
|---|---|
| Epoch | 1000 |
| Episodes per epoch | 2500 |
| Batch | 128, 마지막 batch 68 |
| Rollouts per episode | 100 |
| 첫 고객 | 강제 시작점이 아니라 정책이 선택 |
| Encoder | batch 128개를 실제 인코딩; 단일 encoding 공유하지 않음 |
| Decoder | 기존 attention decoder 및 slack head 재사용; 필요한 행동 매핑만 변경 |
| Baseline | episode 내 100 rollout reward 평균, advantage std 정규화 |
| Loss | 전체 경로 선택 확률의 log 합에 advantage를 곱한 REINFORCE 평균 loss |
| Backward | 전체 경로 그래프 유지, batch당 한 번 |
| Optimizer | 기존 Adam config, gradient clipping 1 |
| Precision | 환경 FP32, 기존 autocast/GradScaler 적용 |
| Seed | 기존 seed helper, seed 42 |

기존 코드처럼 batch마다 학습 pool에서 instance를 선택한다. 서로 다른 rollout은 독립적인 경로와 고객 예약 상태를 갖는다. 모델 파라미터는 공유한다. 동일한 목적지나 첫 고객이 여러 rollout에서 선택되는 것은 금지하지 않는다.

학습 reward log의 기본 epoch 지표는 기존처럼 각 episode의 최대 rollout reward를 batch 크기로 가중 평균한다. 진단용 전체 rollout 평균 reward도 별도로 기록한다. 이 두 수치를 혼동하지 않는다.

AMP overflow로 optimizer step이 생략된 batch는 별도로 기록한다. 이를 정상적인 파라미터 업데이트 횟수와 동일하게 해석하지 않는다.

## 9. 평가와 현재 범위

- Test는 instance마다 한 번의 greedy 실행이다. 실현된 미래 reward를 보고 여러 경로 중 best를 고르지 않는다.
- 초기 양상 확인은 고정된 train 사례의 greedy probe와 sampled rollout 통계를 함께 사용한다.
- 현재 구현은 pure RL이다. 이 동시 운행 환경에서 LLM 사고 재호출이나 proposed model 전체가 구현/검증됐다는 뜻은 아니다.
- LKH-3 비교도 같은 사건 공개 및 온라인 의사결정 조건인지 별도로 확인해야 한다.
- 초기 25대 연속 투입은 관측된 학습 양상이지, 25대를 강제로 투입하는 규칙이 아니다.

## 10. 검증 및 버전 구분

로컬 구현 위치:

```text
experiments/concurrent_fleet_v5/env.py
experiments/concurrent_fleet_v5/policy.py
experiments/concurrent_fleet_v5/train.py
experiments/concurrent_fleet_v5/test_tensor.py
```

테스트는 CPU 기준 환경과 CUDA 환경의 행동 마스크/고객 상태, 시간/reward의 FP32 허용오차 내 일치, 실제 RC101 rain/accident 사례, 미래 사고 관측 차단, 기존 decoder/파라미터 구조, 실제 encoder batch 크기 및 단일 backward를 확인한다. 모든 일반화 성능 또는 모든 가능한 정보 누출을 증명한 것은 아니다.

v1-v3의 별도 fleet MLP와 CPU 중심 진단 구조, v4의 step별 재계산/backward 및 encoder 결과 공유는 현재 v5 본 학습에 사용하지 않는다. 과거 결과를 삭제하지 않고 버전을 분리한다. 이 변경은 episode의 MDP/환경을 바꾸는 것이지 별도의 학습 알고리즘을 제안하는 것이 아니다.
