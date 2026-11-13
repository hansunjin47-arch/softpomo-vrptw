# SoftPOMO-VRPTW: 결과 경로 정리

프로젝트 루트: `soft_pomo/`

## Pure RL (Phase 1, 1000 epoch, batch32)

```
result/config_E_{r1,rc1,c1}_final_weightedsum_b32_aug8/{r102-r112,rc102-rc108,c102-c109}/
    checkpoint-500.pt    -- Phase 2 시작점
    checkpoint-1000.pt   -- Pure RL 최종 결과
```
로그: `logs/phase1_retrain_{r1,rc1,c1}_out.log`

## Phase 2 v2 (LLM bias, slack_head 아키텍처 통일 + 평가 aug×8, batch64)

```
result_soft/{r102-r112,rc102-rc108,c102-c109}/
    checkpoint-1000.pt
```
로그: `logs/phase2_v2_{r1,rc1,c1}_out.log`

## Phase 2 matched (LLM bias, batch32로 Pure RL과 맞춤, `_active_route_cluster` 클러스터 앵커링 반영)

```
result_soft_matched/with_bias/{r102-r112,rc102-rc108,c102-c109}/
    checkpoint-1000.pt   -- 가장 최신 결과
```
로그: `logs/matched_with_{r1,rc1,c1}_out.log`

※ `result_soft_matched/no_bias/`는 실행하다 중단함 (Pure RL 1000과 중복이라 판단) — 체크포인트 없음, 무시할 것.

## LLM confidence 캐시

```
result_soft/llm_cache_final_nomisfit/   -- 현재 사용 중 (misfit gate OFF)
result_soft/llm_cache/                  -- 구버전 (misfit gate ON, 미사용)
```

## 각 파일에서 뭘 보면 되는지

- 학습 진행(`Epoch N/1000 train_reward=...`), `[Final Eval] test_reward=...`, `BEST SOLUTION` 블록
  (vehicles/served/late_stops/total_late/reward, base→ACC2→RAIN2 순서)은 전부 위 `_out.log` 파일에
  있음.
- 실제 라우트/솔루션 파일: 각 체크포인트 디렉토리의 `plots/{인스턴스명}_solution.txt`,
  `{인스턴스명}_best_route.png`.

## 지금까지 나온 세 버전 요약 (R1/RC1 기준, C1은 세 버전 다 동일)

| | Pure RL | Phase2 v2 (batch64) | Phase2 matched (batch32) |
|---|---|---|---|
| 격차 원인 | 기준선 | slack_head 없음 + 평가예산 8배 부족 (이후 수정) | batch_size까지 맞춤 |
| 결과 | late 3~8건 | late 10~11건 | R1은 더 나빠짐(late 23~25), RC1은 항목별 혼재 |

batch_size를 맞춰도 격차가 깔끔하게 안 줄어들어서, 단일 실행 간 RL 학습 분산 문제일 가능성 있음 —
결론 내리기 전 반복 실행(시드 변경) 필요.
