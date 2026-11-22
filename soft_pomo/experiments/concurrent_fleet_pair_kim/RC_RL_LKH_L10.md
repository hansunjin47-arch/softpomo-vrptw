# RC101: RL + LKH-3 예시, 지각 가중치 10

기존 C/R 실행 코드를 바꾸지 않는 별도 실험이다. 동시 routing, 차량 100·이동시간 1·지각 10, NEW bias 5·routing beta 5, 용량/복귀 mask 및 advantage std ON. 학습률은 500 epoch까지 1e-4, 이후 선형 감소하여 1000 epoch에 1e-5. Seed 44·45·46 각각 처음부터 1000 epoch 학습한다.

## 다른 PC에서 실행

기존 저장소에서 작업한 변경은 먼저 보존하고 `code-sync-20261113` 브랜치를 갱신한다. 기존 학습 프로세스의 코드를 실행 중에 갱신하지 말고, 종료 후 또는 별도 checkout에서 실행한다.

```powershell
git switch code-sync-20261113
git pull --ff-only origin code-sync-20261113
cd soft_pomo
python -m experiments.concurrent_fleet_pair_kim.rl_lkh validate --late 10
powershell -ExecutionPolicy Bypass -File .\scripts\run_rc_rl_lkh_l10.ps1 -Python python
```

Python은 기존 실험용 CUDA/PyTorch 가상환경을 사용한다. 필요하면 `-Python C:\경로\.venv\Scripts\python.exe`로 지정한다. 키가 환경변수에 없으면 OpenRouter API 키를 숨김 입력으로 한 번 받는다. 키는 Git에 저장하지 않는다. 실행 터미널은 학습이 끝날 때까지 열어 둔다. 다른 GPU 학습이 끝난 뒤 실행한다.

스크립트는 BASE 1 + Train 20개 캐시를 생성/검증한 뒤, seed44 →45 →46 순으로 학습한다. 정상 응답은 재사용하며 요청별 실패 재시도는 최대 3회다. 세 seed는 동일 캐시를 공유한다. 중단 후 같은 스크립트를 다시 실행하면 저장된 체크포인트에서 재개하고 완료된 seed는 건너뛴다. 오류가 발생하면 다음 seed를 시작하지 않는다.

## 예시와 정보 공개

- 모델: `openai/gpt-5.4`, 기존 request template 사용.
- 그룹 최고 confidence 고객의 NEW 선택에 +5가 적용된다는 수정 설명을 포함한다.
- 기존 독립 RAIN 예시에서 Pure RL과 LKH-3의 route·서비스 시작 시각·지각·차량별 이동시간을 함께 제공한다.
- LKH-3는 Train20 이력으로 구한 고정 계획을 독립 예시에 재평가한 값이다. 새 대상이나 Test80의 해답이 아니다.
- 두 예시의 보고된 성능은 지각 가중치 5의 역사적 결과다. 대상은 10이며, 프롬프트에서 이를 명시한다. 예시를 λ10으로 다시 최적화했다고 주장하지 않는다.
- ACC는 사고 전 BASE, 사고 후 event confidence를 공개 시점부터 적용한다. Train 캐시는 offline이며 실시간 차량 상태를 포함하지 않는다.

데이터·예시·설정은 `experiment_inputs`에 포함되어 있으므로 서버 다운로드나 체크포인트 복사가 필요 없다. 생성 캐시는 `cache/rc_rl_lkh_v100_l10`, 결과는 `result/rc_rl_lkh_v100_l10_s44`(45/46 동일)에 저장한다. 기존 λ5 캐시는 재사용하지 않는다.

**학습 완료와 Test 완료는 다르다.** 이 명령은 학습만 수행한다. `run/late10/training_complete.json`은 1000 epoch 완료를 뜻하며, 최종 causal Test80은 별도 평가가 필요하다. Test에서도 이 모듈의 `enrich`로 같은 비교 예시와 λ10 설정을 제공하고, ACC 공개 시점의 실제 상태로 API를 호출해야 한다. 기존 평가 스크립트를 설정 변경 없이 사용하지 않는다.

검증: `python -m unittest analysis.test_rc_rl_lkh_portable analysis.test_pair_kim_rl_example`. 테스트는 모의 응답을 사용하므로 API 비용과 GPU 학습이 발생하지 않는다.
