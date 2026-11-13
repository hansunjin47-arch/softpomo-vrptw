# LLM 캐시 경로 정리 (TW-close 정렬 / OpenRouter 테스트)

프로젝트 루트: `soft_pomo/`

## 1. C1/R1/RC1 TW_close 정렬 프롬프트 캐시 (전체 재생성, Ollama/gpt-oss:20b)

```
result_soft/llm_cache_twclose_c1/    -- C101~C109 base+RAIN2, ACC2 3-wave (18+27개 파일)
result_soft/llm_cache_twclose_r1/    -- R101~R112 base+RAIN2, ACC2 3-wave (24+36개 파일)
result_soft/llm_cache_twclose_rc1/   -- RC101~RC108 base+RAIN2, ACC2 3-wave (16+24개 파일)
```
- 기존 프로덕션 캐시(`result_soft/llm_cache_final_nomisfit/`)와 별도 경로 — 프로덕션은 안 건드림
- 프롬프트: 클러스터 내 고객을 TW_close 오름차순으로 정렬 + "이 순서는 방문 순서가 아니다" 5개항 disclaimer 포함 버전
- 생성 로그: `logs/cachegen_twclose_{c1,r1,rc1}_out.log` (본생성) / `_fix_out.log` (온도 정정 재생성분)
- 145개 결과 전부 temperature=0.0으로 통일 완료 확인, 에러/fallback 없음

**온도 정정 이력 (2026-09-17)**: 최초 생성 중 4건(RC103 base, C103_ACC2/C104_ACC2/RC108_ACC2 각 웨이브 1개)이
thinking-overflow 재시도 또는 outer-retry 임시 수정으로 인해 temperature=0.15로 생성됐던 것을 발견 → 전부
placeholder 리셋/파일 삭제 후 `--refresh-cache` 없이 재실행하여 순수 0.0으로만 재생성(나머지 141개는 스킵,
불필요한 재작업 없음). 이 과정에서 RC103이 0.0에서 결정론적으로 실패하는 근본 원인(모델이 마지막 닫는 `}` 를
빠뜨리고 응답을 끝내는 패턴)을 발견 → `parse_all_clusters_confidence`에 "닫는 괄호 누락 시 최대 3개까지 보충 후
재파싱" 순수 문법 보정을 추가(`SoftClusterLLMModule.py`, 데이터 조작 없음, completeness 검증은 기존과 동일하게
적용)하여 해결. `_ensure_soft_cluster_cache`의 재시도 횟수도 5→20으로 늘림(temperature는 계속 0.0 고정).

## 1.5 Pure RL 500 기반 + Priority Analysis 프롬프트 수정 캐시 (claude-sonnet-4.6, 2026-09-19~20)

```
result_soft/llm_cache_r1_conf_sonnet46/   -- R101~R112 base+RAIN2(24개) + ACC2 3-wave(36개)
result_soft/llm_cache_rc1_conf_sonnet46/  -- RC101~RC108 base+RAIN2(16개) + ACC2 3-wave(24개)
```
- 클러스터 구조(멤버십)는 `llm_cache_twclose_{r1,rc1}/`(Pure RL checkpoint-500 기반, extract_rl_routes.py 결과)에서 그대로 복사해와 재사용 — 구조는 안 건드리고 confidence만 새로 생성
- kim2006 클러스터링이 아니라 **Pure RL checkpoint-500**(`result/config_E_{r1,rc1}_final_weightedsum_b32_aug8/`)에서 바로 이어감 — R1/RC1 둘 다 kim500 대비 late_stops/total_late이 같거나 더 낮아서 Pure RL 쪽으로 결정
- `SoftClusterLLMModule.py`의 Priority Analysis 지시문 수정 반영: "RL visit order는 현재 solution일 뿐 존중할 기본값 아님, slack<0인 stop은 순서 무관하게 무조건 최우선(복수면 더 나쁜 쪽이 위), 과도한 boost로 다른 stop을 새로 지각시키지 말 것" + 클러스터 내 방문순서를 보여주는 `order` 컬럼 추가
- 검증: R1 12/12, RC1 28/28 음수-slack 클러스터 전부 정확히 최우선 처리 확인(다중-음수 클러스터 상대순위도 100% 정확). depot은 여전히 전부 0.0(별개 미해결 이슈)
- (참고: 중간에 confidence 대신 순서 자체를 출력시키는 `ranking` 스키마로 실험했다가 오히려 부정확해져서 폐기, confidence 스키마로 복귀함 — `SoftClusterLLMModule.py`에 fallback 형태로만 남아있음)
- 이 캐시로 R1 Phase 2 학습 진행 중(batch32, `result_soft_r1_conf_sonnet46/`), RC1은 R1 완료 후 이어서 진행 예정

## 2. OpenRouter 연동 테스트 결과물 (C101 단일 인스턴스, 1회성 검증용)

```
result_soft/_openrouter_test_c101.json          -- OpenRouter (openai/gpt-oss-20b) 호출 결과
result_soft/_openrouter_test_c101_sonnet46.json -- OpenRouter (anthropic/claude-sonnet-4.6) 호출 결과
result_soft/_ollama_test_c101.json              -- 동일 조건, 로컬 Ollama (gpt-oss:20b) 호출 결과 (비교용)
```
- 클러스터 구조는 기존 `result_soft/llm_cache_final_nomisfit/C101_rl_cluster.json`의 RL 클러스터 배정을 그대로 재사용 (confidence 값 자체는 재사용 안 하고 새로 호출)
- 프로덕션 캐시가 아닌 임시 검증 파일 — 학습 파이프라인에서 참조되지 않음, 필요 없어지면 삭제해도 무방
- 결과 요약: 3개 백엔드/모델(로컬 gpt-oss:20b, OpenRouter gpt-oss-20b, OpenRouter claude-sonnet-4.6) 전부 10개 클러스터 노드 순위(order) 완전 일치, depot score 전부 0.0 동일. 절대 confidence 값만 모델/백엔드별 소폭(±0.05~0.34) 편차. claude-sonnet-4.6은 reasoning overflow 없이 1차 시도에 즉시 성공(eval_tokens=1201), gpt-oss-20b는 OpenRouter 경로에서 유독 재시도가 더 자주 발생(온도=0에서 장황한 reasoning 경향)

## 3. OpenRouter 연동 코드

```
VRPTWLLMModule.py
    OPENROUTER_URL, OPENROUTER_API_KEY (os.environ에서 읽음)
    _query_openrouter()  -- 신규 헬퍼, Ollama query_llm()과 동일한 재시도/온도-escalation 설계
    query_llm()           -- model 문자열에 "/" 있으면 OpenRouter로 자동 라우팅, 없으면 기존 Ollama 경로 그대로
```
- API 키는 Windows 사용자 영구 환경변수(`OPENROUTER_API_KEY`)로 등록 완료 — 새 세션/터미널부터 자동 적용
- 사용법: `--model <vendor>/<model-name>` (예: `openai/gpt-oss-20b`, `anthropic/claude-sonnet-4.6`) 형태로 넘기면 자동 라우팅

### P0 응답 검증 하드닝 (2026-09-17 추가)

외부 검토 문서("OpenRouter 연동 보완 지시서")를 코드와 대조 검증한 결과, 아래 실제 버그를 확인하고 수정함:
- `finish_reason=='length'`인데 content가 비어있지 않으면 그대로 성공 반환되던 버그 → length는 content 유무와 무관하게 항상 불완전 처리, 재시도 소진 시 raise
- `choices` 없는 에러 응답이 `finish_reason` 기본값 `'stop'`으로 떨어져 빈 문자열 성공으로 둔갑하던 버그 → `error`/`choices`/`finish_reason`/`message`/`content` 각각 명시적으로 검증, 누락 시 raise
- `content_filter` 등 거절/필터링 종료 사유를 정상 완료와 구분 안 하던 문제 → 별도 실패로 분류, 무의미한 재시도 없이 즉시 raise
- `urllib.error.HTTPError`가 `URLError`로 뭉뚱그려져 401/402 같은 영구 오류도 재시도하던 문제 → HTTPError 먼저 구분, 401/402/403/400/404/422는 즉시 raise(재시도 없음), 429/5xx는 `Retry-After` 존중 + exponential backoff로 제한 재시도
- `think`/`num_ctx` 파라미터가 조용히 무시되던 문제 → `think`는 이 백엔드에 대응 기능이 없음을 경고 로그로 명시, `num_ctx`는 로컬 전용 설정임을 문서화(원격 context 한도에 영향 없음)

검증: `analysis/` 밖 스크래치 mock 테스트(11개 케이스, 실제 API 호출 없음)로 정상/length/error-body/choices누락/빈content/content_filter/401/402/429/복구 케이스 전부 의도대로 동작 확인.

미반영(범위 밖, 필요시 별도 작업): confidence 파싱(`parse_all_clusters_confidence`)의 NaN/Infinity/bool 거부, 클러스터 누락 감지, JSON 중복키 검출 — 프로덕션 Ollama 캐시 파이프라인이 의존하는 공용 코드라 리스크 있는 변경으로 별도 검토 필요. provider 고정, 요청별 메타데이터 sidecar 기록, 모델별 버전 분리 캐시 디렉토리 구조도 본격적인 다중모델 비교 실험 시 별도 진행.
