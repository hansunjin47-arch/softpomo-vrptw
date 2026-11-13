"""Inventory archived pool45 solutions without evaluating or changing models."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pattern = re.compile(
    r'vehicles=(\d+)\s+served=\s*(\d+)/(\d+)\s+dist=\s*([\d.]+)'
    r'\s+reward=([-\d.]+)\s+late_stops=(\d+)\s+total_late=([\d.]+)'
)
rows = []
for directory in sorted((ROOT / 'result').glob('*pool45')):
    for path in sorted(directory.glob('**/*101*solution.txt')):
        match = pattern.search(path.read_text(encoding='utf-8'))
        if not match:
            raise ValueError(str(path))
        k, served, total, distance, reward, late, amount = match.groups()
        rows.append(dict(source=str(path.relative_to(ROOT)), run=directory.name,
                         view=path.parent.name, instance=path.stem.removesuffix('_solution'),
                         vehicles=int(k), served=int(served), total=int(total),
                         distance=float(distance), reward=float(reward),
                         late=int(late), amount=float(amount)))
(ROOT / 'analysis/pool45_history.json').write_text(
    json.dumps(rows, indent=2), encoding='utf-8')
lines = [
    '# Pool45 과거 결과', '',
    '2026-09-20 기존 로그와 solution 조사. 신규 평가/학습/LLM 호출 없음.', '',
    '## 범위와 주의사항', '',
    '- CH/RH/RCH 각 15개 BASE + 15개 RAIN + 15개 ACC를 사용한 고정 좌표 TW 변형 데이터.',
    '- 좌표, depot, 수요, 서비스 시간을 유지한다. 새로운 좌표로의 일반화 실험과 구분한다.',
    '- 아래 전체 표는 현재 보존된 solution 기록이다. 디렉터리 이름만으로 학습/평가 설정을 확정하지 않는다.',
    '- plots_purerl500은 별도 500 평가 폴더다. plots와 섞어서 최고 값을 고르지 않는다.',
    '- 과거 RAIN/ACC는 현재 RAIN2/ACC2와 동일한 시나리오라고 가정하지 않는다.',
    '- 지각은 당시 기록값이며 현재 strict arrival 기준으로 재계산하지 않았다.',
    '- BKS 등 다른 guidance 실행도 보존 현황에 포함하되 Pure RL로 분류하지 않는다.', '',
    '## 초기 로그에 남은 1000 epoch 결과', '',
    'train_c1_pool45.log / train_r1_pool45.log: batch16, POMO100, no aug, MC1, seed42.',
    '시나리오 출력 순서는 BASE -> ACC -> RAIN이다.', '',
    '| 유형 | 시나리오 | 차량 | 거리 | 지각 수 | 지각량 |',
    '|---|---|---:|---:|---:|---:|',
    '| C | BASE | 11 | 859.66 | 0 | 0.00 |',
    '| C | ACC | 11 | 859.66 | 0 | 0.00 |',
    '| C | RAIN | 11 | 941.18 | 0 | 0.00 |',
    '| R | BASE | 22 | 1977.80 | 5 | 19.32 |',
    '| R | ACC | 23 | 1953.72 | 1 | 1.66 |',
    '| R | RAIN | 25 | 2104.74 | 1 | 3.12 |', '',
    'C/R 로그 최종 train_reward는 -0.8505/-0.9001. 초기 RC 로그는 데이터 로딩까지만 남았다.',
    '중요: 초기 로그가 가리키는 현재 plots의 수치는 이 로그와 다르다. 동일 경로 재사용 등으로 산출물이 달라진 것으로 보이며, 초기 성과를 현재 체크포인트 성과로 간주하면 안 된다.', '',
    '## 체크포인트 구조', '',
    '- C/R slack_purerl_pool45 checkpoint-1000.pt: 현재 VRPTWModel strict=True 로딩 실제 성공.',
    '- 기본 pool45 / purerl_pool45 및 C/R purerl_hybridlex_pool45: slack_head weight/bias만 없음. 나머지 key/shape 일치.',
    '- 위 모델들은 예전 50k 체크포인트와 달리 사고 embedding shape 불일치가 없다.',
    '- strict 로딩 성공은 전처리, 이벤트 적용, reward, 평가 설정까지 같다는 뜻이 아니다.',
    '- slack 없는 모델을 평가하려면 명시적으로 zero-head 호환 처리를 하고 별도 평가로 기록해야 한다.', '',
    '## 현재 저장 솔루션 전체', '',
    '| 출처 | 평가 대상 | 차량 | 방문 | 거리 | 지각 수 | 지각량 |',
    '|---|---|---:|---:|---:|---:|---:|',
]
for r in rows:
    lines.append(f"| `{r['source']}` | {r['instance']} | {r['vehicles']} | "
                 f"{r['served']}/{r['total']} | {r['distance']:.2f} | {r['late']} | {r['amount']:.2f} |")
(ROOT / 'analysis/pool45_history_review.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(f'Saved {len(rows)} archived solutions to analysis/pool45_history_review.md')
