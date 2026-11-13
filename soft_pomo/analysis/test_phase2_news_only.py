import json
from pathlib import Path
from phase2_news_only import revise


def test_saved_prompts():
    root = Path(__file__).resolve().parents[1] / 'result/rc101_pure500_train80_phase2_20260930'
    paths = list((root/'requests').glob('*/request.json')) + list((root/'eval_1000_test80/requests').glob('*/request.json'))
    assert len(paths) == 160
    counts = {}
    marker = 'Exact public event observation (times in minutes after dispatch; overrides rounded event text above): '
    for path in paths:
        prompt = json.loads(path.read_text())['messages'][0]['content']
        observation, _ = json.JSONDecoder().raw_decode(prompt.split(marker)[1])
        updated = revise(prompt, observation)
        assert updated.count(observation['news_text']) == 1
        assert 'Required membership:' in updated
        assert '[RL visit order' in updated
        assert 'affected_nodes' not in updated
        counts[observation['event_type']] = counts.get(observation['event_type'], 0) + 1
    print('Validated', len(paths), 'saved prompts:', counts)


if __name__ == '__main__':
    test_saved_prompts()
