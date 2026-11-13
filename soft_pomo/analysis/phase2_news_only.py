"""Remove duplicated event encodings without changing reference-route feedback."""
import json
import re


def revise(prompt, observation):
    news = observation['news_text'].strip()
    if not news:
        raise ValueError('Missing event news')
    lines = []
    in_table = False
    for line in prompt.splitlines():
        if re.match(r'\s*veh\s+ID\s+', line):
            assert line.rstrip().endswith('events')
            line = line.rsplit('events', 1)[0].rstrip()
            in_table = True
        elif in_table and line.startswith('## '):
            in_table = False
        elif in_table and re.match(r'^\s*\d+\s+\d+\s+', line):
            # Columns through travel_time are unchanged; only the event cell is removed.
            cells = line.split(None, 10)
            assert len(cells) == 11, line
            line = '  '.join(cells[:10])
        lines.append(line)
    prompt = '\n'.join(lines)
    prompt, count = re.subn(r'## Disruption Events\n.*?(?=\n## )',
                           '## Disruption Events\n' + news + '\n', prompt, flags=re.S)
    assert count == 1
    marker = 'Exact public event observation (times in minutes after dispatch; overrides rounded event text above): '
    start = prompt.index(marker)
    _, length = json.JSONDecoder().raw_decode(prompt[start + len(marker):])
    prompt = prompt[:start] + prompt[start + len(marker) + length:]
    prompt = prompt.replace('is applied only from revealed_at_min, never before.',
                            'is applied only from the accident disclosure time, never before.')
    assert 'Exact public event observation' not in prompt
    assert 'affected_nodes' not in prompt
    assert not re.search(r'(?:Rain|Accident)_\d+\(', prompt)
    return prompt
