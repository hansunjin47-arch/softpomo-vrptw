"""Export wording-only v2 prompts without changing the active v1 experiment."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'analysis/prompts/ontology_routes_only_customer_only_ac_v1'
NEW = ROOT / 'analysis/prompts/ontology_routes_only_customer_only_ac_v2'


def render(text):
    start = text.index('For an active-group customer i, effective bias')
    end = text.index('Customers may be served outside their reference group', start)
    text = text[:start] + '''For an active-group customer i, add 5*c_i to its RL logit.
Leave all other action logits, including depot, unchanged.
Customer scores are used directly, with no negative transformation.
Masks enforce visited/capacity constraints. Zero is neutral; equal positive scores
favor active-group customers over other actions, rather than disabling guidance.
''' + text[end:]
    text = text.replace(
        'Use one-based cluster IDs and every required CUSTOMER exactly once, 100 entries total.\n'
        'Never output customer 0 or depot scores. No explanations, route edits, assignments or extra pairs.',
        'Use one-based cluster IDs. Output each customer ID 1-100 exactly once in its listed group,\n'
        '100 entries total. No explanations, route edits, assignments or extra pairs.')
    return text


if __name__ == '__main__':
    NEW.mkdir(parents=True, exist_ok=True)
    for variant in ('A', 'C'):
        target = NEW / f'phase2_config{variant}.txt'
        text = render((OLD / target.name).read_text(encoding='utf-8'))
        if target.exists():
            assert target.read_text(encoding='utf-8') == text
        else:
            target.write_text(text, encoding='utf-8')
        print(target)
