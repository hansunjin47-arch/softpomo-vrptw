"""Replace legacy priority instructions while preserving instance evidence."""


def revise(prompt):
    replacements = {
        'Minimise in order of priority:':
            'Serve every customer. For fully served solutions minimize the weighted cost '
            '0.04*K + TT/5760 + 0.0075*Lt for this RC101 setting. '
            'K is vehicles, TT is actual driving time including event delays, and Lt is '
            'summed customer and depot-return lateness in minutes. This is not a '
            'lexicographic objective. Waiting and service affect arrivals but are not TT. '
            'Do not trade a small route saving for a large increase in lateness.',
        "The '[RL visit order]' line above":
            'The RL visit order is a reference solution. Preserve useful parts and suggest '
            'priority changes supported by the observations to reduce the weighted cost.',
        "The 'slack' column (real remaining buffer":
            'Slack describes arrivals under the reference route only. Negative slack '
            'identifies a past late arrival, not a mandatory highest-priority customer. '
            'Reordering or reassignment changes arrival times and downstream slack. '
            'Consider geography, service, deadlines and events together; a positive '
            'reference slack does not guarantee safe postponement.',
        'When raising a negative-slack':
            'Use moderate, evidence-based priority changes. Consider delay propagated '
            'to other customers and depot return, not only the originally late customer. '
            'Do not claim insertion feasibility without a replay of the modified route.',
        'Also score "0" (depot':
            'Score depot 0 on the same scale as customers. All scores are additive soft '
            'preferences, not hard masks, cutoffs, fixed visit orders or reassignment '
            'commands. A customer below depot is not automatically rejected. Depot '
            'preference persists until return even after all group customers are visited. '
            'The active group is anchored to the first customer of the vehicle. Other '
            'groups receive no direct group bias; an absorbed customer does not carry '
            'its original group score to the receiving vehicle. Do not assume another '
            'vehicle can accept a deprioritized customer. Balance depot return against '
            'remaining service, vehicle cost and lateness; zero depot score is allowed '
            'but is not mandatory for a punctual reference route.',
    }
    lines = prompt.splitlines()
    for prefix, replacement in replacements.items():
        hits = [i for i, line in enumerate(lines) if line.startswith(prefix)]
        assert len(hits) == 1, (prefix, len(hits))
        lines[hits[0]] = replacement
    result = '\n'.join(lines)
    assert 'ANY stop with slack < 0 must' not in result
    assert "it's a cutoff" not in result
    return result
