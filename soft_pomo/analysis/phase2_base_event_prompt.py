"""Explicit BASE/event request contract for the new sequential Phase 2 pipeline."""
import copy
import json
import argparse
from pathlib import Path
from base_event_prompt import BASE_INSTRUCTION, EVENT_INSTRUCTION


def make_request(template, context, *, training, prior_base_confidence=None,
                 base_context=None):
    """Require an independently prepared no-event reference, never redact event feedback."""
    context = copy.deepcopy(context)
    is_base = prior_base_confidence is None
    if is_base and context.get('public_event'):
        raise ValueError('BASE requires a no-event context, not event-conditioned RL feedback')
    if not is_base:
        if not context.get('public_event'):
            raise ValueError('Event revision requires disclosed event information')
        if base_context is None or base_context.get('public_event'):
            raise ValueError('Missing event-free BASE provenance')
        if context['fixed_instance'] != base_context['fixed_instance']:
            raise ValueError('BASE/event instance and groups must match; regenerate BASE for these groups')
    for supplied in [context] + ([base_context] if base_context is not None else []):
        if training:
            feedback = supplied.get('train_execution_feedback', {})
            if feedback.get('scope') != 'offline_train_only':
                raise ValueError('Training requires measured RL execution feedback')
        else:
            # Test is instance-only: enforce the established Kim context allowlist.
            from generate_kim_l5_all_gpt import validate
            groups = [g['members'] for g in supplied['fixed_instance']['groups']]
            validate(supplied, groups)
    instruction = BASE_INSTRUCTION if is_base else EVENT_INSTRUCTION
    if training:
        instruction += ('\nRL routes and measured performance are offline TRAIN evidence. '
                        'Distinguish no-event BASE performance from event-scenario performance. '
                        'Do not label event-conditioned performance as BASE performance. '
                        'Use measured metrics to assess the reference, not to claim improvement '
                        'of an unevaluated alternative.')
    else:
        instruction += '\nUse instance and public event information only. No test RL reference is supplied.'
    if not is_base:
        context['prior_base_confidence'] = copy.deepcopy(prior_base_confidence)
    request = copy.deepcopy(template)
    text = template['messages'][0]['content'].rsplit('\n', 1)[0]
    request['messages'] = [dict(role='user',content=text+'\n'+instruction+'\n'+
                              json.dumps(context,separators=(',', ':'),allow_nan=False))]
    return request


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--split', choices=['train','test'], required=True)
    parser.add_argument('--base-context', type=Path)
    parser.add_argument('--base-confidence', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    def read(path):
        return json.loads(path.read_text(encoding='utf-8')) if path else None
    request = make_request(read(args.template), read(args.context), training=args.split=='train',
        prior_base_confidence=read(args.base_confidence), base_context=read(args.base_context))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as f:
        json.dump(request,f,indent=2)
