"""Verify train requests preserve the original GPT contract exactly, without API calls."""
import run_phase2_online_v2 as run


def main():
    run.verify()
    count=0
    for weight in (1,5,10,20):
        for row in run.io.read(run.OUT/'source/split.json')['train']:
            name=row['name']
            original=run.OUT/f'legacy_train/late{weight}'/name
            context=run.io.read(original/'context.json')
            groups=[g['reference_order'] for g in context['fixed_instance']['groups']]
            folder=run.OUT/f'prompt_restore_tests/late{weight}'/name
            run.make_request(folder,weight,{'name':name},groups,
                             [run.public_observation(name)],training=True)
            assert run.io.read(folder/'context.json')==context
            assert run.io.read(folder/'request.json')==run.io.read(original/'request.json')
            count+=1
    run.io.write(run.OUT/'train_prompt_verification.json',dict(
        status='passed',requests_compared=count,exact_request_equality=True,paid_calls=0))
    print(f'PASS: all {count} train requests exactly match original prompts, ontology and schema')


if __name__=='__main__':
    main()
