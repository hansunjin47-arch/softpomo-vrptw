"""Single matched K18 mechanism control; reuse saved free40 and K17 arms."""
import inspect
from analysis import train_rc_pure_k17_curriculum_20261112 as original

pilot = original.pilot
OUT = pilot.ROOT / 'result/rc_pure_k18_curriculum_20261112'


def matched_protocol(cfg, order, hashes):
    previous = pilot.read(original.OUT / 'protocol.json')
    assert cfg == previous['config']
    assert order == previous['data_order']
    assert hashes == previous['input_hashes']


def main():
    source = inspect.getsource(original.main)
    marker = '    namespace=dict(pilot.__dict__'
    assert source.count(marker) == 1
    extra = '''    source=source.replace('17','18')
    source=source.replace("arms=['control','curriculum']", "arms=['curriculum']")
    source=source.replace("for arm in ('control','curriculum'):", "for arm in ('curriculum',):")
    source=source.replace("stage='complete',arms=2", "stage='complete',arms=1")
    source=source.replace("    write(OUT/'protocol.json'", "    matched_protocol(cfg,order,hashes)\\n    write(OUT/'protocol.json'", 1)
'''
    source = source.replace(marker, extra + marker)
    source = source.replace('check_original_probes=check_original_probes)',
                            'check_original_probes=check_original_probes,tests=tests,matched_protocol=matched_protocol)')
    unit_source = inspect.getsource(pilot.tests).replace('17', '18')
    namespace = dict(pilot.__dict__)
    exec(unit_source, namespace)
    run_namespace = dict(original.__dict__, OUT=OUT, tests=namespace['tests'], matched_protocol=matched_protocol)
    exec(source, run_namespace)
    run_namespace['main']()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        pilot.write(OUT / 'failure.json', dict(error=repr(error)))
        raise
