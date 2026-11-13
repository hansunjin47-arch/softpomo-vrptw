"""Pre-fixed one-vehicle reduction control on the good original Pure seed45."""
import inspect
from analysis import train_rc_pure_k17_curriculum_20261112 as original

pilot = original.pilot
OUT = pilot.ROOT/'result/rc_pure45_k15_curriculum_20261112'
SOURCE = pilot.ROOT/'result/rc_pair_returnmask_v100_l5_s45_20261102'
CHECKPOINT = SOURCE/'late5/checkpoint-1000.pt'


def check_original_probes(folder):
    values = {r['name']:r for r in pilot.read(folder/'free_epoch000.json')}
    for expected in pilot.read(SOURCE/'late5/checks/update_020000.json')['results']:
        for key in ('reward','vehicles','travel_time','total_late','unserved'):
            assert abs(values[expected['name']][key]-expected[key]) < 1e-5


def main():
    source = inspect.getsource(original.main)
    marker = '    namespace=dict(pilot.__dict__'
    assert source.count(marker) == 1
    extra = '''    source=source.replace('17','15')
    source=source.replace('rc_pair_returnmask_v100_l5_s44_20261102','rc_pair_returnmask_v100_l5_s45_20261102')
'''
    source = source.replace(marker, extra + marker)
    source = source.replace('check_original_probes=check_original_probes)', 'check_original_probes=check_original_probes,tests=tests)')
    unit_source = inspect.getsource(pilot.tests).replace('17','15').replace('[16]','[14]')
    namespace = dict(pilot.__dict__)
    exec(unit_source, namespace)
    run_namespace = dict(original.__dict__, OUT=OUT, SOURCE=SOURCE, CHECKPOINT=CHECKPOINT,
                         tests=namespace['tests'], check_original_probes=check_original_probes)
    exec(source, run_namespace)
    run_namespace['main']()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        pilot.write(OUT/'failure.json', dict(error=repr(error)))
        raise
