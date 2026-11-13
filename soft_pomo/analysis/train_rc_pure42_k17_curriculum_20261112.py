"""Independent seed42 replication with fixed K17 and unchanged curriculum."""
import inspect
from analysis import train_rc_pure_k17_curriculum_20261112 as parent

OUT=parent.pilot.ROOT/'result/rc_pure42_k17_curriculum_20261112'
SOURCE=parent.pilot.ROOT/'result/rc_pair_returnmask_v100_l5_s42_20261031'
CHECKPOINT=SOURCE/'late5/checkpoint-1000.pt'


def check_original_probes(folder):
    values={r['name']:r for r in parent.pilot.read(folder/'free_epoch000.json')}
    for expected in parent.pilot.read(SOURCE/'late5/checks/update_020000.json')['results']:
        for key in ('reward','vehicles','travel_time','total_late','unserved'):
            assert abs(values[expected['name']][key]-expected[key])<1e-5,(key,values[expected['name']][key],expected[key])


def main():
    src=inspect.getsource(parent.main)
    old='source=inspect.getsource(pilot.main)'
    assert src.count(old)==1
    src=src.replace(old,old+"\n    source=source.replace('rc_pair_returnmask_v100_l5_s44_20261102','rc_pair_returnmask_v100_l5_s42_20261031')")
    namespace=dict(parent.__dict__,OUT=OUT,SOURCE=SOURCE,CHECKPOINT=CHECKPOINT,check_original_probes=check_original_probes)
    exec(src,namespace);namespace['main']()


if __name__=='__main__':
    try:main()
    except Exception as error:
        parent.pilot.write(OUT/'failure.json',dict(error=repr(error)))
        raise
