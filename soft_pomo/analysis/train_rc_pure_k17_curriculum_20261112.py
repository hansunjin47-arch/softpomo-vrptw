"""Replicate K17 curriculum from original PureRL epoch1000, not entropy model."""
import inspect
from analysis import train_rc_k17_curriculum_20261112 as pilot

OUT=pilot.ROOT/'result/rc_pure_k17_curriculum_20261112'
SOURCE=pilot.ROOT/'result/rc_pair_returnmask_v100_l5_s44_20261102'
CHECKPOINT=SOURCE/'late5/checkpoint-1000.pt'


def check_original_probes(folder):
    values={r['name']:r for r in pilot.read(folder/'free_epoch000.json')}
    for expected in pilot.read(SOURCE/'late5/checks/update_020000.json')['results']:
        for key in ('reward','vehicles','travel_time','total_late','unserved'):
            assert abs(values[expected['name']][key]-expected[key])<1e-5,(key,values[expected['name']][key],expected[key])


def main():
    source=inspect.getsource(pilot.main)
    replacements={
        "cfg=read(SOURCE/'protocol.json')['config']":"cfg=read(SOURCE/'protocol.json')['configs']['5']",
        "assert saved['epoch']==100":"assert saved['epoch']==1000\n    saved['rng']=(saved['python_rng'],saved['cpu_rng'],saved['cuda_rng'])",
        "hashes=read(SOURCE/'protocol.json')['original_source_sha256']":"hashes=read(SOURCE/'protocol.json')['source_hashes']",
        "orderrng=random.Random(44)":"orderrng=random.Random();orderrng.setstate(saved['python_rng'])",
        "allorder=[orderrng.choice(cfg['trainer']['train_instances']) for _ in range(2800)];order=allorder[2000:]":"order=[orderrng.choice(cfg['trainer']['train_instances']) for _ in range(800)]",
        "assert allorder[:2000]==read(SOURCE/'protocol.json')['data_order']":"assert len(order)==800",
        "100+epoch,500":"1000+epoch,500",
        "skips=0\n        for epoch":"check_original_probes(folder)\n        skips=0\n        for epoch",
        "limits='K17 selected from prior Train result, no Test information. Fixed starting checkpoint, not fresh training.'":"limits='Original PureRL epoch1000 model/Adam/scaler/Python CPU CUDA RNG restored identically; dataorder continues saved Python RNG. Epsilon1001-1040. K17 from prior Train result, no Test.'",
    }
    for old,new in replacements.items():
        assert source.count(old)==1,(old,source.count(old))
        source=source.replace(old,new)
    namespace=dict(pilot.__dict__,OUT=OUT,SOURCE=SOURCE,CHECKPOINT=CHECKPOINT,check_original_probes=check_original_probes)
    exec(source,namespace);namespace['main']()


if __name__=='__main__':
    try:main()
    except Exception as error:
        pilot.write(OUT/'failure.json',dict(error=repr(error)))
        raise
