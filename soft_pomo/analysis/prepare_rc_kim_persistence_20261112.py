"""CPU-only fixed100 persistence protocol; missing in-flight checkpoints marked pending."""
import hashlib
import json
import random
import torch
from analysis import train_rc_k17_curriculum_20261112 as pilot
from analysis.prepare_rc_kim_curriculum_20261112 import RECEIVED

OUT=pilot.ROOT/'result/rc_kim_persistence_cpu_design_20261112'
SOURCES={42:pilot.ROOT/'result/rc_kim_curriculum_deterministic_20261112',43:pilot.ROOT/'result/rc_kim43_curriculum_deterministic_20261112'}


def main():
    result={}
    for seed,folder in SOURCES.items():
        p=pilot.read(folder/'preregistration.json');cfg=p['config']
        original=RECEIVED/f'runs/vehicle100/seed{seed}/late5/checkpoint-1000.pt'
        state=torch.load(original,map_location='cpu',weights_only=False)
        rng=random.Random();rng.setstate(state['python_rng'])
        order=[rng.choice(cfg['trainer']['train_instances']) for _ in range(2000)]
        assert order[:800]==p['data_order']
        arms={}
        for arm in ('control','curriculum'):
            path=folder/arm/'checkpoint-40.pt'
            if not path.exists():arms[arm]=dict(status='pending_current_training',path=str(path));continue
            saved=torch.load(path,map_location='cpu',weights_only=False)
            assert saved['epoch']==40 and len(saved['rng'])==3
            assert all(k in saved for k in ['model_state_dict','optimizer_state_dict','scaler'])
            arms[arm]=dict(status='state_available_cpu_verified',path=str(path),sha256=pilot.digest(path),
                restore='Own model/Adam/scaler and Python/CPU/CUDA RNG,not original1000')
        result[str(seed)]=dict(config=cfg,arms=arms,continuation_order=order[800:],
            order_sha256=hashlib.sha256(json.dumps(order[800:]).encode()).hexdigest(),initial800_prefix_exact=True)
    base=SOURCES[42]
    runtime=(base/'status.json').stat().st_mtime-(base/'preregistration.json').stat().st_mtime
    OUT.mkdir(exist_ok=True)
    pilot.write(OUT/'protocol_draft.json',dict(seeds=result,
        schedule='Each of4arms free60additionalepochs,finalepoch100 fixed; no family constraint in extension',
        evaluation='Train20 greedy at60/80/100; final100 only for outcome; no early stopping by superiority',
        hypotheses=['Initial curriculum gains persist','Gains regress','Control catches up'],
        outcome='Report eachseed original1000/control100/curr100 and40->100change; require matched positive reward delta/U0 for persistence,not selective checkpoint',
        runtime80_seconds=runtime,projected240_seconds=3*runtime,budget='240epochs4800updates approximately3.5-4hours',
        order_note='Precomputed order does not consume globalPythonRNG; regenerate original canonical2000,verify800prefix,then use remaining1200 sharedwithinseed.',
        numerical='DeterministicTrue,CUBLAS4096:8 subprocess-only,unchangedAMP/LR/epsilon.01',
        restrictions='NoAPI/Test; originaloutputsimmutable; pending43checkpointmustcompletebeforefreeze/execution',
        status='CPUdesignonly; no training launch'))
    print({seed:{arm:info['status'] for arm,info in row['arms'].items()} for seed,row in result.items()})


if __name__=='__main__':main()
