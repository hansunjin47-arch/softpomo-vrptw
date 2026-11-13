"""Fixed-target14 Train-only mechanism diagnostic, no tuning."""
from pathlib import Path
import torch
from analysis import train_rc_k17_curriculum_20261112 as pilot
from analysis.train_rc_curriculum_confirmatory_20261112 import means, compare
from analysis.train_rc_v200_transfer_20261112 import OUT as SOURCE, SOURCE as BASE

OUT = pilot.ROOT/'result/rc_v200_k14_mechanism_20261112'


def main():
    assert pilot.read(SOURCE/'status.json')['stage']=='complete'
    assert not OUT.exists()
    torch.set_num_threads(1)
    cfg = pilot.read(SOURCE/'protocol.json')['config']
    assert pilot.read(SOURCE/'protocol.json')['target']==14
    rows = pilot.read(BASE/'split.json')['train']
    original = means(pilot.read(SOURCE/'initial_train20.json'))
    hashes = {str(p):pilot.digest(p) for p in [Path(__file__),SOURCE/'protocol.json',BASE/'split.json',
        *[SOURCE/a/f'checkpoint-{e}.pt' for a in ('control','curriculum') for e in (20,40)]]}
    OUT.mkdir()
    pilot.write(OUT/'protocol.json',dict(target=14,checkpoint_hashes=hashes,
        scope='Train20 only; 4x20 inference, no API/Test/training or target sweep',
        hypotheses=['Failed to learn14 routing','Improved14 routing still worse than original15.1','Free adaptation degrades14 routing']))
    report = dict(original=original,results={})
    for arm in ('control','curriculum'):
        for epoch in (20,40):
            assert all(pilot.digest(Path(p))==h for p,h in hashes.items())
            model = pilot.Policy(**cfg['model']).cuda().eval()
            model.load_state_dict(torch.load(SOURCE/arm/f'checkpoint-{epoch}.pt',map_location='cpu',weights_only=False)['model_state_dict'])
            model.fallback_count=0
            model.constrained_states=0
            forced = pilot.evaluate(model,rows,cfg,14)
            pilot.write(OUT/f'{arm}_forced14_epoch{epoch}.json',forced)
            free = pilot.read(SOURCE/arm/f'free_epoch{epoch:03d}.json')
            result = compare(dict(control=free,curriculum=forced))
            result['fallback_count']=model.fallback_count
            result['constrained_states']=model.constrained_states
            avg = means(forced)
            delta = {k:avg[k]-original[k] for k in original}
            result['forced_minus_original']=delta
            result['forced_minus_original_cost']=dict(vehicle=200*delta['vehicles'],
                travel=delta['travel_time'],late=5*delta['total_late'],
                note='Unserved penalty excluded here; see unserved/reward separately')
            report['results'][f'{arm}_{epoch}']=result
            print(arm,epoch,avg,'fallback',model.fallback_count,flush=True)
            del model
            torch.cuda.empty_cache()
    pilot.write(OUT/'report.json',report)
    pilot.write(OUT/'status.json',dict(stage='complete',api_calls=0,test_accessed=False,evaluations=80))


if __name__=='__main__':
    main()
