"""Separate original1000 reference for the fixed V200 transfer experiment."""
import torch
from analysis import train_rc_k17_curriculum_20261112 as pilot
from analysis.train_rc_curriculum_confirmatory_20261112 import means, validation, compare
from analysis.train_rc_v200_transfer_20261112 import OUT, SOURCE


def main():
    assert pilot.read(OUT/'status.json')['stage'] == 'complete'
    assert not (OUT/'original_validation20.json').exists()
    torch.set_num_threads(1)
    protocol = pilot.read(OUT/'protocol.json')
    assert all(pilot.digest(pilot.ROOT/p) == h for p,h in protocol['source_hashes'].items())
    cfg = protocol['config']
    model = pilot.Policy(**cfg['model']).cuda().eval()
    checkpoint = SOURCE/'late5/checkpoint-1000.pt'
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
    original = validation(model,cfg)
    pilot.write(OUT/'original_validation20.json',original)
    train = dict(original=pilot.read(OUT/'initial_train20.json'),
        **{a:pilot.read(OUT/a/'free_epoch040.json') for a in ('control','curriculum')})
    val = dict(original=original,**{a:pilot.read(OUT/a/'validation20.json') for a in ('control','curriculum')})
    report = {}
    for label,rows in (('train',train),('validation',val)):
        report[label] = dict(means={a:means(r) for a,r in rows.items()},
            curriculum_vs_original=compare(dict(control=rows['original'],curriculum=rows['curriculum'])),
            control_vs_original=compare(dict(control=rows['original'],curriculum=rows['control'])))
    report['caveat']='Matched control success and improvement over original1000 are separate. No Test or selection.'
    report['original_checkpoint_sha256']=pilot.digest(checkpoint)
    pilot.write(OUT/'three_way_comparison.json',report)
    print(report['train']['means'],report['validation']['means'],flush=True)


if __name__ == '__main__':
    main()
