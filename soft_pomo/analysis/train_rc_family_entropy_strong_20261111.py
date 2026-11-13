"""One preregistered stronger-signal arm; reuse matched fresh eta-zero control."""
import inspect
import shutil
from analysis import train_rc_family_entropy_pilot_20261111 as weak

OUT=weak.ROOT/'result/rc_family_entropy_strong_20261111'
BASE=weak.ROOT/'result/rc_family_entropy_pilot_v3_20261111'
ETA=.14426950408889635


def main():
    baseline=weak.read(BASE/'protocol.json')
    assert all(weak.digest(weak.ROOT/p)==h for p,h in baseline['original_source_sha256'].items())
    assert weak.read(BASE/'status.json')['stage']=='complete'
    expected=weak.read(BASE/'control/initial.json')['sha256']
    src=inspect.getsource(weak.main)
    replacements={
        "arms=['control','treatment'],eta_initial=ETA,":"arms=['treatment'],reused_control=str(BASE/'control'),eta_initial=ETA,",
        "rationale='Heuristic max auxiliary entropy loss .01; not optimized or validated.',":"rationale='One stronger-signal pilot, initial two-case ratio about1.06%/.413%, maximum entropy auxiliary loss .1. No additional coefficient sweep.',success='epoch100 Train20 reward exceeds reused control with zero unserved; higher entropy alone is not success.',",
        "for arm,eta in (('control',0.),('treatment',ETA)):":"for arm,eta in (('treatment',ETA),):",
        "for arm in ('control','treatment'):":"for arm in ('treatment',):",
        "if initial_hash is None: initial_hash=h;initial_rng=start_rng":"assert h==expected_initial_sha, 'Initial weights differ from reused control'\n        if initial_hash is None: initial_hash=h;initial_rng=start_rng",
        "folder/'checkpoint-last.pt')":"folder/'checkpoint-last.pt')\n            if epoch%20==0: shutil.copy2(folder/'checkpoint-last.pt',folder/f'checkpoint-{epoch}.pt')",
        "epochs_per_arm=100,arms=2,input_hashes_unchanged=True":"epochs_per_arm=100,arms=1,reused_control=str(BASE/'control'),input_hashes_unchanged=True",
    }
    for old,new in replacements.items():
        assert src.count(old)==1,(old,src.count(old))
        src=src.replace(old,new)
    namespace=dict(weak.__dict__,OUT=OUT,ETA=ETA,BASE=BASE,shutil=shutil,expected_initial_sha=expected)
    exec(src,namespace)
    namespace['main']()


if __name__=='__main__':
    try:main()
    except Exception as error:
        weak.write(OUT/'failure.json',dict(error=repr(error)))
        raise
