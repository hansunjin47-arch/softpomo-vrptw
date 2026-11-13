"""One same-seed control replay, no coefficient search or Test evaluation."""
import inspect
import platform
import shutil
import torch
from analysis import train_rc_family_entropy_pilot_20261111 as original

OUT=original.ROOT/'result/rc_family_control_repeat_20261111'
BASE=original.ROOT/'result/rc_family_entropy_pilot_v3_20261111'


def main():
    baseline=original.read(BASE/'protocol.json')
    assert all(original.digest(original.ROOT/p)==h for p,h in baseline['original_source_sha256'].items())
    expected=original.read(BASE/'control/initial.json')['sha256']
    src=inspect.getsource(original.main)
    replacements={
        "arms=['control','treatment'],eta_initial=ETA,":"arms=['control'],comparison_control=str(BASE/'control'),eta_initial=0.,environment=environment,",
        "rationale='Heuristic max auxiliary entropy loss .01; not optimized or validated.',":"rationale='Single identical-control replay to measure observed reproducibility, not a variance estimate or tuning run.',",
        "for arm,eta in (('control',0.),('treatment',ETA)):":"for arm,eta in (('control',0.),):",
        "for arm in ('control','treatment'):":"for arm in ('control',):",
        "if initial_hash is None: initial_hash=h;initial_rng=start_rng":"assert h==expected_initial_sha, 'Initial weights differ from reused control'\n        if initial_hash is None: initial_hash=h;initial_rng=start_rng",
        "folder/'checkpoint-last.pt')":"folder/'checkpoint-last.pt')\n            if epoch%20==0: shutil.copy2(folder/'checkpoint-last.pt',folder/f'checkpoint-{epoch}.pt')",
        "epochs_per_arm=100,arms=2,input_hashes_unchanged=True":"epochs_per_arm=100,arms=1,comparison_control=str(BASE/'control'),input_hashes_unchanged=True",
    }
    for old,new in replacements.items():
        assert src.count(old)==1,(old,src.count(old))
        src=src.replace(old,new)
    env=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark)
    namespace=dict(original.__dict__,OUT=OUT,BASE=BASE,shutil=shutil,expected_initial_sha=expected,environment=env)
    exec(src,namespace);namespace['main']()


if __name__=='__main__':
    try:main()
    except Exception as error:
        original.write(OUT/'failure.json',dict(error=repr(error)))
        raise
