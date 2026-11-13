"""One-update deterministic diagnostic; independent subprocess environment only."""
import argparse
import copy
import hashlib
import torch
from analysis import train_rc_k17_curriculum_20261112 as pilot
from analysis.train_rc_kim_curriculum_20261112 import OUT as SOURCE,SOURCE as BASE,ART,BUNDLE
from analysis.prepare_rc_kim_curriculum_20261112 import CurriculumGuidedPolicy
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_bias_guided.support import load
from experiments.concurrent_fleet_v3 import run as data_module

OUT=pilot.ROOT/'result/rc_kim_deterministic_repeat_20261112'


def digest_state(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            h.update(str((str(x.dtype),tuple(x.shape))).encode())
            h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=repr):h.update(repr(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def main(arm):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg=pilot.read(SOURCE/'protocol.json')['config'];protocol=pilot.read(SOURCE/'preregistration.json')
    assert all(pilot.digest(pilot.ROOT/p)==h for p,h in protocol['hashes'].items())
    dest=OUT/arm;assert not dest.exists();dest.mkdir(parents=True)
    groups=pilot.read(ART/'groups.json');data_module.SOURCE=BUNDLE/'dataset'
    rows={r['name']:r for r in pilot.read(BASE/'split.json')['train']}
    class CachedEnv(GuidedEnv):
        def __init__(self,inst,count,params,device='cuda'):
            cache=pilot.read(ART/'train_cache'/(inst['name']+'.json'))
            super().__init__(inst,count,params,device,groups=groups,scores=parse_response(cache['initial'],groups),beta=5.)
            if cache['post_disclosure'] is not None:
                t=cache['revealed_at_min'];self.update_event_confidence(parse_response(cache['post_disclosure'],groups),revealed_at_min=t,effective_time_min=t)
    pilot.trainer.TensorFleetEnv=CachedEnv;pilot.trainer.load=load
    saved=torch.load(SOURCE/'curriculum/checkpoint-30.pt',map_location='cpu',weights_only=False)
    model=CurriculumGuidedPolicy(**cfg['model']).cuda().train();model.load_state_dict(saved['model_state_dict'],strict=True)
    opt=torch.optim.Adam(model.parameters(),**cfg['optimizer']['optimizer']);opt.load_state_dict(copy.deepcopy(saved['optimizer_state_dict']))
    scaler=torch.amp.GradScaler('cuda');scaler.load_state_dict(copy.deepcopy(saved['scaler']))
    pilot.restore(saved['rng']);model.target=None;model.new_customer_epsilon=.01
    hashes=[]
    def capture(update):
        state=dict(model=model.state_dict(),adam=opt.state_dict(),scaler=scaler.state_dict(),rng=pilot.rng())
        hashes.append(dict(after_update=update,**{k:digest_state(v) for k,v in state.items()}))
        torch.save(state,dest/f'after-{update}.pt')
    capture(600)
    stats=[]
    for update in range(601,602):
        name=protocol['data_order'][update-1];inst,public=load(rows[name]);model.fallback_count=0
        result,_=pilot.trainer.train_batch(model,opt,scaler,inst,public,cfg['environment'],128,100,True)
        stats.append(dict(update=update,name=name,**result));capture(update)
        print(arm,update,result['reward'],result['gradient_norm'],flush=True)
    pilot.write(dest/'report.json',dict(hashes=hashes,metrics=stats,
        settings=dict(torch=torch.__version__,cuda=torch.version.cuda,deterministic=torch.are_deterministic_algorithms_enabled(),
            cudnn_benchmark=torch.backends.cudnn.benchmark,matmul_tf32=torch.backends.cuda.matmul.allow_tf32),
        scope='One deterministic update,independentprocess; no API/Test/FP32/LR alteration'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['A','B']);main(p.parse_args().arm)
