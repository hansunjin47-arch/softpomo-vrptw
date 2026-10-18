import random
import copy
import json
import unittest
import numpy as np
import torch
from experiments.concurrent_fleet_v5.test_fixture import instance
from experiments.concurrent_fleet_v5.reference_env import FleetEnv
from experiments.concurrent_fleet_v5.env import TensorFleetEnv
from experiments.concurrent_fleet_v5.policy import FleetPolicy
from train_vrptw import VRPTWModel

PARAMS=dict(vehicle_penalty=3/14,time_penalty=1,TT_max=16.9695,late_penalty=5,Lt_max=16.9695,
            unserved_penalty=500,dist_penalty_weight=0,late_count_penalty=0)
MP=dict(embedding_dim=64,encoder_layer_num=2,head_num=4,qkv_dim=16,ff_hidden_dim=128,
        logit_clipping=10,learn_first_customer=True,eval_type='argmax')


class Tests(unittest.TestCase):
    def parity(self,device):
        for variant in range(4):
            inst=instance(); inst['vehicle_limit']=2
            if variant==1:
                inst['node_service'][0]=.3
            if variant==2:
                inst['tt'][0,1]=0
            if variant==3:
                inst['preset_events']=[dict(trigger_time=5.,duration=50.,nodes=[2,3],multiplier=2.)]
                inst['node_tw_close'][:]=.1
            cpu=[FleetEnv(inst,[],environment_params=PARAMS) for _ in range(20)]
            gpu=TensorFleetEnv(inst,20,PARAMS,device)
            rng=random.Random(42)
            for step in range(100):
                mask=gpu.mask().cpu().numpy()
                choices=[]
                for i,e in enumerate(cpu):
                    self.assertEqual(e.done,bool(gpu.done[i]))
                    if e.done:
                        choices.append(0); continue
                    actions=e.actions(); mapping={}
                    for a in actions:
                        slot=a[1]-1 if a[0]==-1 else (2*e.n+1 if a[1]==-1 else e.n+a[1])
                        mapping[slot]=a
                    self.assertEqual(set(np.flatnonzero(mask[i])),set(mapping))
                    choice=rng.choice(list(mapping)); choices.append(choice); e.step(mapping[choice])
                gpu.step(torch.tensor(choices,device=device),validate=True)
                for i,e in enumerate(cpu):
                    np.testing.assert_allclose(float(gpu.time[i]),e.time,rtol=2e-6,atol=1e-4)
                    self.assertEqual(e.customer_state.tolist(),gpu.customer[i].tolist())
                if all(e.done for e in cpu):
                    break
            self.assertTrue(bool(gpu.done.all()))
            metrics=gpu.metrics()
            for i,e in enumerate(cpu):
                for key in ('reward','vehicles','travel_time','total_late','late_count','unserved'):
                    np.testing.assert_allclose(float(metrics[key][i]),e.metrics()[key],rtol=2e-6,atol=1e-4)

    def test_cpu_parity(self): self.parity('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_parity(self): self.parity('cuda')

    def test_legacy_parameter_schema_and_decoder(self):
        torch.manual_seed(42); model=FleetPolicy(**MP)
        legacy=VRPTWModel(**MP)
        legacy.load_state_dict(model.state_dict(),strict=True)
        self.assertEqual(list(model.state_dict()),list(legacy.state_dict()))
        inst=instance(); env=TensorFleetEnv(inst,1,PARAMS,'cpu')
        env.step(torch.tensor([0])); env.step(torch.tensor([7]))
        f=env.frame(); starts,enc=model.encode_events(inst,[])
        p=model.distribution(f,starts,enc,env).probs[:,3:7]
        valid=f['mask'][:,3:7]
        model.decoder.set_kv(enc[0])
        node=f['node']; time=f['time']; tt=(env.base/env.horizon).float()[node]
        arrival=time[:,None]+tt
        slack=torch.stack([((env.close/env.horizon).float()[None]-arrival).clamp(-2,2),
                           ((env.open/env.horizon).float()[None]-arrival).clamp(0,2),tt],-1)
        expected=model.decoder(enc[0][:,node,:],f['load'][None],time[None],
                 torch.zeros_like(valid,dtype=torch.float).masked_fill(~valid,-torch.inf)[None],slack[None])
        torch.testing.assert_close(p,expected[0])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_full_graph_single_backward_and_encoder_batch(self):
        from experiments.concurrent_fleet_v5.train import train_batch
        from unittest.mock import patch
        torch.manual_seed(17)
        inst=instance(); model=FleetPolicy(**MP).cuda()
        opt=torch.optim.Adam(model.parameters(),lr=1e-4)
        before=[p.detach().clone() for p in model.parameters()]
        batches=[]
        hook=model.encoder.register_forward_pre_hook(lambda module,args: batches.append(args[0].shape[0]))
        with patch('torch.autograd.backward',wraps=torch.autograd.backward) as backward:
            summary,_=train_batch(model,opt,torch.amp.GradScaler('cuda',enabled=False),inst,[],PARAMS,2,4,amp=False)
            self.assertEqual(backward.call_count,1)
        hook.remove()
        self.assertEqual(batches,[2])
        self.assertFalse(summary['optimizer_step_skipped'])
        self.assertTrue(any(not torch.equal(p,q) for p,q in zip(model.parameters(),before)))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_hidden_accident_encoding_is_not_used_early(self):
        inst=instance(); model=FleetPolicy(**MP)
        env=TensorFleetEnv(inst,1,PARAMS,'cpu')
        public=[dict(event_type='ACCIDENT',revealed_at_min=15,start_min=15,end_min=50,
                     affected_nodes=[2],vehicles_involved=3)]
        starts,enc=model.encode_events(inst,public)
        a=model.distribution(env.frame(),starts,enc,env).probs
        b=model.distribution(env.frame(),*model.encode_events(inst,[]),env).probs
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_rc101_rain_and_accident_parity(self):
        from experiments.concurrent_fleet_v5.io_utils import SOURCE,ROOT,load
        split=json.loads((SOURCE/'split.json').read_text())
        params=json.loads((SOURCE/'late5.json').read_text())['environment']
        for row in (split['train'][0],split['train'][10]):
            inst,public=load(row)
            cpu=[FleetEnv(inst,public,environment_params=params) for _ in range(8)]
            gpu=TensorFleetEnv(inst,8,params)
            rng=random.Random(53)
            for step in range(200):
                mask=gpu.mask().cpu().numpy(); actions=[]
                for i,e in enumerate(cpu):
                    self.assertEqual(e.done,bool(gpu.done[i]))
                    if e.done: actions.append(0); continue
                    mapped={a[1]-1 if a[0]==-1 else (2*e.n+1 if a[1]==-1 else e.n+a[1]):a for a in e.actions()}
                    self.assertEqual(set(mapped),set(np.flatnonzero(mask[i])))
                    a=rng.choice(list(mapped)); actions.append(a); e.step(mapped[a])
                gpu.step(torch.tensor(actions,device='cuda'),validate=True)
                for i,e in enumerate(cpu):
                    np.testing.assert_allclose(float(gpu.time[i]),e.time,rtol=2e-6,atol=1e-4)
                    self.assertEqual(e.customer_state.tolist(),gpu.customer[i].tolist())
                if all(e.done for e in cpu): break
            self.assertTrue(bool(gpu.done.all()))
            for key,values in gpu.metrics().items():
                if key not in cpu[0].metrics(): continue
                for i,e in enumerate(cpu): np.testing.assert_allclose(float(values[i]),e.metrics()[key],rtol=2e-6,atol=1e-4)


if __name__=='__main__': unittest.main()
