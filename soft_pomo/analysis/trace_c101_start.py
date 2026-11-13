"""Trace current-code C101 decisions after first customer20; no LLM calls."""
import json
from diagnose_final_runs import ROOT, trainer, soft, torch, load_solomon


def main():
    torch.set_num_threads(4)
    obj,_=trainer('c1','c102-c109',ROOT/'result_soft_matched/with_bias/c102-c109/checkpoint-1000.pt',False)
    obj.trainer_params['start_mode']='det_conf'
    obj.llm_params['bias_strength']=5
    rows=[]
    aug_index=0

    def hook(module,args,kwargs,out):
        if obj.env.selected_count != 2:
            return
        for p in (obj.env.selected_node_list[0,:,-1]==20).nonzero().flatten().tolist():
            bias=kwargs['llm_bias'][0,p]
            probs=out[0,p]
            delta=(probs[24].log()-probs[0].log()).item()
            rows.append(dict(aug=aug_index,rollout=p,selected=int(probs.argmax()),
                             probability_depot=float(probs[0]),probability_24=float(probs[24]),
                             bias_depot=float(bias[0]),bias_24=float(bias[24]),
                             raw_logit_24_minus_depot=delta-float(bias[24]-bias[0]),
                             final_logit_24_minus_depot=delta))

    handle=obj.model.decoder.register_forward_hook(hook,with_kwargs=True)
    inst=load_solomon(str(ROOT.parent/'data/Solomon/c101.txt'))
    for aug_index in range(8):
        soft._set_seed(42)
        with torch.inference_mode():
            obj._best_solution(obj._aug_instance(inst,aug_index))
    handle.remove()
    print(json.dumps(rows,indent=2))
    (ROOT/'analysis/c101_start_trace.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
