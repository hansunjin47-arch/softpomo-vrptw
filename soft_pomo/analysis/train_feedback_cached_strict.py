"""Cache-only continuation that rejects model/optimizer incompatibility."""
import pathlib
import runpy
import sys

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import train_vrptw_llm as trainer
import SoftClusterLLMModule
import VRPTWLLMModule


def strict_model(model,state):
    model.load_state_dict(state,strict=True)


def strict_optimizer(optimizer,model,state):
    # The parent trainer catches ValueError/RuntimeError; AssertionError must stop it.
    try:
        optimizer.load_state_dict(state)
        for param,values in optimizer.state.items():
            for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
                if key in values and values[key].shape != param.shape:
                    raise AssertionError(f'Optimizer shape mismatch: {key}')
    except (ValueError,RuntimeError) as exc:
        raise AssertionError('Strict optimizer resume failed') from exc
    print('[Strict resume] Model and optimizer restored without fallback',flush=True)


def reject(*args,**kwargs):
    raise AssertionError('Cache-only run: unexpected LLM generation is prohibited')


trainer._load_model_flexible=strict_model
trainer._load_optimizer_flexible=strict_optimizer
SoftClusterLLMModule.query_llm=reject
VRPTWLLMModule.query_llm=reject
runpy.run_path(str(ROOT/'train_soft_cluster.py'),run_name='__main__')
