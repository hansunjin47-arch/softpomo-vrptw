"""Launch the normal trainer while prohibiting all LLM generation."""
import pathlib
import runpy
import sys

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import SoftClusterLLMModule
import VRPTWLLMModule

def reject(*args,**kwargs):
    raise RuntimeError('This experiment is cache-only: unexpected LLM call rejected')

SoftClusterLLMModule.query_llm=reject
VRPTWLLMModule.query_llm=reject
runpy.run_path(str(ROOT/'train_soft_cluster.py'),run_name='__main__')
