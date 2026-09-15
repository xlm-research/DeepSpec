"""Check fixed compiler policies against the captured reference reduction."""
import importlib.util
from pathlib import Path
import torch
base=Path('output/dspark_torchtitan_orchestration_20260914')
r=torch.load(base/'flex-layer1-replay.pt',weights_only=True)
o=r['out'].cuda().transpose(1,2); g=r['grad_out'].cuda().transpose(1,2)
z=torch.zeros((1,24,6),device='cuda'); expected=torch.empty_like(z)
p='/tmp/torchinductor_root/rb/crbwyk4sy4chdxdcfelw376irnqq2qnaiclfbq4dblotwaudzhfx.py'
s=importlib.util.spec_from_file_location('baseline_delta',p); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.triton_per_fused_flex_attention_backward_mul_0.run(expected,o,g,z,6,24,16,stream=torch.cuda.current_stream().cuda_stream)
def delta(o,g,z):
    return (o*g).sum(-1)-z*0.6931471805599453*1.4426950408889634
for options in [{'deterministic':True},{'deterministic':True,'triton.max_tiles':1},{'batch_invariant':True,'triton.max_tiles':1},{'deterministic':True,'triton.prefer_nd_tiling':True}]:
    fn=torch.compile(delta,dynamic=False,fullgraph=True,options=options)
    chunks=[]
    for i in range(4):
        a=o.chunk(4,1)[i].transpose(1,2).contiguous().transpose(1,2)
        b=g.chunk(4,1)[i].transpose(1,2).contiguous().transpose(1,2)
        chunks.append(fn(a,b,z.chunk(4,1)[i].contiguous()))
    d=(torch.cat(chunks,1)-expected).abs()
    print(options,'nonzero',d.count_nonzero().item(),'max',d.max().item(),flush=True)
for name,value in [('eager',delta(o,g,z)),('fp64',(o.double()*g.double()).sum(-1).float())]:
    d=(value-expected).abs();print(name,d.count_nonzero().item(),d.max().item(),flush=True)
