"""Compare the actual per-device compiled FlexAttention delta reductions."""
from pathlib import Path
import importlib.util
import re
import torch
base=Path('output/dspark_torchtitan_orchestration_20260914')
ref=torch.load(base/'gqa-fp32-trace-tp1/float32/projections-rank0.pt',weights_only=True)
for rank in range(4):
    actual=torch.load(base/'gqa-fp32-trace-tp4/float32'/f'projections-rank{rank}.pt',weights_only=True)
    key='layers.1.self_attn.q_norm'
    a=actual[key][0]['grad_output']; b=ref[key][0]['grad_output'].chunk(4,2)[rank]
    print('rank',rank,'q_norm_grad_diff',(a-b).abs().count_nonzero().item(),flush=True)
record=torch.load(base/'flex-layer1-replay.pt',weights_only=True)
cache=Path('/tmp/torchinductor_root')
wrappers=['6r/c6rhssrdfitbxk6btpmrm3noryt6lfjpemcnuuwsyvoqyg77hxgf.py','nn/cnnor7exayr3ifaih6ylgxnryzeh6yu427xbt5tvd4e2w6kgklsd.py','gm/cgmceqlewl7xm2wneflr4fppxk447drbib7niwyiwp4r6wkqrp2u.py','6n/c6nrgdk5uppiq6c6nfdxuyzi3jyueafns6gtnxch2fsq2i5pqcct.py','yg/cygmzmpure2jlcn3eqv4uc6xbi7hlthog7uqfilunmp4vh6mv6mq.py']
results=[]
for index,wrapper in enumerate(wrappers):
    rank=max(0,index-1)
    torch.cuda.set_device(rank)
    text=(cache/wrapper).read_text()
    source=re.search(r'# kernel path: (.*\.py)',text)[1]
    spec=importlib.util.spec_from_file_location(f'delta_{index}',source)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    o=record['out']; g=record['grad_out']
    if index:
        o=o.chunk(4,2)[rank].contiguous(); g=g.chunk(4,2)[rank].contiguous()
    o=o.cuda().transpose(1,2);g=g.cuda().transpose(1,2)
    h=o.shape[1]; zero=torch.zeros((1,h,6),device='cuda'); out=torch.empty_like(zero)
    module.triton_per_fused_flex_attention_backward_mul_0.run(out,o,g,zero,6,h,16,stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    results.append(out.cpu())
for rank in range(4):
    difference=(results[rank+1]-results[0].chunk(4,1)[rank]).abs()
    print('rank',rank,'delta_difference',difference.count_nonzero().item(),difference.max().item(),flush=True)
