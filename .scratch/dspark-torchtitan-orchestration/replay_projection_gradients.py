"""Replay captured real projection VJPs with complete TP contractions."""
from pathlib import Path
import torch
root=Path('output/dspark_torchtitan_orchestration_20260914/gqa-fp32-trace-tp1/float32')
torch.backends.cuda.matmul.fp32_precision='ieee'
traces=torch.load(root/'projections-rank0.pt',weights_only=True)
for name,entries in traces.items():
    for index,item in enumerate(entries):
        weight=item['weight'].cuda().requires_grad_()
        if weight.ndim != 2: continue
        x=item['input'].cuda().requires_grad_(); grad=item['grad_output'].cuda()
        output=torch.nn.functional.linear(x,weight)
        dx,dw=torch.autograd.grad(output,(x,weight),grad)
        x=x.detach().reshape(-1,x.shape[-1]); g=grad.reshape(-1,grad.shape[-1]); weight=weight.detach()
        if name.rsplit('.',1)[-1] in ('o_proj','down_proj'):
            actual_dx=torch.cat([g@w.contiguous() for w in weight.chunk(4,1)],1).reshape_as(dx)
        else:
            actual_dx=torch.cat([g@w.contiguous() for w in weight.chunk(4,1)],1).reshape_as(dx)
        actual_dw=torch.cat([g.T@part for part in x.chunk(4,1)],1)
        for label,actual,expected in [('dx',actual_dx,dx),('dw',actual_dw,dw)]:
            d=(actual-expected).abs()
            if d.count_nonzero():
                print(name,index,label,'shape',tuple(x.shape),tuple(weight.shape),'count',d.count_nonzero().item(),'max',d.max().item(),flush=True)
print('done',flush=True)
