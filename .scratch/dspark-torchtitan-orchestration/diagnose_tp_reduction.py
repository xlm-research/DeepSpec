"""Diagnose TP rounding using actual DSpark activations and archived weights."""
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from torchtitan.models.dspark_draft.model import Qwen3DSparkModel

root=Path('output/dspark_torchtitan_orchestration_20260914')
dtype=getattr(torch,os.environ.get('DEEPSPEC_DIAGNOSTIC_DTYPE','bfloat16'))
fixture=torch.load(root/'gqa-reference'/f'{dtype}_rank0.pt',weights_only=True)
torch.backends.cuda.matmul.fp32_precision='ieee'
config=Qwen3_5TextConfig.from_dict(fixture['model_config'])
config._attn_implementation='flex_attention'
model=Qwen3DSparkModel(config).cuda().to(dtype)
model.load_state_dict(fixture['initial_weights'])
model.set_embedding_head_trainable(False)
seen=set()
records={}
def hook(name,module,args,output):
    if name in seen: return
    seen.add(name)
    value=args[0].detach(); weight=module.weight.detach()
    records[name]=[value,weight,None]
    output.register_hook(lambda grad: records[name].__setitem__(2,grad.detach()))
    if name.rsplit('.',1)[-1] in ('o_proj','down_proj'):
        chunks=[F.linear(x.float(),w.float()) for x,w in zip(value.chunk(4,-1),weight.chunk(4,1))]
        actual=(chunks[0]+chunks[1]+chunks[2]+chunks[3]).to(value.dtype)
        accurate=F.linear(value.double(),weight.double()).to(value.dtype)
        kind='row'
    else:
        actual=torch.cat([F.linear(value,w) for w in weight.chunk(4,0)],-1)
        accurate=F.linear(value.double(),weight.double()).to(value.dtype)
        kind='column'
    difference=(actual-output).abs()
    if difference.count_nonzero():
        print(name,kind,'shape',tuple(value.shape),tuple(weight.shape),'different',difference.count_nonzero().item(),'max',difference.max().item(),'fp64_vs_reference',(accurate-output).abs().count_nonzero().item(),flush=True)
        torch.save({'input':value.cpu(),'weight':weight.cpu(),'reference':output.detach().cpu()},root/(name+'-rounding.pt'))
for name,module in model.named_modules():
    if name.startswith('layers.') and isinstance(module,torch.nn.Linear):
        module.register_forward_hook(lambda m,a,o,n=name:hook(n,m,a,o))
torch.set_rng_state(fixture['initial_cpu_rng'])
torch.cuda.set_rng_state(fixture['initial_cuda_rng'])
batch={k:v.cuda() for k,v in fixture['features'][0].items()}
output=model(**batch)
print('full_model_first_logits_equal',torch.equal(output.draft_logits.cpu(),fixture['result']['microbatches'][0]['output']['draft_logits']),flush=True)

from torchtitan.models.dspark_draft.loss import _collect_local_terms, _build_loss
terms,has_confidence=_collect_local_terms(outputs=output,loss_decay_gamma=4.,l1_loss_alpha=.9)
den=fixture['result']['microbatches'][0]['denominator'].cuda()
loss=_build_loss(loss_terms=terms,global_denominators={k:den for k in ('ce_loss_den','l1_loss_den','confidence_loss_den')},ce_loss_alpha=.1,l1_loss_alpha=.9,confidence_head_alpha=1.,has_confidence=has_confidence,world_size=1)/2
loss.backward()
for name,(value,weight,grad) in records.items():
    grad=grad.reshape(-1,grad.shape[-1]); value=value.reshape(-1,value.shape[-1])
    expected_input=grad @ weight
    expected_weight=grad.T @ value
    if name.rsplit('.',1)[-1] in ('o_proj','down_proj'):
        actual_input=torch.cat([grad.float() @ w.float() for w in weight.chunk(4,1)],-1).to(value.dtype)
        actual_weight=torch.cat([grad.float().T @ x.float() for x in value.chunk(4,1)],1).to(value.dtype)
    else:
        partials=[g.float() @ w.float() for g,w in zip(grad.chunk(4,-1),weight.chunk(4,0))]
        actual_input=(partials[0]+partials[1]+partials[2]+partials[3]).to(value.dtype)
        actual_weight=torch.cat([g.T @ value for g in grad.chunk(4,-1)],0)
    complete_input=torch.cat([grad @ w for w in weight.chunk(4,1)],-1)
    complete_weight=torch.cat([grad.T @ x for x in value.chunk(4,1)],1)
    for label,actual,expected in [('input_gradient',actual_input,expected_input),('weight_gradient',actual_weight,expected_weight)]:
        d=(actual-expected).abs()
        if d.count_nonzero():
            print(name,label,'different',d.count_nonzero().item(),'max',d.max().item(),'complete_reduction_difference',((complete_input if label=='input_gradient' else complete_weight)-expected).abs().max().item(),flush=True)
            torch.save({'input':value.cpu(),'weight':weight.cpu(),'grad':grad.cpu(),'reference_input_gradient':expected_input.cpu(),'reference_weight_gradient':expected_weight.cpu()},root/(name+'-backward-rounding.pt'))
