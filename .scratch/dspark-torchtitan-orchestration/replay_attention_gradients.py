"""Replay actual FlexAttention inputs with global and local head counts."""
from pathlib import Path
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from torchtitan.models.dspark_draft.model import Qwen3DSparkModel, ALL_ATTENTION_FUNCTIONS
from torchtitan.models.dspark_draft.loss import _collect_local_terms, _build_loss
root=Path('output/dspark_torchtitan_orchestration_20260914')
fixture=torch.load(root/'gqa-reference/torch.float32_rank0.pt',weights_only=True)
torch.backends.cuda.matmul.fp32_precision='ieee'
config=Qwen3_5TextConfig.from_dict(fixture['model_config']); config._attn_implementation='flex_attention'
model=Qwen3DSparkModel(config).cuda(); model.load_state_dict(fixture['initial_weights']); model.set_embedding_head_trainable(False)
original=ALL_ATTENTION_FUNCTIONS['flex_attention']
records=[]
def capture(module,q,k,v,mask,**kwargs):
    output,lse=original(module,q,k,v,mask,**kwargs)
    item={'module':module,'inputs':[q.detach(),k.detach(),v.detach()],'mask':mask,'kwargs':kwargs,'output':output.detach(),'lse':lse.detach()}
    records.append(item)
    output.register_hook(lambda grad:item.__setitem__('grad_output',grad.detach()))
    return output,lse
ALL_ATTENTION_FUNCTIONS.register('flex_attention',capture)
torch.set_rng_state(fixture['initial_cpu_rng']); torch.cuda.set_rng_state(fixture['initial_cuda_rng'])
output=model(**{k:v.cuda() for k,v in fixture['features'][0].items()})
terms,confidence=_collect_local_terms(outputs=output,loss_decay_gamma=4.,l1_loss_alpha=.9)
den=fixture['result']['microbatches'][0]['denominator'].cuda()
loss=_build_loss(loss_terms=terms,global_denominators={k:den for k in ('ce_loss_den','l1_loss_den','confidence_loss_den')},ce_loss_alpha=.1,l1_loss_alpha=.9,confidence_head_alpha=1.,has_confidence=confidence,world_size=1)/2
loss.backward()
for index,item in enumerate(records):
    reference_inputs=[v.detach().requires_grad_() for v in item['inputs']]
    y,lse=original(item['module'],*reference_inputs,item['mask'],**item['kwargs'])
    expected=torch.autograd.grad(y,reference_inputs,item['grad_output'])
    parts=[]; outputs=[]; lses=[]
    for rank in range(4):
        inputs=[]
        for j,value in enumerate(item['inputs']):
            local=value.chunk(4,1)[rank].detach()
            local=local.transpose(1,2).contiguous().transpose(1,2) if j==0 else local.contiguous()
            inputs.append(local.requires_grad_())
        actual,local_lse=original(item['module'],*inputs,item['mask'],**item['kwargs'])
        grad=item['grad_output'].chunk(4,2)[rank].contiguous()
        parts.append(torch.autograd.grad(actual,inputs,grad))
        outputs.append(actual.detach()); lses.append(local_lse.detach())
    for name,actual,reference in [('output',torch.cat(outputs,2),y),('lse',torch.cat(lses,1),lse)]+[(f'grad_{j}',torch.cat([p[j] for p in parts],1),expected[j]) for j in range(3)]:
        difference=(actual-reference).abs()
        print('layer',index,name,'count',difference.count_nonzero().item(),'max',difference.max().item(),flush=True)
    torch.save({k:v.cpu() for k,v in {'q':reference_inputs[0],'k':reference_inputs[1],'v':reference_inputs[2],'out':y,'grad_out':item['grad_output'],**{f'grad_{j}':v for j,v in enumerate(expected)}}.items()},root/f'flex-layer{index}-replay.pt')
