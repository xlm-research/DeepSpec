"""Diagnose TP rounding using actual DSpark activations and archived weights."""
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from torchtitan.models.dspark_draft.model import Qwen3DSparkModel

root=Path('output/dspark_torchtitan_orchestration_20260914')
fixture=torch.load(root/'gqa-reference/torch.bfloat16_rank0.pt',weights_only=True)
torch.backends.cuda.matmul.fp32_precision='ieee'
config=Qwen3_5TextConfig.from_dict(fixture['model_config'])
config._attn_implementation='flex_attention'
model=Qwen3DSparkModel(config).cuda().bfloat16()
model.load_state_dict(fixture['initial_weights'])
model.set_embedding_head_trainable(False)
seen=set()
def hook(name,module,args,output):
    if name in seen: return
    seen.add(name)
    value=args[0].detach(); weight=module.weight.detach()
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
