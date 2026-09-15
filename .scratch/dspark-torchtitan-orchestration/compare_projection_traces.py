"""Find the first mismatched gradient boundary in actual TP phase traces."""
from pathlib import Path
import time
import torch
base=Path('output/dspark_torchtitan_orchestration_20260914')
root=base/'gqa-fp32-trace-tp4/float32'
deadline=time.monotonic()+600
while not all((root/f'projections-rank{i}.pt').is_file() for i in range(8)):
    if time.monotonic()>deadline: raise TimeoutError('Native projection traces not written')
    time.sleep(1)
for dp in range(2):
    reference=torch.load(base/'gqa-fp32-trace-tp1/float32'/f'projections-rank{dp}.pt',weights_only=True)
    ranks=[torch.load(root/f'projections-rank{dp*4+i}.pt',weights_only=True) for i in range(4)]
    for name in reversed(reference):
        attr=name.rsplit('.',1)[-1]
        row=attr in ('o_proj','down_proj')
        col=attr in ('q_proj','k_proj','v_proj','gate_proj','up_proj')
        heads=attr in ('q_norm','k_norm')
        for i,entry in enumerate(reference[name]):
            for key,expected in entry.items():
                values=[r[name][i][key] for r in ranks]
                axis=None
                if key=='weight': axis=1 if row else 0 if col else None
                elif heads: axis=-2
                elif key=='input' and row: axis=-1
                elif key!='input' and col: axis=-1
                actual=torch.cat(values,axis) if axis is not None else values[0]
                difference=(actual-expected).abs()
                if difference.count_nonzero():
                    print('DP',dp,name,'call',i,key,'different',difference.count_nonzero().item(),'max',difference.max().item(),flush=True)
