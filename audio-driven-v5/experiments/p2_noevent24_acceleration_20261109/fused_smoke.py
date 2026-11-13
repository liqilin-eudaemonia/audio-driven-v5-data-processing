import torch,copy,json
from pathlib import Path
torch.cuda.set_device(0)
p=torch.nn.Parameter(torch.tensor([.2,-.3],device='cuda'));old=torch.optim.AdamW([p],lr=1e-4)
p.grad=torch.tensor([.1,.2],device='cuda');old.step();state=copy.deepcopy(old.state_dict())
q=torch.nn.Parameter(p.detach().clone());new=torch.optim.AdamW([q],lr=1e-4,fused=True);new.load_state_dict(state)
for g in new.param_groups:g.update(fused=True,foreach=False)
for param,s in new.state.items():s['step']=s['step'].to(param.device)
for i in range(10):
 p.grad=torch.tensor([.1+i/10,.2],device='cuda');q.grad=p.grad.clone();old.step();new.step()
assert torch.allclose(p,q,rtol=1e-6,atol=1e-8)
Path(__file__).with_name('FUSED_SMOKE_ACCEPTED.json').write_text(json.dumps(dict(success=True,updates=10,max_parameter_difference=float((p-q).abs().max()),step_device=str(new.state[q]['step'].device))))
