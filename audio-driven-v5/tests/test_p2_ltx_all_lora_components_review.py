"""Review gaps: each new arm's recomputation and actual CFG inference path."""
import pytest
import torch

from test_p2_ltx_all_lora_components import ARMS, build, forward, objective
from test_p2_whisper_av import opened


@pytest.mark.parametrize('arm', ARMS)
def test_all_recomputation_matches_loss_and_every_gradient(arm):
    from p2_listen.performance import configure_renderer
    c, _, model, bundle, strategy, sampler = build(arm)
    if (arm=='no_feature'):
        opened(model)
    model.train()
    results = []
    counts = {k: v['known'].sum() for k, v in bundle.get('events', {}).items()}
    for policy in ('none', 'all'):
        configure_renderer(model.renderer, policy)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(731)
        values = forward(model, bundle, strategy, sampler)
        loss = objective(values, c, counts)[0]
        loss.backward()
        results.append((loss.detach().clone(), {
            n: None if p.grad is None else p.grad.detach().clone()
            for n, p in model.named_parameters()
        }))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for name, grad in results[0][1].items():
        other = results[1][1][name]
        if grad is None:
            assert other is None, name
        else:
            torch.testing.assert_close(grad, other, rtol=0, atol=0, msg=name)



from pathlib import Path
from dataclasses import replace
from p2_ltx_all_lora_components import training
from test_p2_current_component_ablation import _distributed_values

def worker(rank,arm,folder):
 import torch.distributed as dist
 from p2_whisper_av.execution import StreamingFP64Reduction
 torch.set_num_threads(1);dist.init_process_group('gloo',rank=rank,world_size=2,init_method='file://'+str(Path(folder)/'rendezvous'))
 try:
  x=torch.nn.Parameter(torch.tensor([.2,-.4]));o=torch.optim.AdamW([x],lr=.001,weight_decay=0.)
  for _ in range(3):
   o.zero_grad(set_to_none=True);v,labels,c=_distributed_values(x,arm,rank)
   counts=training._event_counts([{'events':labels}],list(labels),'cpu',2)
   objective(v,c,counts,1,2)[0].backward();acc=StreamingFP64Reduction([x],1);acc.add();acc.finish(2);o.step()
  torch.save(dict(parameter=x.detach(),optimizer=o.state_dict()),Path(folder)/f'rank{rank}.pt')
 finally:dist.destroy_process_group()

@pytest.mark.parametrize('arm',ARMS)
def test_new_training_global_event_denominator_and_adam(arm,tmp_path):
 import torch.multiprocessing as mp
 mp.spawn(worker,args=(arm,str(tmp_path)),nprocs=2,join=True)
 x=torch.nn.Parameter(torch.tensor([.2,-.4]));o=torch.optim.AdamW([x],lr=.001,weight_decay=0.)
 for _ in range(3):
  items=[_distributed_values(x,arm,r)for r in range(2)];counts=training._event_counts([{'events':labels}for v,labels,c in items],list(items[0][1]),'cpu',1);grads=[]
  for v,labels,c in items:
   o.zero_grad(set_to_none=True);objective(v,c,counts,1,2)[0].backward();grads.append(x.grad.double().clone())
  x.grad=sum(grads).float();o.step()
 for rank in range(2):
  saved=torch.load(tmp_path/f'rank{rank}.pt',weights_only=True)
  torch.testing.assert_close(saved['parameter'],x,rtol=0,atol=0)
  for k,v in o.state_dict()['state'][0].items():torch.testing.assert_close(saved['optimizer']['state'][0][k],v,rtol=0,atol=0)

def test_whisper_changes_both_modalities_and_masks_padding():
 from omni_av.ltx_runtime import with_text
 c,_,m,b,s,sa=build('no_feature');opened(m);m.eval()
 prepared=s.prepare_training_inputs(with_text(b['av']['av'],b['text']['positive']),sa)
 u=m.connector(b['condition']);speaker=b['whisper']
 def run(sp):
  v=m.prepare(prepared,u,sp)
  with torch.no_grad():return m.renderer(video=v.video,audio=v.audio,perturbations=None)
 a=run(speaker);masked=speaker.hidden.clone();masked[~speaker.valid]=1000
 z=run(replace(speaker,hidden=masked));assert all(torch.equal(x,y)for x,y in zip(a,z))
 changed=speaker.hidden.clone();changed[speaker.valid]+=1
 z=run(replace(speaker,hidden=changed));assert all(not torch.allclose(x,y)for x,y in zip(a,z))

@pytest.mark.parametrize('arm',ARMS)
def test_mismatched_cli_arm_rejected_before_training(arm,monkeypatch):
 import sys
 import p2_ltx_all_lora_components_train as cli
 from p2_ltx_all_lora_components.config import DEFAULT_OUTPUT
 monkeypatch.setattr(training,'train',lambda *a:pytest.fail('wrong experiment launched'))
 other=next(a for a in ARMS if a!=arm)
 monkeypatch.setattr(sys,'argv',['cli','train','--arm',other,'--config',str(DEFAULT_OUTPUT/arm/'config.json')])
 with pytest.raises(SystemExit)as e:cli.main()
 assert e.value.code==2
