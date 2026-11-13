import copy,tempfile,unittest
from pathlib import Path
import torch
from p2_loss_search28.config import CANDIDATES,make,validate,normalized
from p2_loss_search28.schedule import Schedule
from p2_ltx_events.heads import objective
class LossSearch(unittest.TestCase):
 def test_coefficients_reach_shared_gradient(self):
  for name,(e,f)in CANDIDATES.items():
   x=torch.tensor(2.,requires_grad=True);c=make(name)
   values=dict(av=x,features=dict(video=2*x,audio=3*x),event_sums={'clip/example':4*x})
   loss,parts,event=objective(values,c,{'clip/example':torch.tensor(1.)},1,28)
   loss.backward();self.assertAlmostEqual(x.grad.item(),1/28+4*e+5*f/28,places=5)
 def test_only_allowed_fields_vary(self):
  views=[]
  for n in CANDIDATES:
   c=validate(make(n));v=copy.deepcopy(c)
   for k in ('loss_search','output','loss_weights'):v.pop(k)
   views.append(v)
   for group,k in [('training','global_batch'),('training','connector_lr'),('text_lora','rank')]:
    bad=copy.deepcopy(c);bad[group][k]+=1
    with self.assertRaises(ValueError):validate(bad)
  self.assertTrue(all(x==views[0]for x in views))
 def test_full_epoch_no_lost_samples_or_noise(self):
  c=make('baseline');s=Schedule(c)
  self.assertEqual(len({i for i,o in s.pairs}),284218)
  self.assertEqual(len({o for i,o in s.pairs}),len(s.pairs))
  for step in range(1,c['training']['steps']+1):
   self.assertEqual(s.indices(s.size,step,28,s.seed),[s.indices(s.size,step,28,s.seed,r,28)[0]for r in range(28)])
 def test_checkpoint_roundtrip_and_wrong_candidate(self):
  from p2_loss_search28.checkpoint import save,resume,checked
  from p2_listen.training import rng_state
  class Model(torch.nn.Linear):
   def assert_ownership(self):pass
  m=Model(2,1);opt=torch.optim.AdamW(m.parameters());m(torch.ones(1,2)).sum().backward();opt.step()
  ident={'experiment':make('baseline')}
  with tempfile.TemporaryDirectory()as d:
   path=Path(d)/'step_000001.pt';save(path,m,opt,1,ident,[rng_state()])
   original=m.weight.detach().clone();m.weight.data.zero_()
   self.assertEqual(resume(path,m,opt,ident),1);self.assertTrue(torch.equal(m.weight,original))
   with self.assertRaises(ValueError):resume(path,m,opt,{'experiment':make('event100')})
 def test_no_implicit_promotion(self):
  from p2_loss_search28.queue import stages
  self.assertEqual(list(stages()),[(500,list(CANDIDATES))])
if __name__=='__main__':unittest.main()
