import copy,tempfile,unittest
from pathlib import Path
import torch
from p2_loss_search28.config import make,validate_cluster,CANDIDATES
from p2_listen.contracts import read,atomic_json,sha
class Review(unittest.TestCase):
 def test_cluster_rejects_reserved_gpu_network_and_entry_changes(self):
  for name in CANDIDATES:
   c=make(name);cl=read(Path(c['output'])/'cluster.json');validate_cluster(c,cl)
   for field in ('gpu','network','script','port'):
    bad=copy.deepcopy(cl)
    if field=='gpu':bad['nodes'][-1]['gpus']=[4,5,6,7]
    elif field=='network':bad['network']['NCCL_IB_HCA']='mlx5_7'
    elif field=='script':bad['training_script']='other.py'
    else:bad['master_port']+=1
    with self.assertRaises(ValueError):validate_cluster(c,bad)
 def test_stage_rejects_other_candidate_source_or_missing_rng(self):
  from p2_loss_search28.checkpoint import validate_stage
  c=make('baseline');v=dict(step=500,identity=dict(experiment=c,source={'p':'h'},world=28),rank_rng=[{}]*28)
  validate_stage(v,c,500,{'p':'h'})
  for mutation in ('candidate','source','world','rng'):
   b=copy.deepcopy(v)
   if mutation=='candidate':b['identity']['experiment']=make('event100')
   elif mutation=='source':b['identity']['source']={}
   elif mutation=='world':b['identity']['world']=31
   else:b['rank_rng'].pop()
   with self.assertRaises(ValueError):validate_stage(b,c,500,{'p':'h'})
 def test_nonfinite_adam_is_rejected_even_with_matching_checksum(self):
  from p2_loss_search28.checkpoint import checked
  from p2_loss_search28 import VERSION
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'step_000001.pt';value=dict(schema=VERSION,step=1,model={'p':torch.ones(1)},optimizer={'state':{0:{'step':torch.tensor(1.),'exp_avg':torch.tensor(float('nan'))}}})
   torch.save(value,p);atomic_json(p.with_suffix('.complete.json'),dict(step=1,sha256=sha(p)))
   with self.assertRaisesRegex(ValueError,'Adam'):checked(p)
 def test_wrong_adam_step_is_rejected(self):
  from p2_loss_search28.checkpoint import checked
  from p2_loss_search28 import VERSION
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'step_000002.pt';torch.save(dict(schema=VERSION,step=2,model={'p':torch.ones(1)},optimizer={'state':{0:{'step':torch.tensor(1.),'exp_avg':torch.zeros(1)}}}),p)
   atomic_json(p.with_suffix('.complete.json'),dict(step=2,sha256=sha(p)))
   with self.assertRaisesRegex(ValueError,'Adam step'):checked(p)
 def test_initialization_gate_detects_tensor_change(self):
  from unittest.mock import patch
  import p2_loss_search28.checkpoint as C
  current=dict(model={'x':torch.ones(1)},optimizer=dict(state={},param_groups=[]));baseline=copy.deepcopy(current)
  with patch.object(C,'checked',side_effect=[current,baseline]),patch.object(C,'atomic_json')as write:C.check_initialization(make('event100'));self.assertTrue(write.called)
  baseline['model']['x'][0]=2
  with patch.object(C,'checked',side_effect=[current,baseline]),patch.object(C,'atomic_json'):
   with self.assertRaisesRegex(ValueError,'initial model'):C.check_initialization(make('event100'))
if __name__=='__main__':unittest.main()
