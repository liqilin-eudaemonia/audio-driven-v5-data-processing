from pathlib import Path
import copy,math
from types import SimpleNamespace
from unittest.mock import patch
import torch
import pytest
from p2_listen.contracts import atomic_json,sha
from p2_stage2_joint_v2.calibration import pair_statistics,sum_reduce_vectors,run,verify_receipt

def reduce_worker(rank,folder):
 import torch.distributed as dist
 dist.init_process_group('gloo',rank=rank,world_size=4,init_method='file://'+str(Path(folder)/'rendezvous'))
 try:
  g=torch.tensor([1.,0.],dtype=torch.float64)if rank==0 else torch.tensor([-1.,1.],dtype=torch.float64)if rank==1 else torch.zeros(2,dtype=torch.float64)
  p=torch.tensor([0.,2.],dtype=torch.float64)if rank==0 else torch.tensor([3.,0.],dtype=torch.float64)if rank==1 else torch.zeros(2,dtype=torch.float64)
  a,b=sum_reduce_vectors(g,p,'cpu',1)
  torch.save(dict(g=a,p=b,stats=pair_statistics(a,b,.0001)),Path(folder)/f'{rank}.pt')
 finally:dist.destroy_process_group()

def test_vector_sum_direction_with_empty_ranks_matches_analytic_reference(tmp_path):
 import torch.multiprocessing as mp
 mp.spawn(reduce_worker,args=(str(tmp_path),),nprocs=4,join=True)
 for rank in range(4):
  x=torch.load(tmp_path/f'{rank}.pt',weights_only=True)
  assert torch.equal(x['g'],torch.tensor([0.,1.],dtype=torch.float64))
  assert x['stats']['generation_norm']==1
  assert x['stats']['weighted_planning_to_generation']==pytest.approx(.0001*math.sqrt(13))
  assert x['stats']['cosine']==pytest.approx(2/math.sqrt(13))

def test_native_all_lora_calibration_is_read_only_and_weight_scaling_exact(tmp_path):
 from test_p2_stage2_joint_v2 import joint_fixture
 m,g,b,pref,strategy,sampler=joint_fixture('full');b['planning_tokens']=4
 original=copy.deepcopy(m.state_dict());rng=torch.random.get_rng_state().clone()
 records=[dict(index=i,sample_id=str(i))for i in range(8)];p=tmp_path/'manifest.json';atomic_json(p,dict(samples=records))
 class Dataset:
  ids=list(map(str,range(8)));planning_ids=set(ids)
  def sample(self,index):return copy.deepcopy(b)
 c=dict(generator=g,planning_weight=.0001,global_batch=31,calibration=dict(samples=8,seed=7,chunk_elements=8,target_ratio=.2,max_global_ratio=1.,reference_band=[.1,.3]),calibration_manifest=str(p),calibration_manifest_sha256=sha(p))
 pos={k:1. for k in b['events']}
 with patch('p3_listen.features.prefix',return_value=pref):r=run(m,c,Dataset(),strategy,sampler,pos,list(pos),'cpu')
 assert r['samples']==8 and r['world']==1 and r['global_planning_tokens']==32
 assert len(r['per_sample'])==8 and r['optimizer_updates']==0
 assert r['aggregate']['generation_norm']>0 and r['aggregate']['planning_unweighted_norm']>0
 assert r['coefficient_sweep']['0.01']==pytest.approx(r['coefficient_sweep']['0.0001']*100)
 assert torch.equal(rng,torch.random.get_rng_state())
 assert all(torch.equal(v,m.state_dict()[k])for k,v in original.items())
 assert all(p.grad is None for p in m.parameters())

def test_receipt_cannot_be_reused_with_different_config_or_failed_calibration(tmp_path):
 from p2_listen.contracts import digest
 c=dict(output=str(tmp_path),planning_weight=.0001,calibration_manifest_sha256='m',global_batch=31)
 record=dict(passed=True,configuration_sha256=digest(c),source_sha256=digest({}),manifest_sha256='m',planning_weight=.0001,world=31)
 atomic_json(tmp_path/'CALIBRATION.json',record)
 with patch('p2_stage2_joint_v2.config.source_identity',return_value={}):
  assert verify_receipt(c)['passed']
  with pytest.raises(ValueError):verify_receipt(dict(c,planning_weight=.001))
  record['passed']=False;atomic_json(tmp_path/'CALIBRATION.json',record)
  with pytest.raises(ValueError):verify_receipt(c)
