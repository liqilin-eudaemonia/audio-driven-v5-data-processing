import sys,unittest
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from p2_testsets_eval.batch_sampling import pack,sample
from p2_listen.runtime import guided_sample

@dataclass
class Mod:
 latent:torch.Tensor
 sigma:torch.Tensor
 timesteps:torch.Tensor
 positions:torch.Tensor
 context:torch.Tensor
 context_mask:torch.Tensor
 enabled:bool=True
 attention_mask:object=None

class Renderer:
 def eval(self):return self
 def __call__(self,video,audio,perturbations):
  def run(m):
   c=(m.context*m.context_mask[...,None]).sum((1,2))/m.context_mask.sum(1)
   return m.latent*.125+c[:,None,None]*.03+m.sigma[:,None,None]*.01
  return run(video),run(audio)

class BatchSamplingTests(unittest.TestCase):
 def test_batching_preserves_rng_masked_conditions_and_pinned_frames(self):
  ps=[];ns=[]
  for i in range(2):
   v=Mod(torch.full((1,3,6),float(i+1)).transpose(1,2),torch.ones(1),torch.ones(1,6),torch.zeros(1,3,6),torch.full((1,2+i,3),float(i+1)),torch.ones(1,2+i,dtype=torch.bool))
   a=Mod(torch.full((1,2,4),float(i+1)).transpose(1,2),torch.ones(1),torch.ones(1,4),torch.zeros(1,1,4),torch.full((1,3+i,2),float(i+2)),torch.ones(1,3+i,dtype=torch.bool))
   mask=torch.ones(1,6,dtype=torch.bool);mask[:,:2]=False
   ps.append(SimpleNamespace(video=v,audio=a,video_loss_mask=mask))
   ns.append(dict(video=torch.zeros(1,1+i,3),video_mask=torch.ones(1,1+i,dtype=torch.bool),audio=torch.zeros(1,2+i,2),audio_mask=torch.ones(1,2+i,dtype=torch.bool)))
  p,n=pack(ps,ns);self.assertFalse(p.video.context_mask[0,-1]);self.assertFalse(n['audio_mask'][0,-1])
  sigmas=torch.linspace(1,0,4);baseline=[]
  for current,neg in zip(ps,ns):
   torch.manual_seed(137);baseline.append(guided_sample(Renderer(),current,neg,3,3,7,sigmas))
  module=SimpleNamespace(LTX2Scheduler=lambda:SimpleNamespace(execute=lambda steps:sigmas))
  with patch.dict(sys.modules,{'ltx_core.components.schedulers':module}):
   for combined in [False,True]:
    result=sample(Renderer(),p,n,3,3,7,combine_cfg=combined)
    for key in result:torch.testing.assert_close(result[key],torch.cat([x[key] for x in baseline]),rtol=0,atol=0)
    torch.testing.assert_close(result['video'][:,:2],p.video.latent[:,:2],rtol=0,atol=0)
if __name__=='__main__':unittest.main()
