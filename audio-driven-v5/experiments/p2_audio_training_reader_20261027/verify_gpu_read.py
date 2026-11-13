import sys
from pathlib import Path
import torch
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path[:0]=[str(ROOT/'releases/p2_audio_postprocess_v1/src'),str(ROOT/'src')]
from p2_listen.config import load
from p2_listen.contracts import read,atomic_json
from p2_listen.training_data import load_training_sample
from p2_omni_eval.common import check_device
check_device('cuda:0');results=[]
for name in ['p2_train50h_audio_postprocess_20261026','p2_full_audio_postprocess_crop384_20261026']:
 view=OUT/(name+'_view');c=load(view/'config.json');obs={o['sample_id']:o for o in read(view/'manifests/observations.json')};targets={t['sample_id']:t for t in read(view/'manifests/targets.json')};rows=read(OUT/(name+'.json'))['rows']
 for kind in ['changed','reused']:
  for row in [r for r in rows if r['audio']==kind][:2]:
   sid=row['sample_id'];cpu=load_training_sample(c,obs[sid],targets[sid]);gpu=load_training_sample(c,obs[sid],targets[sid],'cuda:0')
   for key in ['latents','audio_latents']:
    value=gpu[3]['av'][key]['latents'];assert value.is_cuda;torch.testing.assert_close(value.cpu(),cpu[3]['av'][key]['latents'],rtol=0,atol=0)
   for mod in gpu[4]:
    for k,v in gpu[4][mod].items():
     if isinstance(v,torch.Tensor):assert v.is_cuda;torch.testing.assert_close(v.cpu(),cpu[4][mod][k],rtol=0,atol=0)
   assert gpu[2] is None;results.append(dict(dataset=name,sample_id=sid,audio=kind));del cpu,gpu
atomic_json(OUT/'gpu_read_validation.json',dict(status='passed',samples=len(results),rows=results,models_loaded=False,training_started=False));print('GPU read passed',len(results))
