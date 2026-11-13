import os,sys,json,copy,time
from pathlib import Path
import numpy as np,torch
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');W=ROOT/'experiments/p2_l2l_reliable_retrain_20261107'
sys.path.insert(0,str(W/'code_snapshot'))
from l2l_reliable_core import backend,factories,tensor,GROUPS,save,sha
from l2l_reliable_data import load
backend();data,stats=load(W);model,opt,_,_=factories(W/'code_snapshot/l2l_src','vq',137)
rows=[]
for name in ['initial','best','best_reconstruction','last']:
 path=W/'seed_137/vq'/(name+'.pth');state=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(state['state_dict']);model.eval()
 row=dict(checkpoint=name,step=state['step'],sha256=sha(path),modes={})
 for mode in ['eval','train']:
  model.load_state_dict(state['state_dict']);model.train(mode=='train');target=tensor(data['validation']['vq'][:32])
  h=model.module.encoder(target);q,quant,_=model.module.quantize(h);prediction=model.module.decode(q)
  parts={k:(prediction[:,:,sl]-target[:,:,sl]).abs().mean() for k,sl in GROUPS.items()};rec=sum(parts.values())
  rh=torch.autograd.grad(rec,h,retain_graph=True)[0];qh,qe=torch.autograd.grad(quant,[h,model.module.quantize.embedding.weight],retain_graph=False)
  row['modes'][mode]=dict(reconstruction=float(rec),quantization=float(quant),groups={k:float(v) for k,v in parts.items()},encoder_rms=float(h.square().mean().sqrt()),
   codebook_rms=float(model.module.quantize.embedding.weight.square().mean().sqrt()),quantized_rms=float(q.square().mean().sqrt()),
   reconstruction_latent_gradient_norm=float(rh.norm()),commitment_latent_gradient_norm=float(qh.norm()),codebook_gradient_norm=float(qe.norm()),
   latent_gradient_ratio=float(rh.norm()/qh.norm()))
 rows.append(row);print(json.dumps(row),flush=True)
scale={}
for role in ['speaker','listener']:
 x=np.asarray(data['train'][role]);std=x.reshape(-1,56).std(0)
 scale[role]=dict(normalized_global_std_min=float(std.min()),normalized_global_std_median=float(np.median(std)),normalized_global_std_max=float(std.max()),
  official_std_min=float(stats[role+'_std'].min()),official_std_max=float(stats[role+'_std'].max()))
save(W/'QUANTIZATION_DIAGNOSIS.json',dict(checkpoints=rows,scale=scale,optimizer_updates=0,test_media_opened=False,stopped_at=27749))
