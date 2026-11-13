"""Preserve native renderer and scoring; strict stage1500 model provenance."""
from pathlib import Path
import sys,copy,importlib
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
TRAIN=ROOT/'experiments/p2_event_feature_interaction28_20261111';LEGACY=ROOT/'experiments/p2_loss_search28_evaluation_20261110'
sys.path[:0]=[str(TRAIN),str(ROOT/'src'),str(ROOT/'scripts'),str(LEGACY)]
import extension
from p2_listen.contracts import read,sha,atomic_json as save
import eval_runtime as E
_old_code=E.code_identity;_old_export=E.export;_old_load=E.load_model

def select(path):
 p=Path(path)
 if p.parent==TRAIN/'runs/event020_feature010':extension.activate();return 'candidate28'
 if p.parent==extension.ORIGINAL/'runs/baseline':
  import p2_loss_search28.config as config
  importlib.reload(config);return 'baseline28'
 if p.parent==ROOT/'experiments/p2_ltx_all_attention_ffn_lora31_20261108':return 'historical31'
 raise ValueError('Unapproved checkpoint path')

def export(cp,work):
 kind=select(cp)
 if kind!='historical31':return _old_export(cp,work)
 from p2_whisper_av.bootstrap import activate_ltx
 activate_ltx()
 import torch
 torch.set_num_threads(2)
 from p2_ltx_all_lora.checkpoint import checked
 from p2_ltx_all_lora.config import validate,source_identity
 from p2_listen.cache import atomic_torch
 cp=Path(cp);marker=read(cp.with_suffix('.complete.json'));f=work/'export/RECEIPT.json'
 if f.exists():
  r=read(f);assert r['original_sha256']==marker['sha256'] and E.stamp(r['checkpoint'])==r['stamp'];return r
 v=checked(cp);assert v['step'] in (1500,6000) and cp.name==f"step_{v['step']:06d}.pt" and len(v['rank_rng'])==31
 assert v['identity']==read(cp.parent/'run.json') and v['identity']['source']==source_identity()
 c=validate(v['identity']['experiment']);assert c['loss_weights']==dict(av=1.,event=.05,video_feature=.025,audio_feature=.025)
 assert v['identity']['ltx_lora_report']['projections']==1632
 assert all(torch.isfinite(t).all() for s in v['optimizer']['state'].values() for t in s.values() if torch.is_tensor(t))
 target=work/'export/model.pt';atomic_torch(target,{k:v[k] for k in ['schema','step','identity','model']})
 loaded=torch.load(target,map_location='cpu',weights_only=True,mmap=True)
 assert set(v['model'])==set(loaded['model']) and all(torch.equal(t,loaded['model'][k]) for k,t in v['model'].items())
 r=dict(success=True,original=str(cp),original_sha256=marker['sha256'],checkpoint=str(target),checkpoint_sha256=sha(target),stamp=E.stamp(target),step=v['step'],identity=v['identity'],schema=v['schema'],model_bitwise_equal=True)
 save(f,r);return r

def load_model(path,work,device='cuda:0'):
 r=read(work/'export/RECEIPT.json');kind=select(r['original'])
 if kind!='historical31':return _old_load(path,work,device)
 import torch
 from unittest.mock import patch
 from p2_ltx_all_lora import inference,checkpoint
 assert E.stamp(path)==r['stamp']
 v=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
 assert E.stamp(path)==r['stamp'] and v['identity']==r['identity'] and v['step']==r['step'] and v['step'] in (1500,6000) and v['schema']==r['schema']
 with patch.object(checkpoint,'checked',lambda _:v):return inference.load_model(path,device)

def code_identity():return {**_old_code(),**{str(WORK/n):sha(WORK/n) for n in ['adapter.py','evaluate28.py']}}
def prepare(work):
 from p2_combined_whisper_eval import link
 cp=Path(read(work/'REQUEST.json')['checkpoint']);r=export(cp,work);expected_step=1661 if select(cp)=='candidate28' else int(cp.stem.split('_')[-1]);assert r['step']==expected_step
 if (work/'protocol.json').exists():
  p=read(work/'protocol.json');assert p['execution_code']==code_identity() and p['arms'][0]['original_sha256']==r['original_sha256'];return p
 template=TRAIN/'evaluations/event020_feature010_500_seed137';p=copy.deepcopy(read(template/'protocol.json'))
 for d in ('conditions','whisper','negative'):link(template/d,work/d)
 save(work/'shared_items.json',read(template/'shared_items.json'),immutable=True)
 kind=select(cp)
 if kind=='historical31':from p2_ltx_all_lora.data import data_config
 else:from p2_loss_search28.data import data_config
 save(work/'DATA_CONFIG.json',data_config(r['identity']['experiment']))
 name='baseline' if kind!='candidate28' else 'event020_feature010'
 arm=dict(name='losssearch_'+name+'_'+str(expected_step),kind='ltx_all_lora_full',step=expected_step,checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],original_checkpoint=str(cp),original_sha256=r['original_sha256'],planner=p['planner'])
 if expected_step==6000:arm['name']='ltx_all_lora_event_feature_whisper_6000_sft2752'
 p.update(arms=[arm],checkpoint_identity=r['identity'],execution_code=code_identity(),workers=28,seed=137,cluster=str(work/'cluster.json'),metrics_cluster=str(work/'cluster.json'),shared_items_manifest=str(work/'shared_items.json'),worker_cli=str(WORK/'evaluate28.py'))
 p.pop('scoring_device',None);save(work/'protocol.json',p);return p

def install():E.code_identity=code_identity;E.prepare=prepare;E.load_model=load_model
