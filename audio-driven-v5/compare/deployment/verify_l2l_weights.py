import sys,json,torch,gc,hashlib
from pathlib import Path
base=Path(__file__).resolve().parents[1];src=base/'learning2listen/src';sys.path.insert(0,str(src))
from modules.fact_model import setup_model
from vqgan.vqmodules.gan_models import setup_vq_transformer
from argparse import Namespace
records=[]
for person in ['conan','fallon','stephen','trevor']:
 prefix='' if person=='conan' else person+'_'
 cfg=json.loads(Path(f'configs/vq/{prefix}delta_v6.json').read_text());vq=json.loads(Path(cfg['l_vqconfig']).read_text())
 vp=Path('vqgan')/vq['model_path']/f"{vq['tag']}{vq['pipeline']}_best.pth"
 pp=Path('models')/f'{prefix}delta_v6_er2er_best.pth'
 a,*_=setup_vq_transformer(Namespace(),vq,load_path=str(vp),test=True)
 b,*_=setup_model(cfg,vq,mask_index=0,test=True,s_vqconfig=None,load_path=str(pp))
 records.append({'person':person,'vq':str(vp),'predictor':str(pp),'strict_load':True,'gpu':torch.cuda.get_device_name()})
 del a,b;gc.collect();torch.cuda.empty_cache()
(base/'deployment/logs/l2l_weights_verified.json').write_text(json.dumps(records,indent=2))
print('ALL FOUR PAIRED CHECKPOINTS LOADED',flush=True)
