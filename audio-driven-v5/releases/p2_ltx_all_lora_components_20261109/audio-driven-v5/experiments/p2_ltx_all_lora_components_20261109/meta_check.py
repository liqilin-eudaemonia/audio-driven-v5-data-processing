from pathlib import Path
import sys,json,gc
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
import torch
from safetensors import safe_open
from p2_ltx_all_lora_components.config import load
from p2_ltx_all_lora_components.data import data_config
from p2_ltx_all_lora_components.model import Model
from p2_listen.config import extractor_config
from p2_listen.contracts import atomic_json
from ltx_core.model.transformer.model_configurator import LTXModelConfigurator
for arm in ['no_feature','no_whisper']:
 c=load(Path(__file__).parent/arm/'config.json');cfg=extractor_config(data_config(c))
 with safe_open(cfg.ltx_path,framework='pt',device='cpu')as f:config=json.loads(f.metadata()['config'])
 with torch.device('meta'):
  renderer=LTXModelConfigurator.from_config(config).requires_grad_(False)
  model=Model(c,cfg,renderer)
 r=model.ltx_lora_report;groups={g['name']for g in model.groups()}
 assert r['blocks']==48 and r['projections']==1632 and r['parameters']==84221952
 assert groups=={'connector','ltx_lora','events','whisper' if arm=='no_feature'else'features'}
 assert bool(model.decoders)==(arm=='no_whisper')
 assert r['whisper_full_parameters']==(2779450368 if arm=='no_feature'else 0)
 atomic_json(Path(__file__).parent/arm/'META_COVERAGE.json',dict(r,success=True,meta_device_only=True,optimizer_groups=sorted(groups),feature_parameters=sum(p.numel()for p in model.decoders.parameters()),event_parameters=sum(p.numel()for p in model.event_heads.parameters())))
 print(arm,r,flush=True);del model,renderer;gc.collect()
assert not torch.cuda.is_initialized()
