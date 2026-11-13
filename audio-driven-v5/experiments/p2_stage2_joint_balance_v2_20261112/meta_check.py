from pathlib import Path
import sys,json,torch
W=Path(__file__).resolve().parent;ROOT=W.parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_stage2_joint_v2.config import validate
from p2_stage2_joint_v2.data import data_config
from p2_stage2_joint_v2.generator import Model
from p2_listen.config import extractor_config
from p2_listen.contracts import read,atomic_json
from safetensors import safe_open
from ltx_core.model.transformer.model_configurator import LTXModelConfigurator
c=validate(read(W/'candidate_full6000/config.json'));cfg=extractor_config(data_config(c['generator']))
with safe_open(cfg.ltx_path,framework='pt',device='cpu')as f:architecture=json.loads(f.metadata()['config'])
with torch.device('meta'):
 renderer=LTXModelConfigurator.from_config(architecture).requires_grad_(False)
 model=Model(c['generator'],cfg,renderer)
r=model.ltx_lora_report;assert r['blocks']==48 and r['projections']==1632 and r['parameters']==84221952
assert r['whisper_full_parameters']==2779450368
assert {g['name']for g in model.groups()}=={'connector','ltx_lora','whisper','features','events'}
parent=torch.load(c['assets']['stage1_checkpoint']['path'],map_location='cpu',weights_only=True,mmap=True)
expected=model.state_dict();assert set(expected)==set(parent['model'])
for name,t in expected.items():
 assert t.shape==parent['model'][name].shape and t.dtype==parent['model'][name].dtype,name
assert parent['identity']['experiment']==read(c['parent_config'])
assert not torch.cuda.is_initialized()
atomic_json(W/'META_ACCEPTED.json',dict(success=True,meta_only=True,parent_keys=len(expected),model_shapes_and_dtypes_match_parent=True,scope=r,gpu_training_started=False))
print('Full22B meta coverage and selected6000 checkpoint shapes/dtypes match',flush=True)
