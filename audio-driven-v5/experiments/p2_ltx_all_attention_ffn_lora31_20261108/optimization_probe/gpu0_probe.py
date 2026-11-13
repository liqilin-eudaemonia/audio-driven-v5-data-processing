from pathlib import Path
import sys,os,time,json
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
import torch
from p2_listen.contracts import read,atomic_json
OUT=ROOT/'experiments/p2_ltx_all_attention_ffn_lora31_20261108';P=OUT/'optimization_probe'
try:
 torch.set_num_threads(2);torch.cuda.set_device(0)
 torch.cuda.set_per_process_memory_fraction(108*2**30/torch.cuda.get_device_properties(0).total_memory)
 from p2_ltx_all_lora.config import load
 from p2_ltx_all_lora.data import Dataset,data_config
 from p2_ltx_all_lora.model import Model
 from p2_ltx_all_lora.training import longest_check
 from p2_listen.config import extractor_config
 from p2_listen.numerics import configure_training_numerics
 from p2_listen.performance import configure_renderer
 from p2_ltx_events.heads import names
 c=load(OUT/'config.json');configure_training_numerics();cfg=extractor_config(data_config(c));dataset=Dataset(c)
 from omni_av_p2.ltx import load_renderer
 renderer,strategy,sampler=load_renderer(cfg,'cuda:0');configure_renderer(renderer,'all')
 torch.manual_seed(c['training']['seed']);model=Model(c,cfg,renderer).to('cuda:0');model.train()
 optimizer=torch.optim.AdamW(model.groups(),weight_decay=0.)
 stats=read(OUT/'preflight.json')['statistics'];classes=names(c);effective=[k for k in classes if stats[k]['enabled']];pos={k:stats[k]['pos_weight']for k in classes}
 torch.cuda.synchronize();begin=time.monotonic()
 report=longest_check(model,dataset,c,optimizer,c['execution'],strategy,sampler,pos,effective,classes,'cuda:0',0,0,1)
 torch.cuda.synchronize();report.update(total_seconds=time.monotonic()-begin,isolated_probe=True,includes_snapshot_and_restore=True,device='node11:0',memory_cap_gib=108,weights='fresh_pretrained_probe_only',production_modified=False)
 atomic_json(P/'GPU0_ACCEPTED.json',report);print(json.dumps(report),flush=True)
except Exception:
 import traceback
 atomic_json(P/'ERROR.json',dict(error=traceback.format_exc(),at=time.time()));raise
