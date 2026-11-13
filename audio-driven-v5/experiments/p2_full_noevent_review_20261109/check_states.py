from pathlib import Path
import sys,json,math,time,importlib
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
import torch
from p2_listen.contracts import read,sha,atomic_json
W=Path(__file__).parent;torch.set_num_threads(8);result={};configs={}
for arm,out,step,package in [('full','p2_ltx_all_lora_full28_batch28_20261109',6500,'p2_ltx_all_lora'),('noevent','p2_ltx_all_lora_noevent24_after6000_20261109',660,'p2_ltx_all_lora_noevent')]:
 p=ROOT/'experiments'/out;run=read(p/'run.json');c=read(p/'config.json');assert c==run['experiment'];configs[arm]=c
 changed=[s for s,h in run['source'].items()if sha(s)!=h];assert not changed,changed
 cp=importlib.import_module(package+'.checkpoint').checked(p/f'step_{step:06d}.pt');assert cp['identity']==run and cp['step']==step and len(cp['rank_rng'])==run['world']
 groups={g['name']:g for g in cp['optimizer']['param_groups']};expected={'connector','ltx_lora','whisper','features'}|({'events'}if arm=='full'else set());assert set(groups)==expected
 events=[k for k in cp['model']if k.startswith('event_heads.')];assert bool(events)==(arm=='full')
 assert run['ltx_lora_report']['projections']==1632 and run['ltx_lora_report']['rank']==8 and run['ltx_lora_report']['whisper_full_parameters']==2779450368
 rows=[json.loads(x)for x in (p/'metrics.jsonl').read_text().splitlines()];checked=rows[-50:]if arm=='full'else rows
 for row in checked:
  target=row['av']+(.05*row['event']if arm=='full'else 0)+.025*(row['feature_video']+row['feature_audio'])
  assert math.isclose(row['weighted_total'],target,abs_tol=1e-10)
  assert all(math.isfinite(v)for v in row['grad_norm'].values())and math.isfinite(target)
  if arm=='noevent':assert row['event']==row['event_clip']==row['event_temporal']==0 and not row['per_event']
 if arm=='noevent':assert(p/'STOP').exists()and read(p/'status.json')['step']==660
 result[arm]=dict(step_checked=step,checksum=read(p/f'step_{step:06d}.complete.json')['sha256'],source_files_unchanged=len(run['source']),groups=sorted(groups),event_tensor_count=len(events),rng_count=len(cp['rank_rng']),metrics_checked=len(checked),latest_step=rows[-1]['step'],model_and_adam_finite=True,strict_checkpoint_identity=True,report=run['ltx_lora_report'])
 del cp
 print(arm,'checked',flush=True)
a,b=configs['full'],configs['noevent']
for k in ['current_cache_index','sample_list','j_cache','plan_cache','teacher_cache','qwen_hidden_source','text_lora','qwen_lora','connector','whisper','whisper_cache','duration_sampling','execution']:
 assert a[k]==b[k],k
for k,v in a['training'].items():
 if k not in ['global_batch','steps']:assert v==b['training'][k],k
atomic_json(W/'STATE_ACCEPTED.json',dict(success=True,at=time.time(),arms=result,shared_data_caption_model_settings_equal=True,execution_difference='full batch28 after6428 vs noEvent batch31; noEvent paused660; not equal training budget'))
