"""Search-only strict inference/export; reuse audited unified conditioning."""
from pathlib import Path
import copy,time,json,sys,os,fcntl,socket
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(WORK)]
from p2_listen.contracts import read,sha,atomic_json as save
PARENT=ROOT/'experiments/p2_ltx6000_latest_sft_dual200_20261107'
CLI=WORK/'evaluate28.py'
ARM='set_per_worker'
def stamp(path):
 s=Path(path).stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def total(work):return read(Path(work)/'protocol.json')['total_samples']
def code_identity():
 files=[WORK/'eval_runtime.py',WORK/'binder.py',CLI,WORK/'run_events.py',WORK/'reference_events.py',WORK/'reference_media_cache.py',ROOT/'src/p2_testsets_eval/runtime.py',ROOT/'src/p2_testsets_eval/shared_items.py',ROOT/'src/p2_testsets_eval/cluster.py',ROOT/'src/p2_progressive_eval_20261108.py',ROOT/'scripts/p2_event_presence.py',*sorted((ROOT/'src/p2_event_presence').glob('*.py'))]
 files.append(WORK/'gpu7_score.py')
 files.append(ROOT/'scripts/p2_all_lora_followups_gpu7_20261109.py')
 return {str(p):sha(p)for p in files}
def export(checkpoint,work):
 from p2_whisper_av.bootstrap import activate_ltx
 activate_ltx()
 import torch
 from p2_loss_search28.checkpoint import checked
 from p2_loss_search28.config import validate,source_identity
 from p2_listen.cache import atomic_torch
 checkpoint=Path(checkpoint);marker=read(checkpoint.with_suffix('.complete.json'));receipt=work/'export/RECEIPT.json'
 if receipt.exists():
  rec=read(receipt)
  if rec['original_sha256']!=marker['sha256']or sha(rec['checkpoint'])!=rec['checkpoint_sha256']:raise ValueError('Export changed')
  return rec
 value=checked(checkpoint);c=validate(value['identity']['experiment'])
 if value['identity']['source']!=source_identity()or len(value['rank_rng'])!=28:raise ValueError('Search training source/world mismatch')
 if value['identity']['ltx_lora_report']['projections']!=1632:raise ValueError('Not full LTX LoRA')
 if not all(torch.isfinite(t).all()for state in value['optimizer']['state'].values()for t in state.values()if torch.is_tensor(t)):raise ValueError('Nonfinite Adam state')
 target=work/'export/model.pt';atomic_torch(target,{k:value[k]for k in ('schema','step','identity','model')})
 loaded=torch.load(target,map_location='cpu',weights_only=True,mmap=True)
 if not all(torch.equal(v,loaded['model'][k])for k,v in value['model'].items()):raise ValueError('Model export changed tensors')
 rec=dict(original=str(checkpoint),original_sha256=marker['sha256'],checkpoint=str(target),checkpoint_sha256=sha(target),stamp=stamp(target),step=value['step'],identity=value['identity'],success=True)
 save(receipt,rec);return rec

def prepare(work):
 work=Path(work);request=read(work/'REQUEST.json');rec=export(request['checkpoint'],work)
 if(work/'protocol.json').exists():
  p=read(work/'protocol.json')
  if p['execution_code']!=code_identity()or p['arms'][0]['original_sha256']!=rec['original_sha256']:raise ValueError('Eval contract changed')
  return p
 from p2_combined_whisper_eval import link
 from p2_testsets_eval.protocol import partition
 from p2_loss_search28.data import data_config
 membership=read(WORK/'VALIDATION_MEMBERSHIP_ACCEPTED.json')
 manifest=WORK/('tuning64.json'if request['samples']==64 else'tuning200.json')
 if sha(manifest)!=membership['outputs'][manifest.name]:raise ValueError('Validation membership changed')
 chosen=read(manifest);keys={'seamless/'+x['id']for x in chosen}
 parent=read(PARENT/'protocol.json');items=[x for x in read(PARENT/'shared_items.json')if x['key']in keys]
 if len(items)!=request['samples']:raise ValueError('Incomplete frozen inference conditioning')
 arm='losssearch_'+rec['identity']['experiment']['loss_search']['candidate']+'_'+str(rec['step'])
 for folder in ('conditions/seamless','whisper/seamless','negative/00'):link(PARENT/folder,work/folder)
 cluster=copy.deepcopy(read(Path(request['checkpoint']).parent/'cluster.json'))
 save(work/'cluster.json',cluster);save(work/'DATA_CONFIG.json',data_config(rec['identity']['experiment']));save(work/'shared_items.json',items)
 for rank,rows in enumerate(partition(items,28)):save(work/'shards'/f'{rank:02d}.json',rows)
 p=copy.deepcopy(parent);p.update(schema='losssearch28-unified-evaluation-v1',arms=[dict(name=arm,kind='ltx_all_lora_full',step=rec['step'],checkpoint=rec['checkpoint'],checkpoint_sha256=rec['checkpoint_sha256'],original_checkpoint=rec['original'],original_sha256=rec['original_sha256'],planner=parent['planner'])],checkpoint_identity=rec['identity'],datasets={'seamless':dict(parent['datasets']['seamless'],prepared=str(WORK/'prepared64')if len(items)==64 else parent['datasets']['seamless']['prepared'],references_sha256=sha(WORK/'prepared64/references.jsonl')if len(items)==64 else parent['datasets']['seamless']['references_sha256'],count=len(items))},workers=28,total_samples=len(items),total_outputs=len(items),cluster=str(work/'cluster.json'),metrics_cluster=str(work/'cluster.json'),worker_cli=str(CLI),shared_items_manifest=str(work/'shared_items.json'),shared_items_sha256=sha(work/'shared_items.json'),execution_code=code_identity(),authorized_scope='node12/node8/node4 GPU0-7, node11 GPU0-3 only; no shared training',dataset_role='hyperparameter validation, not final independent test')
 save(work/'protocol.json',p);save(work/'CPU_ACCEPTANCE.json',dict(success=True,samples=len(items),source_only_conditioning=True,at=time.time()));return p

def load_model(path,work,device='cuda:0'):
 import torch
 from p2_loss_search28.config import validate,source_identity
 from p2_loss_search28.model import Model
 from p2_loss_search28.data import data_config
 from p2_listen.config import extractor_config
 from p2_listen.features import weights_identity
 from omni_av_p2.ltx import load_renderer
 rec=read(work/'export/RECEIPT.json')
 if stamp(path)!=rec['stamp']:raise ValueError('Export file changed before load')
 saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
 if stamp(path)!=rec['stamp']or saved['step']!=rec['step']:raise ValueError('Export changed during load')
 identity=saved['identity'];c=validate(identity['experiment'])
 if identity!=rec['identity']or identity['source']!=source_identity():raise ValueError('Inference source identity changed')
 d=data_config(c)
 if identity['assets']!={k:weights_identity(d,k)for k in ('omni','ltx','gemma')}:raise ValueError('Pretrained assets changed')
 cfg=extractor_config(d);renderer,strategy,sampler=load_renderer(cfg,device);torch.manual_seed(c['training']['seed']);model=Model(c,cfg,renderer).to(device)
 if model.text_targets!=identity['text_targets']:raise ValueError('LoRA target mismatch')
 model.load_state_dict(saved['model'],strict=True);model.eval().requires_grad_(False);renderer.set_gradient_checkpointing(False)
 return model,strategy,sampler

def render(work,items,rank,p):
    import torch
    from p2_testsets_eval import runtime as rt
    from p2_testsets_eval.whisper_runtime import matched_sample
    from p2_testsets_eval.shared_items import claim_items,all_items
    from p2_ltx_events.inference import load_model
    from p2_listen.cache import atomic_torch,cpu,tree_to
    from p2_listen.features import unload,make_observation
    from p2_whisper_av.features import SpeakerFeatures
    from omni_av_p2.connector import ConditionInputs
    from omni_av_p2.inference import reference_latents
    from omni_av_p2.ltx import unpack_latents
    from omni_av.ltx_runtime import with_text
    items=all_items(work,items,p);arm=p['arms'][0];device='cuda:0';_,cfg=rt.data_config()
    if all(rt.ready(work/'latents'/ARM/x['key']) or rt.ready(work/'generated'/ARM/x['key']) for x in items):return
    model,strategy,sampler=load_model(arm['checkpoint'],device)
    model.eval().requires_grad_(False);model.renderer.eval().requires_grad_(False);model.renderer.set_gradient_checkpointing(False)
    negative=tree_to(torch.load(work/'negative'/f'{rank:02d}/payload.pt',map_location='cpu',weights_only=True),device)
    with rt.cached_loaders(['load_video_vae_encoder']),torch.inference_mode():
        for i,item in enumerate(claim_items(work,items,p,arm,'render')):
            begin=time.monotonic();folder=work/'conditions'/item['key'];out=work/'latents'/ARM/item['key'];g=item['geometry']
            assert rt.ready(folder/'qwen') and rt.ready(folder/'text')
            meta=read(work/'whisper'/item['key']/'metadata.json');plan=read(folder/'qwen/plan.json')
            assert meta['encoder']==p['whisper_encoder'] and meta['feature_code']==p['feature_code']
            assert meta['source']['input_audit_sha256']==sha(folder/'input_audit.json')
            assert plan['adapter']==p['planner_adapter_identity'] and plan['target_description_access'] is False
            payload=torch.load(folder/'qwen/hidden.pt',map_location='cpu',weights_only=True);text=torch.load(folder/'text/payload.pt',map_location='cpu',weights_only=True)
            assert text['prompt']==payload['prompt']==plan['prompt']
            inputs=ConditionInputs(**tree_to(payload['condition'],device))
            with torch.autocast('cuda',dtype=torch.bfloat16):u=model.connector(inputs)
            speaker=SpeakerFeatures(**torch.load(work/'whisper'/item['key']/'payload.pt',map_location='cpu',weights_only=True)).validate().to(device)
            native=dict(fps=g['fps'],height=g['height'],width=g['width'],frames=g['native_frames'],duration_s=g['native_duration_s'])
            av=reference_latents(cfg,make_observation(rt.observation(item,folder)),native,device)
            torch.manual_seed(p['seed']);prepared=strategy.prepare_training_inputs(with_text(av,tree_to(text,device)),sampler)
            torch.manual_seed(p['seed']);sampled=matched_sample(model,prepared,u,speaker,negative,**p['generation']);video,audio=unpack_latents(sampled,av)
            assert torch.isfinite(video).all() and torch.isfinite(audio).all()
            atomic_torch(out/'payload.pt',cpu(dict(video=video,audio=audio)))
            save(out/'metadata.json',dict(key=item['key'],arm=ARM,checkpoint=arm['original_checkpoint'],checkpoint_sha256=arm['original_sha256'],inference_checkpoint=arm['checkpoint'],step=arm['step'],seed=p['seed'],geometry=g,planning_sha256=sha(folder/'qwen/plan.json'),planning_adapter=p['planner_adapter_identity'],target_description_access=False,listener_future_input=False,generation=p['generation'],whisper_source=meta,sampling=p['sampling'],seconds=time.monotonic()-begin))
            rt.complete(out,[out/'payload.pt',out/'metadata.json']);rt.progress(work,'render',rank,done=i+1,total=len(items),arm=ARM,last_seconds=time.monotonic()-begin)
            del payload,text,inputs,u,speaker,av,prepared,sampled,video,audio
    del model,strategy,sampler,negative;unload()


def worker(work,stage,rank,slot=0):
 global ARM
 p=read(work/'protocol.json');ARM=p['arms'][0]['name'];cluster=read(work/'cluster.json')
 if not 0<=rank<28 or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(rank%8):raise ValueError('Outside reserved allocation')
 node=cluster['nodes'][rank//8]
 if node['host']not in os.popen('hostname -I').read().split():raise ValueError('Wrong host')
 if p['execution_code']!=code_identity():raise ValueError('Eval code changed')
 from p2_whisper_av.bootstrap import activate_ltx
 activate_ltx()
 import torch
 torch.set_num_threads(4);torch.cuda.set_device(0)
 from p2_listen.config import extractor_config
 from p2_testsets_eval import runtime as rt
 from p2_testsets_eval.shared_items import all_items
 import p2_ltx_events.inference as previous
 d=read(work/'DATA_CONFIG.json');cfg=extractor_config(d)
 folder=work/'locks';folder.mkdir(exist_ok=True)
 with(folder/f'{stage}_{rank}_{slot}.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  save(work/'devices'/f'{stage}_{rank}_{slot}.json',dict(pid=os.getpid(),host=socket.gethostname(),physical_gpu=rank%8,rank=rank,at=time.time()))
  old_progress=rt.progress
  def progress(w,s,r,**kw):return old_progress(w,s,rank+28*slot,**kw)
  def loading(path,device='cuda:0'):return load_model(path,work,device)
  with patch.object(rt,'data_config',lambda:(d,cfg)),patch.object(rt,'progress',progress),patch.object(previous,'load_model',loading):
   if stage=='render':render(work,all_items(work,[],p),0,p)
   else:rt.decode(work,all_items(work,[],p),rank,p,p['arms'])
  progress(work,stage,0,status='complete',slot=slot)

def final(work):
 from p2_testsets_eval.runtime import ready
 p=read(work/'protocol.json');arm=p['arms'][0];items=read(work/'shared_items.json')
 for x in items:
  folder=work/'generated'/arm['name']/x['key']
  if not ready(folder):raise ValueError('Missing generated media')
  m=read(folder/'metadata.json')
  if m['step']!=arm['step']or m['checkpoint_sha256']!=arm['original_sha256']or m['target_description_access']or m['listener_future_input']:raise ValueError('Wrong generation provenance')
 result=read(work/'RESULTS.json')[arm['name']+'/seamless']
 if any(result['coverage'][k]['pairs']!=len(items)for k in ('psnr','ssim','lpips','fid','fvd')):raise ValueError('Missing video metric coverage')
 # N/A sound-quality windows are preserved as N/A, never converted to zero.
 save(work/'FINAL_AUDIT.json',dict(success=True,outputs=len(items),step=arm['step'],source_only=True,role='tuning_validation',audio_errors=result.get('audio',{}).get('pairs_with_errors'),event_presence_scoring='separate_required_before_promotion',at=time.time()))
