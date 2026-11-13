"""Persistent GPU stages; inference reads source audio and exactly one image."""
import contextlib,fcntl,io,math,os,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from .protocol import ROOT,OMNI,read,save,sha


def progress(work,stage,rank,**values):
 save(work/'progress'/stage/f'{rank:02d}.json',dict(stage=stage,rank=rank,pid=os.getpid(),at=time.time(),**values))

def complete(folder,files,**metadata):
 save(folder/'complete.json',dict(**metadata,files={str(Path(p).relative_to(folder)):dict(bytes=Path(p).stat().st_size,sha256=sha(p)) for p in files},completed_at=time.time()))

def ready(folder):
 marker=folder/'complete.json'
 if not marker.exists():return False
 info=read(marker)
 for rel,s in info['files'].items():
  p=folder/rel
  if not p.is_file() or p.stat().st_size!=s['bytes']:raise ValueError('Incomplete committed artifact: '+str(p))
 return True

def data_config():
 from p2_listen.config import extractor_config
 d=read(ROOT/'experiments/p2_j_train500h_cache_20261101/manifest.json')['config']
 return d,extractor_config(d)

def checkpoint_source(c,kind):
 if c.get('qwen_hidden_source')=='planning_sft_frozen':
  from p2_j_sft_event_backend import source_identity
  expected=source_identity()
  # The base-H installer can run earlier in the same inference process. Its
  # extra CLI is not used/bound by the independent SFT-H training entry.
  expected.pop(str(ROOT/'scripts/p2_j_duration_train.py'),None)
  return expected
 from p2_j_duration_execution import install_backend
 expected=install_backend().source_identity()
 if kind=='av_only':
  path=ROOT/'scripts/p2_j_av_only.py';expected=dict(expected,**{str(path):sha(path)})
 return expected

def planner_identity(p):
 if not p.get('planner_adapter'):return None
 from p2_j_sft_hidden import adapter_identity
 actual=adapter_identity(p['planner_adapter'])
 if actual!=p['planner_adapter_identity']:raise ValueError('Planning LoRA changed after evaluation preparation')
 return actual

def verify_checkpoint_source(identity,c,arm):
 expected=checkpoint_source(c,arm['kind'])
 if identity['source']==expected:return
 compatibility=arm.get('source_compatibility')
 if compatibility:
  report=read(compatibility)
  if (report['checkpoint_source']==identity['source'] and report['current_source']==expected
      and report['checkpoint_sha256']==read(Path(arm['checkpoint']).with_suffix('.complete.json'))['sha256']
      and report['inference_math_unchanged'] is True):return
 raise ValueError('Checkpoint implementation differs without a verified compatibility record')

def conditioning(item,folder):
 """The only inference-facing dataset adapter. Never decode full GT or listener audio."""
 import soundfile as sf
 from scipy.signal import resample_poly
 r=item['reference'];g=item['geometry'];folder.mkdir(parents=True,exist_ok=True)
 if item['dataset']=='seamless':
  from evaluation.seamless.reader import load_conditioning
  inputs,audit=load_conditioning(r['gt_reference'])
  wave=np.asarray(inputs['speaker_audio'],dtype=np.float32);rate=inputs['speaker_sample_rate']
  image=inputs['listener_first_frame'].convert('RGB')
 else:
  import av
  wave,rate=sf.read(r['speaker']['audio'],dtype='float32',always_2d=True)
  expected=r['media']['speaker']['audio']
  if len(wave)!=expected['samples'] or rate!=expected['sample_rate']:raise ValueError('Speaker WAV header differs from sealed reference')
  wave=wave.mean(1)
  with av.open(r['listener']['video']) as container:
   stream=container.streams.video[0];stream.thread_type='SLICE';stream.codec_context.thread_count=1
   first=next(container.decode(video=0));image=first.to_image().convert('RGB')
  audit=dict(speaker_audio=r['speaker']['audio'],reference_first_frame=r['listener']['video'],
             source_frame_pts=first.pts,source_frame_time=float(first.time or 0),decoded_listener_frames=1)
 if wave.ndim==2:wave=wave.mean(1)
 if image.size!=(g['source_width'],g['source_height']):raise ValueError('Reference frame geometry mismatch')
 if image.size!=(g['width'],g['height']):
  image=image.resize((g['width'],g['height']),Image.Resampling.BICUBIC)
  audit['reference_resize']='PIL bicubic 640->512; same rule as evaluation GT'
 if rate!=16000:
  common=math.gcd(int(rate),16000);wave=resample_poly(wave,16000//common,int(rate)//common).astype(np.float32)
 count=round(g['duration_s']*16000);delta=count-len(wave)
 tolerance=max(math.ceil(16000/rate),16 if item['dataset']=='responsenet' else 1)
 if abs(delta)>tolerance:raise ValueError(f"Speaker endpoint mismatch {item['key']}: {delta} samples, limit {tolerance}")
 wave=np.pad(wave,(0,max(0,delta)))[:count]
 if not np.isfinite(wave).all():raise ValueError('Nonfinite source audio')
 image.save(folder/'first_frame.png')
 audit.update(speaker_resample_rate=16000,endpoint_adjustment_samples=delta,
              source_only=True,target_description_in_model_input=False,listener_future_media_in_model_input=False)
 save(folder/'input_audit.json',audit)
 return dict(sample_id=item['id'],duration=g['duration_s'],image=image,audio=wave)

def observation(item,folder,wave=None):
 import soundfile as sf
 g=item['geometry'];audio=None
 if wave is not None:
  audio=io.BytesIO();sf.write(audio,wave,16000,format='WAV',subtype='FLOAT');audio.seek(0)
 return dict(sample_id=item['id'],speaker_audio=audio,listener_image=str(folder/'first_frame.png'),
             requested_duration_s=g['duration_s'],geometry={k:g[k] for k in ['fps','width','height']})

def cache(work,items,rank,p):
 import torch
 from p2_listen.cache import atomic_torch,cpu
 from p2_listen.features import unload
 from p2_planning_sft.data import PlanningCollator,SYSTEM,USER
 from p2_j_cache.data import caption_prompt
 from p2_plan_cache.encoding import extract_planned_input,encode_text
 from p3_listen.prompts import FixedPrompt
 d,cfg=data_config();device='cuda:0'
 qtodo=[x for x in items if not ready(work/'conditions'/x['key']/'qwen')]
 if qtodo:
  from omni_av_p2.data import load_omni
  adapter=planner_identity(p)
  if adapter:
   from p2_j_sft_hidden import Reader,load_session
   reader=Reader(p['sft_hidden_cache'])
   if reader.protocol['adapter']!=adapter:raise ValueError('Inference LoRA differs from cached training H')
   runtime,processor,peft=load_session(reader)
  else:runtime,processor=load_omni(cfg,device)
  collator=PlanningCollator(cfg.omni_path);collator.processor=processor
  processor.tokenizer.padding_side='left';eos=processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
  for start in range(0,len(qtodo),p['planner_batch']):
   batch=qtodo[start:start+p['planner_batch']]
   with ThreadPoolExecutor(max_workers=4) as pool:
    raw=list(pool.map(lambda x:conditioning(x,work/'conditions'/x['key']),batch))
   inputs=collator.source_inputs(raw).to(device,dtype=torch.bfloat16)
   runtime.thinker.rope_deltas=None
   with torch.inference_mode():
    generated=runtime.thinker.generate(**inputs,max_new_tokens=p['planner_max_new_tokens'],do_sample=False,use_cache=True,
        use_audio_in_video=False,eos_token_id=eos,pad_token_id=processor.tokenizer.pad_token_id)
   generated=generated[:,inputs.input_ids.shape[1]:].tolist();del inputs
   for item,row,ids in zip(batch,raw,generated):
    folder=work/'conditions'/item['key'];q=folder/'qwen';q.mkdir(parents=True,exist_ok=True)
    attempts=[dict(token_ids=ids,max_new_tokens=p['planner_max_new_tokens'])]
    if eos not in ids:
     single=collator.source_inputs([row]).to(device,dtype=torch.bfloat16);runtime.thinker.rope_deltas=None
     with torch.inference_mode():
      gen=runtime.thinker.generate(**single,max_new_tokens=p['planner_retry_tokens'],do_sample=False,use_cache=True,
        use_audio_in_video=False,eos_token_id=eos,pad_token_id=processor.tokenizer.pad_token_id)
     ids=gen[0,single.input_ids.shape[1]:].tolist();del single,gen
     attempts.append(dict(token_ids=ids,max_new_tokens=p['planner_retry_tokens']))
    if eos not in ids:
     save(q/'failed.json',dict(reason='Planner missing EOS',attempts=attempts));raise ValueError('Planner truncated: '+item['key'])
    ids=ids[:ids.index(eos)+1];text=processor.tokenizer.decode(ids,skip_special_tokens=True).strip()
    if not text or '<|' in text or '|>' in text:raise ValueError('Invalid generated planning text')
    prompt=caption_prompt(text)
    with torch.inference_mode():payload=extract_planned_input(d,observation(item,folder,row['audio']),prompt,runtime,processor,32768)
    atomic_torch(q/'hidden.pt',payload)
    save(q/'plan.json',dict(description=text,prompt=prompt,source=p['planner'],adapter=adapter,
       system=SYSTEM,user=USER.format(duration=row['duration']),attempts=attempts,target_description_access=False))
    complete(q,[q/'hidden.pt',q/'plan.json'],source_only=True)
    progress(work,'cache',rank,part='qwen',sample=item['key'],done=start+batch.index(item)+1,total=len(qtodo))
   del raw,batch,generated
  del runtime,processor,collator
  if adapter:del peft,reader
  unload()
 todo=[x for x in items if not ready(work/'conditions'/x['key']/'text')]
 negative=work/'negative'/f'{rank:02d}'
 if todo or not ready(negative):
  from omni_av.ltx_runtime import NativeTextConditioner
  encoder=NativeTextConditioner(cfg,device)
  if not ready(negative):
   negative.mkdir(parents=True,exist_ok=True)
   atomic_torch(negative/'payload.pt',cpu(encoder.encode(FixedPrompt(''))));complete(negative,[negative/'payload.pt'])
  for start in range(0,len(todo),p['text_batch']):
   batch=todo[start:start+p['text_batch']]
   prompts=[read(work/'conditions'/x['key']/'qwen/plan.json')['prompt'] for x in batch]
   lengths=[len(encoder.encoder.tokenizer.tokenizer(x.strip(),truncation=False)['input_ids']) for x in prompts]
   with torch.inference_mode():
    values=encoder.encode_batch([FixedPrompt(x) for x in prompts]) if max(lengths)<=1024 else [encode_text(encoder,x,8192)[0] for x in prompts]
   for item,text in zip(batch,values):
    folder=work/'conditions'/item['key']/'text';folder.mkdir(parents=True,exist_ok=True)
    atomic_torch(folder/'payload.pt',cpu(text));complete(folder,[folder/'payload.pt'])
   del values
   progress(work,'cache',rank,part='gemma',done=min(start+p['text_batch'],len(todo)),total=len(todo))
  encoder.close();del encoder;unload()

@contextlib.contextmanager
def cached_loaders(names):
 """Reuse native VAE loaders unchanged, retaining each module for this GPU stage."""
 from ltx_trainer import model_loader
 original={name:getattr(model_loader,name) for name in names};memo={}
 def wrap(name):
  def get(*a,**kw):
   if name not in memo:memo[name]=original[name](*a,**kw)
   return memo[name]
  return get
 for name in names:setattr(model_loader,name,wrap(name))
 try:yield
 finally:
  for name,value in original.items():setattr(model_loader,name,value)
  memo.clear()

def render(work,items,rank,p,arms):
 import torch
 from p2_listen.cache import atomic_torch,cpu,tree_to
 from p2_listen.features import unload,make_observation,weights_identity
 from p2_matrix16.checkpoint import checked
 from p2_matrix16.model import MatrixModel
 from p2_listen.runtime import guided_sample,append_negative
 from omni_av_p2.ltx import load_renderer,append_conditions,unpack_latents
 from omni_av_p2.connector import ConditionInputs
 from omni_av_p2.inference import reference_latents
 from omni_av.ltx_runtime import with_text
 d,cfg=data_config();device='cuda:0'
 renderer,strategy,sampler=load_renderer(cfg,device);renderer.set_gradient_checkpointing(False)
 first=checked(arms[0]['checkpoint']);model=MatrixModel(first['identity']['experiment'],cfg,renderer).to(device)
 model.eval().requires_grad_(False);renderer.eval().requires_grad_(False);del first
 negative=tree_to(torch.load(work/'negative'/f'{rank:02d}'/'payload.pt',map_location='cpu',weights_only=True),device)
 assets={k:weights_identity(d,k) for k in ['omni','ltx','gemma']}
 with cached_loaders(['load_video_vae_encoder']),torch.inference_mode():
  for arm in arms:
   saved=checked(arm['checkpoint']);identity=saved['identity'];c=identity['experiment']
   verify_checkpoint_source(identity,c,arm)
   assert identity['assets']==assets
   assert saved['step']==arm['step'] and c['stage']==1 and c['text_condition']=='guided'
   assert model.text_targets==identity['text_targets'] and model.qwen_targets==identity['qwen_targets']
   model.load_state_dict(saved['model'],strict=True);checkpoint_sha=read(Path(arm['checkpoint']).with_suffix('.complete.json'))['sha256'];del saved
   expected_adapter=planner_identity(p) if p.get('enforce_planner_identity') else None
   for index,item in enumerate(items):
    out=work/'latents'/arm['name']/item['key']
    if ready(out):continue
    start=time.monotonic();folder=work/'conditions'/item['key'];g=item['geometry']
    if not ready(folder/'qwen') or not ready(folder/'text'):raise ValueError('Incomplete condition cache')
    payload=torch.load(folder/'qwen/hidden.pt',map_location='cpu',weights_only=True)
    text=torch.load(folder/'text/payload.pt',map_location='cpu',weights_only=True)
    assert text['prompt']==payload['prompt']==read(folder/'qwen/plan.json')['prompt']
    if p.get('enforce_planner_identity'):
     plan=read(folder/'qwen/plan.json')
     if plan['adapter']!=expected_adapter:raise ValueError('Condition cache belongs to another planning model')
     if c.get('qwen_hidden_source')=='planning_sft_frozen' and expected_adapter is None:
      raise ValueError('SFT-H checkpoint requires its matching planning LoRA')
    inputs=ConditionInputs(**tree_to(payload['condition'],device))
    with torch.autocast('cuda',dtype=torch.bfloat16):u=model.connector(inputs)
    native=dict(fps=g['fps'],height=g['height'],width=g['width'],frames=g['native_frames'],duration_s=g['native_duration_s'])
    av=reference_latents(cfg,make_observation(observation(item,folder)),native,device)
    text=tree_to(text,device)
    torch.manual_seed(p['seed']);prepared=strategy.prepare_training_inputs(with_text(av,text),sampler)
    prepared=append_conditions(prepared,u);neg=append_negative(negative,u)
    torch.manual_seed(p['seed']);sampled=guided_sample(renderer,prepared,neg,**p['generation'])
    video,audio=unpack_latents(sampled,av)
    atomic_torch(out/'payload.pt',cpu(dict(video=video,audio=audio)))
    save(out/'metadata.json',dict(key=item['key'],arm=arm['name'],checkpoint=arm['checkpoint'],checkpoint_sha256=checkpoint_sha,
        step=arm['step'],seed=p['seed'],geometry=g,planning_sha256=sha(folder/'qwen/plan.json'),
        target_description_access=False,listener_future_input=False,generation=p['generation'],seconds=time.monotonic()-start))
    complete(out,[out/'payload.pt',out/'metadata.json'])
    del payload,inputs,text,u,av,prepared,neg,sampled,video,audio
    progress(work,'render',rank,arm=arm['name'],done=index+1,total=len(items),last_seconds=time.monotonic()-start)
 del model,renderer,strategy,sampler,negative;unload()

@contextlib.contextmanager
def decoded_output_lock(out):
 """Serialize native export of one sample across regular/assisting GPUs."""
 out.mkdir(parents=True,exist_ok=True)
 with (out/'.decode.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  yield


def decode(work,items,rank,p,arms):
 import subprocess
 import torch
 import soundfile as sf
 from p2_listen.features import unload
 from omni_av_p3_staged.media import decode_and_save
 _,cfg=data_config()
 with cached_loaders(['load_video_vae_decoder','load_audio_vae_decoder','load_vocoder']),torch.inference_mode():
  for arm in arms:
   for index,item in enumerate(items):
    out=work/'generated'/arm['name']/item['key']
    if ready(out):continue
    with decoded_output_lock(out):
     if ready(out):continue
     start=time.monotonic();src=work/'latents'/arm['name']/item['key']
     if not ready(src):raise ValueError('Missing complete generation latents')
     z=torch.load(src/'payload.pt',map_location='cpu',weights_only=True);g=item['geometry']
     # Failed attempts remain separate; only a complete output is eligible for bind.
     attempt=out/('attempt_'+str(time.time_ns()));attempt.mkdir()
     decoded=decode_and_save(cfg,z['video'],z['audio'],g['fps'],attempt/'padded.mp4','cuda:0',raw_audio_path=attempt/'raw_audio.pt')
     subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','2','-i',str(attempt/'padded.mp4'),
         '-map','0:v:0','-map','0:a:0','-frames:v',str(g['frames']),'-af',f"atrim=duration={g['duration_s']:.9f}",
         '-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-c:a','aac',str(attempt/'generated.mp4')],check=True)
     raw=torch.load(attempt/'raw_audio.pt',map_location='cpu',weights_only=True);wave=raw['wave'];count=round(g['duration_s']*raw['rate'])
     if wave.shape[-1]<count:wave=torch.nn.functional.pad(wave,(0,count-wave.shape[-1]))
     sf.write(attempt/'listener.wav',wave[...,:count].T.numpy(),raw['rate'],subtype='FLOAT')
     info=probe(attempt/'generated.mp4')
     assert info['frames']==g['frames'] and abs(info['fps']-g['fps'])<1e-3
     assert (info['width'],info['height'])==(g['width'],g['height'])
     for name in ['generated.mp4','listener.wav']:(attempt/name).replace(out/name)
     save(out/'metadata.json',dict(read(src/'metadata.json'),decoded=decoded,export=info,decode_seconds=time.monotonic()-start))
     complete(out,[out/'generated.mp4',out/'listener.wav',out/'metadata.json'])
     # These are temporary padded/decoded copies, not the retained model outputs.
     for name in ['padded.mp4','raw_audio.pt']:(attempt/name).unlink()
     attempt.rmdir();del z,raw,wave
     progress(work,'decode',rank,arm=arm['name'],done=index+1,total=len(items),last_seconds=time.monotonic()-start)
 unload()

def probe(path):
 import cv2
 cap=cv2.VideoCapture(str(path))
 try:
  if not cap.isOpened():raise ValueError('Cannot read generated video')
  return dict(frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),fps=float(cap.get(cv2.CAP_PROP_FPS)),
              width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
 finally:cap.release()

def worker(work,stage,rank,limit=None,arm_names=None):
 import torch
 torch.set_num_threads(4);torch.cuda.set_device(0)
 work=Path(work);(work/'locks').mkdir(exist_ok=True)
 lock=(work/'locks'/f'{stage}_{rank:02d}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 p=read(work/'protocol.json');items=read(work/'shards'/f'{rank:02d}.json')
 if limit:items=items[:limit]
 arms=[a for a in p['arms'] if arm_names is None or a['name'] in arm_names]
 progress(work,stage,rank,status='starting',samples=len(items))
 try:
  if stage=='cache':cache(work,items,rank,p)
  else:
   data_config()
   globals()[stage](work,items,rank,p,arms)
  progress(work,stage,rank,status='complete',samples=len(items),arms=[a['name'] for a in arms])
 except Exception as e:
  progress(work,stage,rank,status='failed',error=repr(e));raise
