"""ViCo test43: frozen combined8000, native512/25 -> score256/30.

Adapters are process local; checkpoint-bound implementation files stay unchanged.
"""
import argparse,contextlib,fcntl,hashlib,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha
WORK=ROOT/'experiments/p2_vico256_combined8000_20261106';PARENT=ROOT/'experiments/p2_combined_whisper_eval_20261106';PY=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python';MPY=ROOT/'compare/.envs/vico_eval/bin/python'
ARM='base_event_feature_whisper_8000';CURRENT={}

def prepare(work):
 if (work/'protocol.json').exists():return read(work/'protocol.json')
 pre=read(work/'INPUT_PREFLIGHT.json');assert pre['samples']==43 and pre['source_only']
 parent=read(PARENT/'protocol.json');p={k:parent[k] for k in ['planner','planner_adapter','planner_max_new_tokens','planner_retry_tokens','generation','whisper_encoder','whisper_batch','feature_code','checkpoint_identity','sampling']}
 arm=next(a for a in parent['arms'] if a['step']==8000);assert arm['name']==ARM
 p.update(schema='vico-test43-combined8000-native51225-score25630-v1',created_at=time.time(),arms=[arm],seed=137,planner_batch=16,text_batch=16,workers=1,total_samples=43,total_outputs=43,shared_item_queue=False,authorized_scope='node8 172.29.104.237 SSH22 physical GPU7 only; no eviction',generation_geometry=dict(width=512,height=512,fps=25),evaluation_geometry=dict(width=256,height=256,fps=30),temporal_conversion='nearest source timestamp, round-half-up; no speed change or motion interpolation',spatial_conversion='torch bicubic align_corners=False antialias=True, float RGB before encoding',worker_cli=str(Path(__file__).resolve()))
 items=[];index=read(ROOT/'evaluation/vico/index.json');refs={r['uuid']:r for r in index['clips'] if r['split']=='test'}
 for audit in pre['samples_audit']:
  sid=audit['id'];r=refs[sid];gt=audit['original_gt'];n=math.ceil(gt['duration_s']*25-1e-7);native=8*math.ceil((n-1)/8)+1
  g=dict(fps=25.,height=512,width=512,source_width=256,source_height=256,frames=n,native_frames=native,duration_s=gt['duration_s'],native_duration_s=native/25,gt_frames=gt['frames'],gt_fps=30.)
  items.append(dict(id=sid,key='vico/'+sid,dataset='vico',reference=r,geometry=g))
 # Short/long smoke outputs are retained as production outputs.
 items.sort(key=lambda x:(x['geometry']['frames'],x['id']));items=[items[0],items[-1]]+items[1:-1]
 p['datasets']={'vico':dict(count=43,split='test',annotation=index['anno']['test'],annotation_sha256=sha(index['anno']['test']),gt_dir=index['video_dir']['test'],index_sha256=sha(ROOT/'evaluation/vico/index.json'))};p['input_preflight_sha256']=sha(work/'INPUT_PREFLIGHT.json')
 files=[Path(__file__),ROOT/'scripts/p2_vico_input_preflight.py',ROOT/'scripts/p2_combined_whisper_eval.py',ROOT/'src/p2_testsets_eval/runtime.py',ROOT/'src/p2_testsets_eval/whisper_runtime.py',ROOT/'scripts/p2_vico_official_metrics.py'];p['execution_code']={str(f):sha(f) for f in files}
 save(work/'shards/00.json',items);save(work/'protocol.json',p);save(work/'status.json',dict(stage='prepared',completed=0,total=43));return p

def verify(work):
 p=read(work/'protocol.json')
 for f,h in p['execution_code'].items():
  if sha(f)!=h:raise ValueError('Frozen evaluator source changed: '+f)
 if sha(work/'INPUT_PREFLIGHT.json')!=p['input_preflight_sha256']:raise ValueError('Inputs changed')
 spec=p['datasets']['vico']
 if sha(spec['annotation'])!=spec['annotation_sha256']:raise ValueError('Official test list changed')
 return p

def source_wave(item,folder):
 import numpy as np,soundfile as sf
 from scipy.signal import resample_poly
 audit=read(folder/'input_audit.json');path=audit['speaker_audio_path']
 if sha(path)!=audit['speaker_audio_sha256']:raise ValueError('Changed speaker WAV')
 with sf.SoundFile(path) as f:rate=f.samplerate;wave=f.read(dtype='float32',always_2d=True).mean(1)
 common=math.gcd(rate,16000);wave=resample_poly(wave,16000//common,rate//common).astype(np.float32);n=round(item['geometry']['duration_s']*16000);delta=n-len(wave)
 if delta!=audit['endpoint_adjustment_samples'] or abs(delta)>1600:raise ValueError('Unexpected speaker endpoint adjustment')
 wave=np.pad(wave,(0,max(0,delta)))[:n]
 if hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest()!=audit['waveform_sha256']:raise ValueError('CPU/GPU input waveform differs')
 return wave,dict(input_audit_sha256=sha(folder/'input_audit.json'),waveform_sha256=audit['waveform_sha256'],source_sha256=audit['speaker_audio_sha256'],samples=n,rate=16000,origin_s=0.,silence_removed=False,source_only=True,endpoint_adjustment_samples=delta)

def conditioning(item,folder):
 from PIL import Image
 wave,_=source_wave(item,folder);audit=read(folder/'input_audit.json');assert audit['decoded_listener_frames']==1 and audit['source_only']
 with Image.open(folder/'first_frame.png') as im:image=im.convert('RGB')
 if image.size!=(512,512):raise ValueError('Wrong reference size')
 return dict(sample_id=item['id'],duration=item['geometry']['duration_s'],image=image,audio=wave)

# Keep bootstrap outside imports that may select the shared LTX version.
def worker(work,stage):
 from p2_combined_whisper_eval import combined_runtime
 with combined_runtime():
  import torch
  from p2_testsets_eval import runtime as rt,whisper_runtime as wr
  torch.set_num_threads(4);torch.cuda.set_device(0);p=verify(work);items=read(work/'shards/00.json');arms=p['arms']
  rt.data_config()
  import omni_av.data as data_module
  OriginalObservation=data_module.Observation
  class ViCoObservation(OriginalObservation):
   def __post_init__(self):
    if self.output_duration_s<=30:return super().__post_init__()
    if not (self.sample_id and 0<=self.speaker_start_s<self.speaker_end_s<=self.output_duration_s<=35):raise ValueError('Invalid ViCo full-clip bounds')
  data_module.Observation=ViCoObservation
  save(work/'device.json',dict(host=os.uname().nodename,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),cuda_name=torch.cuda.get_device_name(0),cuda_uuid=str(torch.cuda.get_device_properties(0).uuid),pid=os.getpid(),stage=stage))
  if os.environ.get('CUDA_VISIBLE_DEVICES')!='7' or torch.cuda.device_count()!=1:raise ValueError('Expected only physical GPU7')
  with patch.object(rt,'conditioning',conditioning),patch.object(wr,'speaker_wave',source_wave):
   if stage=='conditions':rt.cache(work,items,0,p)
   elif stage=='whisper':wr.cache(work,items,0,p)
   elif stage=='render':render_export(work,items,p,rt,wr)
   else:raise ValueError(stage)

def render_export(work,items,p,rt,wr):
 import torch
 from p2_testsets_eval import shared_items
 import ltx_trainer.video_utils as vu
 native_save=vu.save_video;native_claim=shared_items.claim_items
 score=work/'scoring'/ARM;score.mkdir(parents=True,exist_ok=True)
 def save_video(pixels,path,**kw):
  # This hook receives raw float decoder pixels, avoiding decode/re-encode scaling.
  native_save(pixels,path,**kw)
  item=CURRENT['item'];g=item['geometry'];assert kw['video_format']=='CFHW' and kw['fps']==25
  x=pixels[:,:g['frames']].permute(1,0,2,3);assert tuple(x.shape[1:])==(3,512,512) and 0<=float(x.min())<=float(x.max())<=1
  small=torch.cat([torch.nn.functional.interpolate(v,size=(256,256),mode='bicubic',align_corners=False,antialias=True).clamp(0,1) for v in x.split(32)])
  indices=torch.floor(torch.arange(g['gt_frames'],dtype=torch.float64)*25/30+.5).long().clamp(max=len(small)-1)
  video=small[indices].permute(1,0,2,3).contiguous();wave=kw['audio'];rate=kw['audio_sample_rate'];count=round(g['duration_s']*rate);wave=wave[...,:count]
  if wave.shape[-1]<count:wave=torch.nn.functional.pad(wave,(0,count-wave.shape[-1]))
  dest=score/(item['id']+'.listener.mp4');tmp=dest.with_suffix('.part.mp4');native_save(video,tmp,fps=30,audio=wave,audio_sample_rate=rate,video_format='CFHW');tmp.replace(dest)
  save(work/'conversion'/f'{item["id"]}.json',dict(source_fps=25,target_fps=30,source_frames=g['frames'],target_frames=g['gt_frames'],source_size=[512,512],target_size=[256,256],source_frame_indices=indices.tolist(),max_timestamp_error_s=float((indices.double()/25-torch.arange(g['gt_frames']).double()/30).abs().max()),scaling=p['spatial_conversion'],encoding='one libx264 CRF18 encode directly from downsampled decoder float RGB; yuv420p',file_sha256=sha(dest)))
 def export(item):
  CURRENT['item']=item
  with patch.object(vu,'save_video',save_video):rt.decode(work,[item],0,p,p['arms'])
  dest=score/(item['id']+'.listener.mp4');info=rt.probe(dest);g=item['geometry']
  if not (info['width']==info['height']==256 and info['frames']==g['gt_frames'] and abs(info['fps']-30)<1e-6):raise ValueError('Scoring export differs from original GT grid')
  save(work/'conversion'/f'{item["id"]}.verified.json',dict(probe=info,source_metadata=str(work/'generated'/ARM/item['key']/'metadata.json'),source_only=True,score_sha256=sha(dest)))
  done=len(list((work/'conversion').glob('*.verified.json')));save(work/'status.json',dict(stage='generating_and_exporting',completed=done,total=43,last=item['id'],at=time.time()))
  if done==2:save(work/'GPU_SMOKE_COMPLETE.json',dict(short=items[0]['id'],long=items[1]['id'],outputs_retained=True,source_only=True,generation=[512,512,25],scoring=[256,256,30],at=time.time()))
 def claims(w,fallback,protocol,arm,stage):
  for item in native_claim(w,fallback,protocol,arm,stage):
   yield item
   if stage=='render':export(item)
 # Recover outputs from a previously completed latent stage before loading DiT.
 for item in items:
  if rt.ready(work/'latents'/ARM/item['key']) and not (work/'conversion'/f'{item["id"]}.verified.json').exists():export(item)
 with rt.cached_loaders(['load_video_vae_decoder','load_audio_vae_decoder','load_vocoder']),patch.object(shared_items,'claim_items',claims):wr.render(work,items,0,p,p['arms'])
 for item in items:
  if not (work/'conversion'/f'{item["id"]}.verified.json').exists():export(item)
 if len(list((work/'conversion').glob('*.verified.json')))!=43:raise ValueError('Incomplete scored media')
 save(work/'GENERATION_COMPLETE.json',dict(outputs=43,at=time.time(),generation=[512,512,25],scoring=[256,256,30]))

def run(work):
 work.mkdir(exist_ok=True);p=verify(work)
 with (work/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   for stage in ['conditions','whisper','render']:
    if (work/(stage.upper()+'_COMPLETE.json')).exists():continue
    save(work/'status.json',dict(stage=stage,total=43,at=time.time()))
    with (work/(stage+'.log')).open('ab') as log:
     code=subprocess.call([str(PY),str(Path(__file__).resolve()),'worker','--stage',stage,'--work',str(work)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    if code:raise RuntimeError(f'{stage} failed, exit {code}; see {work/(stage+".log")}')
    save(work/(stage.upper()+'_COMPLETE.json'),dict(at=time.time()))
   save(work/'status.json',dict(stage='official_metrics',outputs=43,at=time.time()))
   with (work/'metrics.log').open('ab') as log:subprocess.run([str(MPY),str(ROOT/'scripts/p2_vico_official_metrics.py'),'--work',str(work)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
   save(work/'COMPLETE.json',dict(outputs=43,protocol_sha256=sha(work/'protocol.json'),results=read(work/'RESULTS.json'),at=time.time()));save(work/'status.json',dict(stage='complete',outputs=43,at=time.time()))
  except Exception as e:save(work/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),at=time.time()));save(work/'status.json',dict(stage='failed',error=repr(e),at=time.time()));raise

def launch(work):
 prepare(work)
 old=work/'LAUNCH.json'
 if old.exists():
  pid=read(old)['pid']
  try:
   if 'p2_vico256_combined_eval.py' in Path(f'/proc/{pid}/cmdline').read_text():raise RuntimeError('Controller is already running')
  except FileNotFoundError:pass
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',MASTER_PORT='20491',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
 with (work/'controller.log').open('ab') as log:proc=subprocess.Popen([str(PY),str(Path(__file__).resolve()),'run','--work',str(work)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(old,dict(pid=proc.pid,host=os.uname().nodename,physical_gpu=7,master_port=20491,single_process_no_distributed_rendezvous=True,at=time.time()));print(json.dumps(read(old)),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','launch','run','worker']);parser.add_argument('--work',type=Path,default=WORK);parser.add_argument('--stage');a=parser.parse_args()
 if a.command=='prepare':prepare(a.work)
 elif a.command=='launch':launch(a.work)
 elif a.command=='run':run(a.work)
 else:worker(a.work,a.stage)
