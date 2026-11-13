"""GPU4-only15-checkpoint screening; native math, shared frozen SFT2752 inputs."""
import argparse,copy,fcntl,importlib.util,json,os,signal,socket,subprocess,sys,time,traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent;SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha,rows,geometry,partition,PYTHON,METRIC_PYTHON
TRAIN=ROOT/'experiments/p2_full_resume5500_to7000_after_ablation_20261112/train'
PARENT=ROOT/'experiments/p2_lora6000_seamless_full_gpu4_20261112'
STEPS=list(range(5600,7001,100));UUID='GPU-c0ca8be5-8935-4e6c-d3d7-a2c5b58c37c0'

def stamp(p):
 p=Path(p);s=p.stat();return dict(path=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def link(a,b):
 a,b=Path(a),Path(b);b.parent.mkdir(parents=True,exist_ok=True)
 if b.exists() or b.is_symlink():assert b.resolve()==a.resolve()
 else:b.symlink_to(a,target_is_directory=a.is_dir())
def gpu():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='4'
 vals=subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=uuid,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20).strip().split(',')
 assert vals[0].strip()==UUID
 return dict(free_gib=float(vals[1])/1024,total_gib=float(vals[2])/1024,util=float(vals[3]))
def runtime_training():
 spec=importlib.util.spec_from_file_location('resume5500_native',TRAIN.parent/'run_runtime.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def child(step):return WORK/f'step_{step:06d}'
def stopped():return (WORK/'STOP').exists()
def own_live(rec):
 p=Path('/proc')/str(rec['pid'])
 try:return p.joinpath('stat').read_text().split(') ',1)[1][0]!='Z' and str(WORK).encode() in p.joinpath('cmdline').read_bytes()
 except FileNotFoundError:return False

def prepare():
 if (WORK/'PREPARED.json').exists():
  assert read(WORK/'PLAN.json')['entry_sha256']==sha(SELF)
  return
 import torch
 torch.set_num_threads(2)
 from p2_whisper_av.bootstrap import activate_ltx
 activate_ltx()
 from p2_testsets_eval.runtime import ready
 from p2_ltx_all_lora.data import data_config
 r=runtime_training();identity=read(TRAIN/'run.json');r.validate(identity['experiment']);assert r.source()==identity['source']
 normalized=r.normalized(identity['experiment']);d=data_config(normalized)
 assert d==read(PARENT/'DATA_CONFIG.json')
 spec=read(ROOT/'evaluation/SEAMLESS_TUNING64.json');prepared=Path(spec['prepared']);refs=rows(prepared/'references.jsonl')
 assert len(refs)==64 and sha(prepared/'references.jsonl')==read(prepared/'seal.json')['references_sha256']
 parent=read(PARENT/'protocol.json');prior={x['id']:x for x in read(PARENT/'shared_items.json')}
 items=[]
 for ref in refs:
  sid=ref['id'];old=prior[sid];assert old['reference']['gt_reference']['row_sha256']==ref['gt_reference']['row_sha256']
  x=dict(old,reference=ref);items.append(x);folder=PARENT/'conditions'/x['key']
  assert ready(folder/'qwen') and ready(folder/'text') and ready(PARENT/'whisper'/x['key'])
  plan=read(folder/'qwen/plan.json');assert plan['adapter']==parent['planner_adapter_identity'] and not plan['target_description_access']
  audit=read(folder/'input_audit.json');assert audit['source_only'] and not audit['listener_future_media_in_model_input']
  assert read(PARENT/'whisper'/x['key']/'metadata.json')['source']['input_audit_sha256']==sha(folder/'input_audit.json')
 save(WORK/'shared_items.json',items);save(WORK/'DATA_CONFIG.json',d)
 cacheproof=[]
 for x in items:
  f=PARENT/'conditions'/x['key'];cacheproof.append(dict(id=x['id'],plan_sha256=sha(f/'qwen/plan.json'),input_audit_sha256=sha(f/'input_audit.json'),qwen_commit_sha256=sha(f/'qwen/complete.json'),text_commit_sha256=sha(f/'text/complete.json'),whisper_commit_sha256=sha(PARENT/'whisper'/x['key']/'complete.json')))
 save(WORK/'CONDITIONS.json',cacheproof)
 checks=[]
 for step in STEPS:
  original=TRAIN/f'step_{step:06d}.pt';marker=read(original.with_suffix('.complete.json'));assert marker['step']==step
  before=stamp(original);v=torch.load(original,map_location='cpu',weights_only=True,mmap=True)
  assert stamp(original)==before and v['step']==step and v['identity']==identity and len(v['rank_rng'])==28
  assert len(v['model'])==4809 and identity['ltx_lora_report']['projections']==1632
  assert {int(s['step']) for s in v['optimizer']['state'].values() if 'step' in s}=={step}
  rec=dict(step=step,path=str(original),sha256=marker['sha256'],stamp=before,model_tensors=len(v['model']),optimizer_states=len(v['optimizer']['state']),rng=28,schema=v['schema']);checks.append(rec);del v
  w=child(step);w.mkdir(exist_ok=True);arm=dict(name=f'continuation28_step{step}_sft2752',kind='ltx_all_lora_full',step=step,checkpoint=str(original),original_checkpoint=str(original),checkpoint_sha256=marker['sha256'],original_sha256=marker['sha256'],planner=parent['planner'])
  for name in ['conditions','whisper','negative']:link(PARENT/name,w/name)
  link(WORK/'shared_items.json',w/'shared_items.json');link(WORK/'DATA_CONFIG.json',w/'DATA_CONFIG.json')
  p=copy.deepcopy(parent)
  for key in ['execution_code','adapter_code','reused_generated_ids','reused_condition_ids','source_media_identity','source_media_identity_sha256','upstream_work','upstream_completion']:p.pop(key,None)
  p.update(schema='resume5500-ltx15-fixedtest64-sft2752-v1',arms=[arm],checkpoint_identity=identity,workers=4,total_samples=64,total_outputs=64,physical_gpus=[4],datasets={'seamless':dict(prepared=str(prepared),references_sha256=sha(prepared/'references.jsonl'),count=64,face_padding=0)},shared_item_queue=True,shared_items_manifest=str(WORK/'shared_items.json'),shared_items_sha256=sha(WORK/'shared_items.json'),seed=137,metric_batch_size=64,metrics_face_device='cuda:0',metrics_emotion_device='cuda:0',metrics_pixel_workers=4,metrics_gt_cache=parent['metrics_gt_cache'],metrics_decoded_gt_cache=str(WORK/'decoded_gt'),metrics_use_released_inference_nodes=False,source_version=spec['source_version'],selection_purpose='test_checkpoint_tuning_not_independent_dev',entry_sha256=sha(SELF))
  save(w/'protocol.json',p)
  for i,shard in enumerate(partition(items,4)):save(w/'shards'/f'{i:02d}.json',shard)
  save(WORK/'prepare_status.json',dict(at=time.time(),step=step,headers_checked=len(checks)))
 save(WORK/'CHECKPOINTS.json',checks)
 save(WORK/'PLAN.json',dict(at=time.time(),entry_sha256=sha(SELF),steps=STEPS,outputs=960,samples=64,physical_gpu=4,planner=parent['planner'],planner_adapter_identity=parent['planner_adapter_identity'],seed=137,generation=parent['generation'],identity=identity,normalized_config=normalized,checkpoint_policy='Full SHA verified once per original checkpoint by sequential CPU worker; mmap original checkpoint, optimizer never moved to GPU; no duplicate35GiB exports',screening='Flag only clear joint LPIPS/FID/FVD inferiority, guard emotion/response/audio; retain ties and incomplete results; no checkpoint files deleted',thresholds=dict(lpips_absolute=.02,fid_relative=.10,fvd_relative=.10,emotion_tolerance=.02,response_tolerance=.02,audio_rate_tolerance=.05),scope='Test tuning64, single seed, coarse screening; no independent final-test claim'))
 save(WORK/'PREPARED.json',dict(success=True,at=time.time(),checkpoints=15,samples=64,outputs=960,reused_qwen_hidden_text_whisper=64,full_checksum_verification='sequential pipeline pending',all_header_steps_identities_and_optimizer_steps_verified=True,all_sources_match_training=True))

def integrity():
 for rec in read(WORK/'CHECKPOINTS.json'):
  if stopped():return
  out=WORK/'integrity'/f'{rec["step"]:06d}.json'
  if out.exists():assert read(out)['stamp']==stamp(rec['path']);continue
  assert stamp(rec['path'])==rec['stamp'];started=time.time();h=sha(rec['path']);assert h==rec['sha256'] and stamp(rec['path'])==rec['stamp']
  save(out,dict(success=True,step=rec['step'],sha256=h,stamp=rec['stamp'],seconds=time.time()-started,at=time.time()))
 save(WORK/'INTEGRITY_COMPLETE.json',dict(success=True,checkpoints=15,at=time.time()))

def pinned(step):
 import torch
 rec=next(r for r in read(WORK/'CHECKPOINTS.json') if r['step']==step)
 verified=WORK/'integrity'/f'{step:06d}.json'
 while not verified.exists():
  if stopped():raise RuntimeError('STOP requested')
  if (WORK/'failure_integrity_0.json').exists():raise RuntimeError('Checkpoint integrity failed')
  time.sleep(2)
 assert read(verified)['sha256']==rec['sha256'] and stamp(rec['path'])==rec['stamp']
 v=torch.load(rec['path'],map_location='cpu',weights_only=True,mmap=True)
 assert v['step']==step and v['identity']==read(WORK/'PLAN.json')['identity'] and v['schema']==rec['schema']
 assert all(torch.isfinite(t).all() for t in v['model'].values())
 assert stamp(rec['path'])==rec['stamp'];return v

def worker(stage,slot=0):
 g=gpu()
 from p2_whisper_av.bootstrap import activate_ltx
 activate_ltx()
 import torch
 torch.set_num_threads(4);torch.cuda.set_device(0)
 cap=88 if stage=='render' else 28
 torch.cuda.set_per_process_memory_fraction(cap/g['total_gib'])
 from p2_testsets_eval import runtime as rt
 from p2_listen.config import extractor_config
 d=read(WORK/'DATA_CONFIG.json');cfg=extractor_config(d);items=read(WORK/'shared_items.json')
 lock=WORK/'locks'/f'{stage}_{slot}.lock';lock.parent.mkdir(exist_ok=True)
 with lock.open('a') as lf,ExitStack() as stack:
  fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB)
  stack.enter_context(patch.object(rt,'data_config',lambda:(d,cfg)))
  save(WORK/'devices'/f'{stage}_{slot}.json',dict(pid=os.getpid(),at=time.time(),host=socket.gethostname(),physical_gpu=4,stage=stage))
  if stage=='render':
   import p2_all_lora_followups_eval_20261109 as native
   import p2_ltx_events.inference as old
   from p2_ltx_all_lora import inference,checkpoint,config
   plan=read(WORK/'PLAN.json');normal=plan['normalized_config'];identity=plan['identity']
   holder={};current_step=None
   def load(path,device='cuda:0'):
    nonlocal current_step
    step=int(Path(path).stem.split('_')[-1]);v=pinned(step)
    if not holder:
     checked_validate=config.validate
     def normalized(c):
      assert c==identity['experiment'];return checked_validate(copy.deepcopy(normal))
     with patch.object(checkpoint,'checked',lambda _:v),patch.object(config,'source_identity',lambda:identity['source']),patch.object(config,'validate',normalized):
      holder['model'],holder['strategy'],holder['sampler']=inference.load_model(path,device)
    else:
     holder['model'].load_state_dict(v['model'],strict=True)
    model=holder['model'];model.eval().requires_grad_(False)
    assert len(model.text_targets)==1632 and list(model.state_dict())==list(v['model'])
    # Native strict load verifies all4809 tensors; same live22B renderer reused.
    current_step=step
    save(WORK/'model_acceptance'/f'{step:06d}.json',dict(success=True,at=time.time(),step=step,strict_state_load=True,model_tensors=4809,all_lora_targets=1632,physical_gpu=4,checkpoint_sha256=read(child(step)/'protocol.json')['arms'][0]['original_sha256'],resident_renderer_reused=len(list((WORK/'model_acceptance').glob('*.json')))>0))
    return holder['model'],holder['strategy'],holder['sampler']
   stack.enter_context(patch.object(old,'load_model',load))
   for step in STEPS:
    if stopped():return
    w=child(step);p=read(w/'protocol.json');native.STEP=step;native.ARM=p['arms'][0]['name'];native.WORK=w
    save(WORK/'render_status.json',dict(step=step,at=time.time(),pid=os.getpid()))
    native.render(w,items,0,p)
    assert all(rt.ready(w/'latents'/native.ARM/x['key']) or rt.ready(w/'generated'/native.ARM/x['key']) for x in items)
    save(w/'RENDER_COMPLETE.json',dict(success=True,at=time.time(),step=step,samples=64))
   save(WORK/'RENDER_COMPLETE.json',dict(success=True,at=time.time(),outputs=960))
  else:
   # Retain VAE decoders across ready passes and checkpoints.
   with rt.cached_loaders(['load_video_vae_decoder','load_audio_vae_decoder','load_vocoder']):
    while not stopped():
     pending=False;remaining=False
     for step in STEPS:
      w=child(step);p=read(w/'protocol.json');arm=p['arms'][0]
      missing=[x for x in items if not rt.ready(w/'generated'/arm['name']/x['key'])];remaining|=bool(missing)
      available=[x for x in missing if rt.ready(w/'latents'/arm['name']/x['key'])]
      if not available:continue
      pending=True;rt.decode(w,items,slot,p,p['arms'])
     if not remaining:break
     if not pending:time.sleep(3)
   save(WORK/f'DECODE_COMPLETE_{slot}.json',dict(success=not stopped(),at=time.time()))

def launch(stage,slot=0,step=0):
 metric=stage in ('metric','merge','report')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONPATH='/private/liqi_workspace/.cache/paired_video_metrics/python' if metric else '')
 if stage=='metric':cmd=[str(METRIC_PYTHON),'-u',str(ROOT/'scripts/p2_pipelined_testsets_eval.py'),'metrics','--work',str(child(step)),'--rank',str(slot),'--arms',read(child(step)/'protocol.json')['arms'][0]['name']]
 elif stage=='merge':cmd=[str(METRIC_PYTHON),'-u',str(ROOT/'scripts/p2_testsets_eval.py'),'merge','--work',str(child(step))]
 else:cmd=[str(PYTHON),'-u',str(SELF),stage,'--slot',str(slot)]
 log=WORK/'logs'/f'{stage}_{step}_{slot}_{time.time_ns()}.log';log.parent.mkdir(exist_ok=True)
 with log.open('w') as stream:p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 save(WORK/'processes'/f'{stage}_{step}_{slot}.json',dict(pid=p.pid,at=time.time(),stage=stage,slot=slot,step=step,command=cmd,log=str(log),physical_gpu=4))
 return p

def publish(step,slot):
 from evaluation.listener_metrics.suite import enrich
 w=child(step);p=read(w/'protocol.json');arm=p['arms'][0];paired=[];mapping=[];selected=read(w/'shards'/f'{slot:02d}.json')
 for x in selected:
  r=x['reference'];folder=w/'generated'/arm['name']/x['key'];m=read(folder/'metadata.json')
  assert m['step']==step and m['checkpoint_sha256']==arm['original_sha256'] and m['planning_adapter']==p['planner_adapter_identity'] and m['seed']==137 and m['generation']==p['generation']
  assert not m['listener_future_input'] and not m['target_description_access']
  assert m['export']['frames']==x['geometry']['frames'] and m['export']['width']==m['export']['height']==512
  paired.append(dict(id=r['id'],dataset='seamless',identity='listener',gt_reference=r['gt_reference'],generated=str(folder/'generated.mp4'),listening_intervals=r['listening_intervals'],source_start=r['source_start'],source_end=r['source_end'],timeline='same_pair_window'))
  mapping.append(dict(id=r['id'],generated_audio=str(folder/'listener.wav')))
 paired=enrich(paired,[x['reference'] for x in selected],mapping,w)
 out=w/'metric_manifests'/arm['name']/'seamless'/f'{slot:02d}.jsonl';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(''.join(json.dumps(r)+'\n' for r in paired))

def results(final=False):
 summaries={}
 for step in STEPS:
  f=child(step)/'RESULTS.json'
  if f.exists():summaries[step]=next(iter(read(f).values()))
 save(WORK/'RESULTS.json',dict(at=time.time(),completed_steps=list(summaries),total_steps=15,results=summaries))
 columns=['psnr','ssim','lpips','fid','fvd','emo_acc','delta_resp']
 lines=['#15份LTX：固定test64＋SFT2752','',f'已完成{len(summaries)}/15；seed137，30步，CFG3/7。test调参子集，单seed初筛。','', '| step | '+' | '.join(columns)+' | 漏回应率 | 静默误发声率 |','|---|'+'---:|'*9]
 for step,r in summaries.items():
  ac=r['audio']['activity']['acoustic'];values=[r['metrics'].get(k) for k in columns]+[ac.get('missed_response_rate'),ac.get('false_sound_clip_rate')]
  lines.append('| '+str(step)+' | '+' | '.join('N/A' if v is None else f'{v:.4f}' for v in values)+' |')
 if final:
  assert len(summaries)==15
  thresholds=read(WORK/'PLAN.json')['thresholds'];flags=[]
  for bad,b in summaries.items():
   for good,a in summaries.items():
    if good==bad:continue
    am,bm=a['metrics'],b['metrics']
    keys=['lpips','fid','fvd','emo_acc','delta_resp']
    if any(am.get(k) is None or bm.get(k) is None for k in keys):continue
    if not all(a['coverage'][k]['pairs']==b['coverage'][k]['pairs']==64 for k in ['lpips','fid','fvd']):continue
    if b['audio']['pairs_with_errors'] or a['audio']['pairs_with_errors']:continue
    worse=(bm['lpips']-am['lpips']>=thresholds['lpips_absolute'] and bm['fid']>=am['fid']*(1+thresholds['fid_relative']) and bm['fvd']>=am['fvd']*(1+thresholds['fvd_relative']))
    guard=am['emo_acc']>=bm['emo_acc']-thresholds['emotion_tolerance'] and am['delta_resp']<=bm['delta_resp']+thresholds['response_tolerance']
    for kind in ['acoustic','speech']:
     for k in ['missed_response_rate','false_sound_clip_rate']:
      av=a['audio']['activity'][kind].get(k);bv=b['audio']['activity'][kind].get(k)
      guard=guard and av is not None and bv is not None and av<=bv+thresholds['audio_rate_tolerance']
    if worse and guard:flags.append(dict(candidate=bad,dominated_by=good,reason='LPIPS worse>=0.02, FID and FVD worse>=10%; no material emotion/response/audio advantage under frozen tolerances'));break
  flagged={r['candidate'] for r in flags};decision=dict(at=time.time(),flags=flags,retain_steps=[s for s in STEPS if s not in flagged],thresholds=thresholds,scope='Conservative engineering screening only, not significance or final ranking; original checkpoints retained')
  save(WORK/'SCREENING.json',decision)
  lines+=['','明显劣势候选：'+(', '.join(map(str,sorted(flagged))) if flagged else '无；当前证据不足以淘汰。'),'保留继续比较：'+', '.join(map(str,decision['retain_steps'])),'','本轮只有64条、单seed。接近或存在指标取舍的权重保留；这些阈值为预先冻结的工程初筛规则，不是统计显著性检验。原始权重全部保留，不自动删除。']
 (WORK/'RESULTS.md').write_text('\n'.join(lines)+'\n')

def controller():
 gpu();prepare()
 from p2_testsets_eval.runtime import ready
 with (WORK/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  live=[f.name for f in (WORK/'processes').glob('*.json') if own_live(read(f))]
  if live:raise RuntimeError('Existing owned children; do not duplicate: '+str(live))
  save(WORK/'CONTROLLER.json',dict(at=time.time(),pid=os.getpid(),host=socket.gethostname(),physical_gpu=4))
  jobs={};attempts={};items=read(WORK/'shared_items.json');merged=set()
  def start(stage,slot=0,step=0):
   key=(stage,slot,step);attempts[key]=attempts.get(key,0)+1;jobs[key]=launch(stage,slot,step)
  try:
   if not (WORK/'INTEGRITY_COMPLETE.json').exists():start('integrity')
   if not (WORK/'RENDER_COMPLETE.json').exists():start('render')
   while not stopped():
    for key,proc in list(jobs.items()):
     if proc.poll() is None:continue
     del jobs[key]
     if proc.returncode:
      if attempts[key]>=3:raise RuntimeError(f'{key} failed3 attempts; inspect logs')
      start(*key)
    g=gpu();generated=0;latent=0;scored=0;pending_decode=False
    for step in STEPS:
     w=child(step);p=read(w/'protocol.json');arm=p['arms'][0];name=arm['name']
     done={x['key'] for x in items if ready(w/'generated'/name/x['key'])};generated+=len(done)
     ld={x['key'] for x in items if ready(w/'latents'/name/x['key'])};latent+=len(ld);pending_decode|=bool(ld-done)
     if (w/'COMPLETE.json').exists():merged.add(step);continue
     score_done=[]
     for slot in range(4):
      marker=w/'progress/metrics'/f'{name}_{slot:02d}.json'
      if marker.exists() and read(marker).get('status')=='complete':score_done.append(slot);scored+=1;continue
      #1 scorer alongside renderer;4 once all sampling ends.
      cap=1 if ('render',0,0) in jobs else 4
      if ('metric',slot,step) in jobs or sum(k[0]=='metric' for k in jobs)>=cap or g['free_gib']<26:continue
      if all(x['key'] in done for x in read(w/'shards'/f'{slot:02d}.json')):
       publish(step,slot);start('metric',slot,step);g=gpu()
     if len(score_done)==4 and ('merge',0,step) not in jobs:
      start('merge',0,step)
    if pending_decode:
     cap=1 if ('render',0,0) in jobs else 2
     for slot in range(cap):
      if ('decode',slot,0) not in jobs and g['free_gib']>=38:start('decode',slot);g=gpu()
    # Completion markers and native pooled merges, not process exits, define success.
    save(WORK/'status.json',dict(at=time.time(),stage='generation_decode_scoring',generated=generated,latent=latent,total=960,completed_checkpoints=sorted(merged),scored_pending_blocks=scored,active=[list(k) for k in jobs],gpu=g))
    results()
    if len(merged)==15 and not jobs:
     results(final=True);save(WORK/'COMPLETE.json',dict(success=True,at=time.time(),outputs=960,steps=STEPS,screening=str(WORK/'SCREENING.json')));break
    time.sleep(15)
  finally:
   for proc in jobs.values():
    if proc.poll() is None:
     try:os.killpg(proc.pid,signal.SIGTERM)
     except ProcessLookupError:pass

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','controller','integrity','render','decode','report']);parser.add_argument('--slot',type=int,default=0);a=parser.parse_args()
 try:
  if a.action=='prepare':prepare()
  elif a.action=='controller':controller()
  elif a.action=='integrity':integrity()
  elif a.action=='report':results(final=True)
  else:worker(a.action,a.slot)
 except BaseException:
  save(WORK/f'failure_{a.action}_{a.slot}.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
