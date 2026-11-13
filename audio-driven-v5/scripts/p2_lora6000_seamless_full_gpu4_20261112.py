"""Single node11 GPU4, native all-LoRA6000/SFT2752, current Seamless6624."""
import argparse,copy,fcntl,json,os,signal,socket,subprocess,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha,rows,geometry,PYTHON,METRIC_PYTHON
WORK=ROOT/'experiments/p2_lora6000_seamless_full_gpu4_20261112'
OLD=ROOT/'experiments/p2_ltx_all_lora6000_seamless200_sft2752_eval32_20261109'
REPAIR=ROOT/'experiments/seamless_upright_eval_fix_20261112/generation6000'
INPUTS=ROOT/'experiments/glare_final_seamless_test6624_20261112'
ARM='ltx_all_lora_event_feature_whisper_6000_sft2752'
UUID='GPU-c0ca8be5-8935-4e6c-d3d7-a2c5b58c37c0'


def link(src,dst):
 src,dst=Path(src),Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists() or dst.is_symlink():assert dst.resolve()==src.resolve()
 else:dst.symlink_to(src,target_is_directory=src.is_dir())


def gpu():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='4'
 r=subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=uuid,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20).strip().split(',')
 assert r[0].strip()==UUID
 return dict(free_gib=float(r[1])/1024,total_gib=float(r[2])/1024,util=float(r[3]))


def verify():
 p=read(WORK/'protocol.json');assert sha(p['shared_items_manifest'])==p['shared_items_sha256']
 assert sha(Path(p['datasets']['seamless']['prepared'])/'references.jsonl')==p['datasets']['seamless']['references_sha256']
 for path,digest in p['adapter_code'].items():assert sha(path)==digest,('Source changed',path)
 assert p['physical_gpus']==[4] and p['total_samples']==6624 and p['arms'][0]['step']==6000
 return p


def prepare():
 if (WORK/'PREPARED.json').exists():return verify()
 from p2_testsets_eval.runtime import ready
 WORK.mkdir(parents=True,exist_ok=True)
 current=read(ROOT/'evaluation/SEAMLESS_CURRENT.json');prepared=Path(current['prepared_full6624']['prepared'])
 refs=rows(prepared/'references.jsonl');assert len(refs)==len({r['id'] for r in refs})==6624
 assert sha(prepared/'references.jsonl')==read(prepared/'seal.json')['references_sha256']
 oldp=read(OLD/'protocol.json');assert oldp['arms'][0]['step']==6000 and oldp['seed']==137
 rec=read(OLD/'export/RECEIPT.json');assert rec['all_model_tensors_bitwise_equal'] and rec['step']==6000
 assert rec['original_sha256']==oldp['arms'][0]['original_sha256']==read(Path(rec['original']).with_suffix('.complete.json'))['sha256']
 old_items={x['id']:x for x in read(OLD/'shared_items.json')}
 repaired={x['id']:x for x in read(REPAIR/'shared_items.json')}
 assert read(REPAIR/'protocol.json')['planner_adapter_identity']==oldp['planner_adapter_identity']
 items=[];reused=[];cached=[]
 for r in refs:
  item=dict(key='seamless/'+r['id'],id=r['id'],dataset='seamless',reference=r,geometry=geometry(r,'seamless'));items.append(item)
  src=None;prior=old_items.get(r['id'])
  if prior and prior['reference']['gt_reference']['row_sha256']==r['gt_reference']['row_sha256']:
   src=OLD
   if ready(OLD/'generated'/ARM/item['key']):
    m=read(OLD/'generated'/ARM/item['key']/'metadata.json')
    assert m['checkpoint_sha256']==rec['original_sha256'] and m['seed']==137 and m['generation']==oldp['generation']
    link(OLD/'generated'/ARM/item['key'],WORK/'generated'/ARM/item['key']);reused.append(r['id'])
  elif r['id'] in repaired and repaired[r['id']]['reference']['gt_reference']['row_sha256']==r['gt_reference']['row_sha256']:src=REPAIR
  if src:
   folder=src/'conditions'/item['key']
   assert ready(folder/'qwen') and ready(folder/'text') and ready(src/'whisper'/item['key'])
   assert read(folder/'qwen/plan.json')['adapter']==oldp['planner_adapter_identity']
   assert read(src/'whisper'/item['key']/'metadata.json')['source']['input_audit_sha256']==sha(folder/'input_audit.json')
   for name in ['first_frame.png','input_audit.json','qwen','text']:link(folder/name,WORK/'conditions'/item['key']/name)
   link(src/'whisper'/item['key'],WORK/'whisper'/item['key']);cached.append(r['id'])
 # Available historical outputs first; never reuse the four changed images' generations.
 priority=set(reused)
 items.sort(key=lambda x:(x['id'] not in priority,-x['geometry']['native_frames'],x['id']))
 save(WORK/'shared_items.json',items);link(OLD/'export',WORK/'export');link(OLD/'negative/00',WORK/'negative/00')
 save(WORK/'DATA_CONFIG.json',read(OLD/'DATA_CONFIG.json'))
 p=copy.deepcopy(oldp)
 for k in ['metrics_decoded_gt_cache','upstream_work','upstream_completion','selection','selection_sha256','execution_code','source_media_identity','source_media_identity_sha256']:p.pop(k,None)
 shards=[items[i:i+32] for i in range(0,len(items),32)]
 cluster=read(OLD/'cluster.json');cluster['nodes']=[n for n in cluster['nodes'] if n['name']=='node11'];cluster['nodes'][0]['gpus']=[4]
 save(WORK/'cluster.json',cluster)
 p.update(schema='all-lora6000-seamless-current6624-node11-gpu4-v1',created_at=time.time(),
  datasets={'seamless':dict(prepared=str(prepared),references_sha256=sha(prepared/'references.jsonl'),count=6624,face_padding=0)},
  total_samples=6624,total_outputs=6624,workers=len(shards),physical_gpus=[4],worker_cli=str(SELF),cluster=str(WORK/'cluster.json'),metrics_cluster=str(WORK/'cluster.json'),
  shared_item_queue=True,shared_items_manifest=str(WORK/'shared_items.json'),shared_items_sha256=sha(WORK/'shared_items.json'),
  source_version=current['version'],changed_from_historical_full_ids=current['prepared_full6624']['replaced_ids'],
  planner_batch=16,text_batch=16,whisper_batch=16,metrics_workers_per_gpu=2,
  metric_batch_size=64,metrics_face_device='cuda:0',metrics_emotion_device='cuda:0',metrics_pixel_workers=4,
  metrics_use_released_inference_nodes=False,reused_generated_ids=reused,reused_condition_ids=cached,
  adapter_code={str(q):sha(q) for q in [SELF,ROOT.parent/'OmniResponse/evaluation/seamless/reader.py',ROOT/'scripts/j_upright_response_sft_data.py']},
  authorized_scope='node11 physical GPU4 only; GPU7 ResponseNet queue and all other tasks preserved',
  acceleration='Reuse196 unchanged generated pairs and200 source-matched SFT/Gemma/Whisper caches; reusable source PNG/audio audit; cache and generation overlap; persistent native generator/VAE; ready32 scoring blocks; CPU pixel/read prefetch; global pooled FID/FVD')
 assert len(reused)==196 and len(cached)==200 and len(shards)==207
 save(WORK/'protocol.json',p)
 for i,shard in enumerate(shards):save(WORK/'shards'/f'{i:02d}.json',shard)
 (WORK/'ready_queue.jsonl').write_text(''.join(json.dumps({'key':x['key']})+'\n' for x in items if x['id'] in set(cached)))
 save(WORK/'PREPARED.json',dict(passed=True,at=time.time(),count=6624,reused_generated=len(reused),reused_conditions=len(cached),remaining_generations=6624-len(reused),source_version=p['source_version'],physical_gpu=4,checkpoint_sha256=rec['original_sha256']))
 return verify()


def inputs():
 p=verify()
 from p2_testsets_eval import runtime as rt
 old={s['id']:s for s in read(INPUTS/'protocol.json')['samples']}
 def one(x):
  folder=WORK/'conditions'/x['key']
  if (folder/'input_audit.json').exists():return
  s=read(old[x['id']]['ready'])['sample'];a=s['evaluation_reference']['gt_reference'];b=x['reference']['gt_reference']
  if all(a.get(k)==b.get(k) for k in ['row_sha256','crop_config_sha256','reader_contract']):
   for name,source in [('first_frame.png',s['reference']),('input_audit.json',s['source_input_audit'])]:link(source,folder/name)
  else:rt.conditioning(x,folder)
 with ThreadPoolExecutor(max_workers=8) as pool:
  for _ in pool.map(one,read(WORK/'shared_items.json')):pass
 save(WORK/'INPUT_COMPLETE.json',dict(count=6624,at=time.time(),source_only=True))


def ready_claims(work,fallback,p,arm,stage):
 """Shared GPU4/7 render claims; commit stays protected across native sampling."""
 assert stage=='render'
 from p2_testsets_eval.runtime import ready
 bykey={x['key']:x for x in read(WORK/'shared_items.json')}
 def done(key):return ready(WORK/'generated'/ARM/key) or ready(WORK/'latents'/ARM/key)
 with (WORK/'ready_queue.jsonl').open() as stream:
  while not (WORK/'STOP').exists():
   line=stream.readline()
   if not line:
    if (WORK/'CACHE_COMPLETE.json').exists():
     # A peer can still own the final in-flight sample. Revisit uncommitted IDs.
     if all(done(key) for key in bykey):break
     stream.seek(0)
    time.sleep(2);continue
   key=json.loads(line)['key']
   if done(key):continue
   x=bykey[key];folder=WORK/'conditions'/key
   assert ready(folder/'qwen') and ready(folder/'text') and ready(WORK/'whisper'/key)
   path=WORK/'sample_claims/render'/ARM/(str(__import__('hashlib').sha256(key.encode()).hexdigest())+'.lock')
   path.parent.mkdir(parents=True,exist_ok=True)
   with path.open('a') as lock:
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:continue
    if done(key):continue
    yield x


def worker(stage,slot):
 p=verify();g=gpu();assert '172.29.189.119' in subprocess.check_output(['hostname','-I'],text=True).split()
 if stage=='inputs':inputs();return
 from p2_whisper_av.bootstrap import activate_ltx
 activate_ltx()
 import torch
 torch.set_num_threads(4);torch.cuda.set_device(0)
 policy=read(WORK/'RESOURCE_POLICY.json') if (WORK/'RESOURCE_POLICY.json').exists() else {}
 cap={'cache':policy.get('cache_memory_gib',64),'render':84,'decode':26}[stage]
 torch.cuda.set_per_process_memory_fraction(cap/g['total_gib'],0)
 from p2_testsets_eval import runtime as rt,whisper_runtime as wr
 from p2_listen.config import extractor_config
 d=read(WORK/'DATA_CONFIG.json');cfg=extractor_config(d);items=read(WORK/'shared_items.json')
 lock=WORK/'locks'/f'{stage}_{slot}.lock';lock.parent.mkdir(exist_ok=True)
 with lock.open('a') as lf,ExitStack() as stack:
  fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB)
  stack.enter_context(patch.object(rt,'data_config',lambda:(d,cfg)));stack.enter_context(patch.object(wr,'data_config',lambda:(d,cfg)))
  save(WORK/'devices'/f'{stage}_{slot}.json',dict(pid=os.getpid(),at=time.time(),physical_gpu=4,stage=stage))
  if stage=='cache':
   p=dict(p,text_batch=policy.get('text_batch',8))
   save(WORK/'CACHE_RUNTIME.json',dict(at=time.time(),pid=os.getpid(),text_batch=p['text_batch'],cache_memory_gib=cap,planner_batch=p['planner_batch'],whisper_batch=p['whisper_batch']))
   while not (WORK/'INPUT_COMPLETE.json').exists():
    if (WORK/'STOP').exists():return
    time.sleep(2)
   # Preserve exact native hashes but avoid re-reading long source WAVs per pair.
   memo={};native_sha=wr.sha
   def cached_sha(path):
    path=Path(path);s=path.stat();key=(str(path),s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if key not in memo:memo[key]=native_sha(path)
    return memo[key]
   stack.enter_context(patch.object(wr,'sha',cached_sha))
   wr.cache(WORK,items,0,p)
   from p2_ltx6000_sft_eval_20261107 import SFTInferenceSession
   stack.enter_context(patch('p2_j_sft_hidden.Reader',SFTInferenceSession))
   old_condition=rt.conditioning
   def condition(x,folder):
    from PIL import Image
    wave,_=wr.speaker_wave(x,folder)
    with Image.open(folder/'first_frame.png') as image:image=image.convert('RGB')
    return dict(sample_id=x['id'],duration=x['geometry']['duration_s'],image=image,audio=wave)
   old_complete=rt.complete
   def commit(folder,files,**kw):
    old_complete(folder,files,**kw)
    if Path(folder).name=='text':
     key='/'.join(Path(folder).relative_to(WORK/'conditions').parts[:-1])
     with (WORK/'ready_queue.jsonl').open('a') as f:f.write(json.dumps(dict(key=key))+'\n');f.flush()
   stack.enter_context(patch.object(rt,'conditioning',condition));stack.enter_context(patch.object(rt,'complete',commit))
   # Re-publish committed text from any interrupted cache batch before continuing.
   queued={json.loads(line)['key'] for line in (WORK/'ready_queue.jsonl').read_text().splitlines()}
   with (WORK/'ready_queue.jsonl').open('a') as f:
    for x in items:
     if x['key'] not in queued and rt.ready(WORK/'conditions'/x['key']/'text'):f.write(json.dumps(dict(key=x['key']))+'\n')
   rt.cache(WORK,items,0,p)
   assert all(rt.ready(WORK/'conditions'/x['key']/'text') and rt.ready(WORK/'whisper'/x['key']) for x in items)
   save(WORK/'CACHE_COMPLETE.json',dict(count=6624,at=time.time(),planner_step=2752))
  elif stage=='render':
   import p2_ltx6000_latest_sft_dual200_20261107 as native
   import p2_ltx_events.inference as previous
   import p2_all_lora6000_eval_20261109 as loading
   def load(path,device='cuda:0'):
    value=loading.load_model(WORK,path,device);assert len(value[0].text_targets)==1632
    save(WORK/'MODEL_ACCEPTANCE.json',dict(passed=True,pid=os.getpid(),at=time.time(),step=6000,physical_gpu=4,targets=1632,checkpoint_sha256=p['arms'][0]['original_sha256']))
    return value
   stack.enter_context(patch.object(native,'ARM',ARM));stack.enter_context(patch.object(previous,'load_model',load))
   stack.enter_context(patch('p2_testsets_eval.shared_items.claim_items',ready_claims))
   native.render(WORK,items,0,p)
   assert all(rt.ready(WORK/'latents'/ARM/x['key']) or rt.ready(WORK/'generated'/ARM/x['key']) for x in items),'Renderer stopped before all6624 commits'
  else:rt.decode(WORK,items,slot,p,p['arms'])


def launch(stage,slot=0):
 metric=stage in ('metric','merge','bind')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONPATH='/private/liqi_workspace/.cache/paired_video_metrics/python' if metric else '')
 if stage=='metric':cmd=[str(METRIC_PYTHON),'-u',str(ROOT/'scripts/p2_pipelined_testsets_eval.py'),'metrics','--work',str(WORK),'--rank',str(slot),'--arms',ARM]
 elif stage in ('merge','bind'):cmd=[str(METRIC_PYTHON),'-u',str(ROOT/'scripts/p2_testsets_eval.py'),stage,'--work',str(WORK)];env['OPENBLAS_NUM_THREADS']='16'
 else:cmd=[str(PYTHON),'-u',str(SELF),'worker','--stage',stage,'--slot',str(slot)]
 log=WORK/'logs'/f'{stage}_{slot}_{time.time_ns()}.log';log.parent.mkdir(exist_ok=True)
 with log.open('w') as f:proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 save(WORK/'processes'/f'{stage}_{slot}.json',dict(pid=proc.pid,at=time.time(),stage=stage,slot=slot,physical_gpu=4,log=str(log),command=cmd))
 return proc


def publish(shard):
 from evaluation.listener_metrics.suite import enrich
 selected=read(WORK/'shards'/f'{shard:02d}.json');paired=[];mapping=[]
 for x in selected:
  r=x['reference'];folder=WORK/'generated'/ARM/x['key'];m=read(folder/'metadata.json')
  assert m['step']==6000 and m['seed']==137 and m['checkpoint_sha256']==read(WORK/'protocol.json')['arms'][0]['original_sha256']
  assert m['export']['frames']==x['geometry']['frames'] and abs(m['export']['fps']-x['geometry']['fps'])<1e-3 and m['export']['width']==m['export']['height']==512 and not m['listener_future_input'] and not m['target_description_access']
  paired.append(dict(id=r['id'],dataset='seamless',identity='listener',gt_reference=r['gt_reference'],generated=str(folder/'generated.mp4'),listening_intervals=r['listening_intervals'],source_start=r['source_start'],source_end=r['source_end'],timeline='same_pair_window'))
  mapping.append(dict(id=r['id'],generated_audio=str(folder/'listener.wav')))
 paired=enrich(paired,[x['reference'] for x in selected],mapping,WORK)
 path=WORK/'metric_manifests'/ARM/'seamless'/f'{shard:02d}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(''.join(json.dumps(r)+'\n' for r in paired))


def controller():
 p=prepare();gpu()
 from p2_testsets_eval.runtime import ready
 with (WORK/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  # Only adopt this run's children; never duplicate a surviving model.
  for f in (WORK/'processes').glob('*.json'):
   r=read(f);proc=Path('/proc')/str(r['pid'])
   if proc.exists() and proc.joinpath('stat').read_text().split(') ',1)[1][0]!='Z' and (str(SELF).encode() in proc.joinpath('cmdline').read_bytes() or str(WORK).encode() in proc.joinpath('cmdline').read_bytes()):raise RuntimeError('Existing run child still alive: '+str(f))
  save(WORK/'CONTROLLER.json',dict(pid=os.getpid(),at=time.time(),host=socket.gethostname(),physical_gpu=4))
  items=read(WORK/'shared_items.json');jobs={};attempts={};started=set();metric_done=set()
  def start(key,stage,slot=0):
   attempts[key]=attempts.get(key,0)+1;jobs[key]=launch(stage,slot)
  try:
   if not (WORK/'INPUT_COMPLETE.json').exists():start('inputs','inputs')
   if not (WORK/'CACHE_COMPLETE.json').exists():start('cache','cache')
   if not all(ready(WORK/'generated'/ARM/x['key']) or ready(WORK/'latents'/ARM/x['key']) for x in items):start('render','render')
   while not (WORK/'STOP').exists():
    for key,proc in list(jobs.items()):
     if proc.poll() is None:continue
     del jobs[key]
     if proc.returncode:
      if key=='cache':
       failure=WORK/'cache_failure.json'
       message=read(failure).get('traceback','') if failure.exists() else ''
       if 'out of memory' in message.lower():
        policy=read(WORK/'RESOURCE_POLICY.json')
        policy.update(text_batch=max(1,policy['text_batch']//2),at=time.time(),reason='Adaptive Gemma batching after actual native encoder OOM; completed cache preserved')
        save(WORK/'RESOURCE_POLICY.json',policy)
      if attempts[key]>=3:raise RuntimeError(f'{key} failed three times; inspect processes/logs')
      stage=key.split(':')[0];slot=int(key.split(':')[1]) if ':' in key else 0
      start(key,stage,slot)
    generated={x['key'] for x in items if (WORK/'generated'/ARM/x['key']/'complete.json').exists()}
    latents=sum((WORK/'latents'/ARM/x['key']/'complete.json').exists() for x in items)
    cache_done=(WORK/'CACHE_COMPLETE.json').exists();g=gpu()
    # Bounded overlap: keep cache/generation peaks isolated from scoring/decoders.
    if cache_done and len(generated)<6624 and latents+len(p['reused_generated_ids'])>len(generated):
     for slot in range(2):
      key='decode:'+str(slot)
      if key not in jobs and g['free_gib']>=40 and sum(k.startswith('decode:') for k in jobs)<(1 if 'render' in jobs else 2):start(key,'decode',slot);g=gpu()
    for i in range(p['workers']):
     marker=WORK/'progress/metrics'/f'{ARM}_{i:02d}.json'
     if marker.exists() and read(marker).get('status')=='complete':metric_done.add(i)
    score_cap=1 if 'render' in jobs else 4
    if cache_done:
     for i in range(p['workers']):
      if sum(k.startswith('metric:') for k in jobs)>=score_cap or g['free_gib']<28:break
      if i in metric_done or 'metric:'+str(i) in jobs:continue
      if all(x['key'] in generated for x in read(WORK/'shards'/f'{i:02d}.json')):
       if i not in started:publish(i);started.add(i)
       start('metric:'+str(i),'metric',i);g=gpu()
    save(WORK/'status.json',dict(stage='cache_generation_decode_scoring',at=time.time(),step=6000,total=6624,generated=len(generated),new_latents=latents,reused_generated=196,cache_complete=cache_done,scored_shards=len(metric_done),total_shards=p['workers'],active=list(jobs),gpu=g))
    if len(generated)==6624 and len(metric_done)==p['workers'] and not jobs:break
    time.sleep(20)
   if (WORK/'STOP').exists():return
   proc=launch('merge');assert proc.wait()==0,'Global pooled metric merge failed'
   result=read(WORK/'RESULTS.json')[ARM+'/seamless'];assert result['requested_pairs']==result['audio']['requested_pairs']==6624
   save(WORK/'FINAL_AUDIT.json',dict(passed=True,at=time.time(),count=6624,checkpoint_sha256=p['arms'][0]['original_sha256'],source_version=p['source_version'],metric_coverage=result['coverage']))
   save(WORK/'COMPLETE.json',dict(passed=True,at=time.time(),count=6624,status=result['status'],results_sha256=sha(WORK/'RESULTS.json')))
   save(WORK/'status.json',dict(stage='complete',at=time.time(),count=6624))
  except BaseException:
   failure=dict(at=time.time(),traceback=traceback.format_exc())
   save(WORK/'FAILURE.json',failure);save(WORK/'status.json',dict(stage='failed',**failure));raise
  finally:
   for proc in jobs.values():
    if proc.poll() is None:
     try:os.killpg(proc.pid,signal.SIGTERM)
     except ProcessLookupError:pass


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','controller','worker']);a.add_argument('--stage',choices=['inputs','cache','render','decode']);a.add_argument('--slot',type=int,default=0);args=a.parse_args()
 if args.action=='prepare':prepare()
 elif args.action=='controller':controller()
 else:
  try:worker(args.stage,args.slot)
  except BaseException:
   save(WORK/(args.stage+'_failure.json'),dict(at=time.time(),traceback=traceback.format_exc()));raise
