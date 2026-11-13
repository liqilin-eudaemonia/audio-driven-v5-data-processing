"""16 fixed-test64 combinations, two-node16GPU native inference and scoring."""
import argparse,collections,copy,fcntl,importlib.util,json,os,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
SELF=Path(__file__).resolve();WORK=SELF.parent;ROOT=WORK.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
PARENT=ROOT/'experiments/p2_ltx15_test64_sft2752_20261113'
spec=importlib.util.spec_from_file_location('verified_ltx_screening',PARENT/'run.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
read,save,sha=base.read,base.save,base.sha;PYTHON=base.PYTHON;METRIC_PYTHON=base.METRIC_PYTHON
STEPS=[6000,6300,6500,6700,6800];QWENS=[0,7500,8256]
GUARD=ROOT.parent/'audio-driven-v4-p3-autoselect-20261013/scripts/pipeline3_cluster.py'
def config():return read(WORK/'cluster.json')
def slots(c=None):
 c=c or config();assert {n['name']:n['gpus'] for n in c['nodes']}=={'node12':list(range(8)),'node8':list(range(8))} and len(c['nodes'])==2
 return [dict(rank=i,node=n['name'],host=n['host'],gpu=g) for i,(n,g) in enumerate((n,g) for n in c['nodes'] for g in n['gpus'])]
def plans():return read(WORK/'PLAN.json')['combinations']
def child(key):return WORK/'combinations'/key
def cache_dir(q):return WORK/'conditioning'/f'qwen{q}'
def stopped():return (WORK/'STOP').exists()
def link(src,dst):base.link(src,dst)
def adapter(q):
 if q==0:return None
 from p2_j_sft_hidden import adapter_identity
 return adapter_identity(ROOT/f'experiments/p2_planning_sft_continue2epochs20_20261113/training/checkpoint-{q}')

def prepare():
 if (WORK/'PREPARED.json').exists():verify();return
 from p2_testsets_eval.runtime import ready
 items=read(PARENT/'shared_items.json');assert len(items)==64
 assert read(PARENT/'COMPLETE.json')['success']
 c=read(PARENT/'cluster29/cluster.json');c['nodes']=[n for n in c['nodes'] if n['name'] in ['node12','node8']];c['training_script']=str(SELF);c['purpose']='Joint Qwen/LTX test64 screening; no node11';slots(c);save(WORK/'cluster.json',c)
 link(PARENT/'shared_items.json',WORK/'shared_items.json');link(PARENT/'DATA_CONFIG.json',WORK/'DATA_CONFIG.json')
 normal=read(PARENT/'PLAN.json');protocol=read(base.child(6800)/'protocol.json');combinations=[]
 for step in STEPS:
  w=base.child(step);r=next(iter(read(w/'RESULTS.json').values()));assert all(r['coverage'][k]['pairs']==64 for k in ['psnr','ssim','lpips','fid','fvd']) and r['audio']['valid_pairs']==64
  combinations.append(dict(key=f'qwen2752_ltx{step}',qwen=2752,ltx=step,work=str(w),reused=True,results_sha256=sha(w/'RESULTS.json')))
 for q in QWENS:
  d=cache_dir(q);d.mkdir(parents=True,exist_ok=True);p=copy.deepcopy(protocol)
  p.update(planner='Qwen2.5-Omni-3B_base' if q==0 else f'SFT_Qwen2.5-Omni-3B_upright{q}',planner_adapter=None if q==0 else adapter(q)['path'],planner_adapter_identity=adapter(q),sft_hidden_cache=f'inference-only-adapter:{q}',conditioning='Own Qwen text+own adapted hidden+own Gemma; same source audio and first frame',cluster=str(WORK/'cluster.json'),metrics_cluster=str(WORK/'cluster.json'),inference_nodes=['node12','node8'],authorized_scope='node12/node8 GPU0-7 only',physical_gpus=list(range(8)))
  link(PARENT/'DATA_CONFIG.json',d/'DATA_CONFIG.json');link(base.child(6800)/'negative',d/'negative');save(d/'protocol.json',p)
  for shard in range(4):save(d/'shards'/f'{shard}.json',items[16*shard:16*(shard+1)])
  for step in ([6800] if q==0 else STEPS):
   key=f'qwen{q}_ltx{step}';w=child(key);w.mkdir(parents=True,exist_ok=True);cp=copy.deepcopy(p);cp['arms']=copy.deepcopy(read(base.child(step)/'protocol.json')['arms']);cp['arms'][0].update(name=key,planner=p['planner'])
   cp.update(schema='joint-qwen-ltx-test64-v1',shared_items_manifest=str(WORK/'shared_items.json'),shared_items_sha256=sha(WORK/'shared_items.json'),execution_order='Own planner cache -> shared native render -> decode ->4 pooled score shards',metric_sampling='Same64 samples; native formulas; pooled features across shards',shared_item_queue=True,workers=4,total_samples=64,total_outputs=64)
   for name,source in [('conditions',d/'conditions'),('negative',d/'negative'),('whisper',base.child(6800)/'whisper'),('shared_items.json',WORK/'shared_items.json'),('DATA_CONFIG.json',WORK/'DATA_CONFIG.json')]:
    if name=='conditions':source.mkdir(exist_ok=True)
    link(source,w/name)
   save(w/'protocol.json',cp)
   for shard in range(4):save(w/'shards'/f'{shard:02d}.json',read(base.child(step)/'shards'/f'{shard:02d}.json'))
   combinations.append(dict(key=key,qwen=q,ltx=step,work=str(w),reused=False))
 for rec in read(PARENT/'CHECKPOINTS.json'):
  if rec['step'] in STEPS:assert rec['stamp']==base.stamp(rec['path']) and read(PARENT/'integrity'/f'{rec["step"]:06d}.json')['sha256']==rec['sha256']
 assert all(sha(p)==h for p,h in normal['identity']['source'].items())
 save(WORK/'PLAN.json',dict(at=time.time(),combinations=combinations,total_combinations=16,samples=64,total_outputs=1024,reused_outputs=320,new_outputs=704,own_qwen_caches=192,seed=137,generation=protocol['generation'],original_plan_sha256=sha(PARENT/'PLAN.json'),normalized_config=normal['normalized_config'],identity=normal['identity'],scope='test tuning64, fixedseed; no independent final-test claim',adapter_binding='Inference-only source/config binding to native PEFT load_session; no training-H cache fabricated'))
 sources=[SELF,PARENT/'run.py',ROOT/'src/p2_testsets_eval/runtime.py',ROOT/'src/p2_all_lora_followups_eval_20261109.py',ROOT/'src/p2_j_sft_hidden.py',ROOT/'src/p2_plan_cache/encoding.py',GUARD]
 frozen=[WORK/'cluster.json',WORK/'PLAN.json',WORK/'shared_items.json',WORK/'DATA_CONFIG.json']+[cache_dir(q)/'protocol.json' for q in QWENS]+[Path(x['work'])/'protocol.json' for x in combinations]
 save(WORK/'PREPARED.json',dict(success=True,at=time.time(),files={str(p):sha(p) for p in sources+frozen},physical_gpus=16,excluded_nodes=['node11','node4']))
 print('Prepared16 combinations:320 reuse+704 new,192 own planner conditions')
def verify():
 p=read(WORK/'PREPARED.json');assert all(sha(k)==v for k,v in p['files'].items());slots();return p

PROBE=r'''
import json,subprocess
lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpus={}
for line in lines.splitlines():
 a=[x.strip() for x in line.split(',')];gpus[a[0]]=dict(uuid=a[1],used=float(a[2]),util=float(a[3]))
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
print(json.dumps(dict(gpus=gpus,apps=[s.split(',')[0].strip() for s in apps.splitlines() if ',' in s],ips=subprocess.check_output(['hostname','-I'],text=True).split())))
'''
def probe(n):
 from listen_cluster import transport
 r=subprocess.run(transport(config(),n,[str(PYTHON),'-c',PROBE]),capture_output=True,text=True,timeout=25)
 if r.returncode:raise ConnectionError(r.stderr[-300:])
 x=json.loads(r.stdout);assert n['host'] in x['ips'];return x

def device(rank):
 s=slots()[rank];assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(s['gpu'])
 assert s['host'] in subprocess.check_output(['hostname','-I'],text=True).split()
 uuid=subprocess.check_output(['nvidia-smi','-i',str(s['gpu']),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==read(WORK/'ACTIVATED.json')['uuids'][str(rank)];return s

def worker(stage,rank,key='',q=0,shard=0):
 verify();s=device(rank)
 from p2_whisper_av.bootstrap import activate_ltx
 activate_ltx()
 import torch
 torch.set_num_threads(4);torch.cuda.set_device(0);cap=96 if stage=='render' else (80 if stage=='cache' else 38)
 torch.cuda.set_per_process_memory_fraction(cap*2**30/torch.cuda.get_device_properties(0).total_memory)
 from p2_testsets_eval import runtime as rt
 from p2_listen.config import extractor_config
 d=read(WORK/'DATA_CONFIG.json');cfg=extractor_config(d);items=read(WORK/'shared_items.json');progress=rt.progress
 record=dict(at=time.time(),pid=os.getpid(),hostname=socket.gethostname(),**s,stage=stage,key=key,qwen=q,shard=shard);save(WORK/'devices'/f'{stage}_{rank}_{key}_{q}_{shard}.json',record)
 with ExitStack() as stack:
  stack.enter_context(patch.object(rt,'data_config',lambda:(d,cfg)))
  stack.enter_context(patch.object(rt,'progress',lambda w,st,unused,**kw:progress(w,st,rank,**kw)))
  if stage=='cache':
   cd=cache_dir(q);p=read(cd/'protocol.json');part=read(cd/'shards'/f'{shard}.json')
   # Reuse native session and extraction; bind explicitly to this inference adapter.
   if q:
    import p2_j_sft_hidden as hidden
    native_load=hidden.load_session
    def reader(path):
     assert path==f'inference-only-adapter:{q}' and adapter(q)==p['planner_adapter_identity']
     return SimpleNamespace(source=SimpleNamespace(manifest={'config':d}),protocol={'adapter':p['planner_adapter_identity']})
    def load(r):
     session=native_load(r)
     from peft import get_peft_model_state_dict
     from safetensors.torch import load_file
     actual=get_peft_model_state_dict(session[2]);expected=load_file(str(Path(r.protocol['adapter']['path'])/'adapter_model.safetensors'))
     assert actual.keys()==expected.keys() and len(actual)==504 and all(torch.equal(actual[k].cpu(),expected[k]) for k in actual)
     save(WORK/'planner_acceptance'/f'{q}_{shard}.json',dict(**record,adapter=r.protocol['adapter'],all_504_tensors_equal=True,native_session=True));return session
    stack.enter_context(patch.object(hidden,'Reader',reader));stack.enter_context(patch.object(hidden,'load_session',load))
   from cache_recorded_truncation import cache as recorded_cache
   recorded_cache(cd,part,0,p)
   for x in part:
    folder=cd/'conditions'/x['key'];old=base.child(6800)/'conditions'/x['key']
    assert sha(folder/'input_audit.json')==sha(old/'input_audit.json') and sha(folder/'first_frame.png')==sha(old/'first_frame.png')
    assert rt.ready(folder/'qwen') and rt.ready(folder/'text') and read(folder/'qwen/plan.json')['adapter']==p['planner_adapter_identity']
   save(cd/f'CACHE_COMPLETE_{shard}.json',dict(success=True,at=time.time(),items=[x['key'] for x in part],adapter=p['planner_adapter_identity'],source_audit_exact=True))
  elif stage=='render':
   combo=next(x for x in plans() if x['key']==key);step=combo['ltx'];w=child(key);p=read(w/'protocol.json')
   assert all((cache_dir(combo['qwen'])/f'CACHE_COMPLETE_{i}.json').exists() for i in range(4))
   import p2_all_lora_followups_eval_20261109 as native
   import p2_ltx_events.inference as old
   from p2_ltx_all_lora import inference,checkpoint,config as conf
   plan=read(WORK/'PLAN.json');identity=plan['identity'];checked_validate=conf.validate
   def load(path,device='cuda:0'):
    assert Path(path)==Path(p['arms'][0]['checkpoint']);v=base.pinned(step)
    def normalize(c):assert c==identity['experiment'];return checked_validate(copy.deepcopy(plan['normalized_config']))
    with patch.object(checkpoint,'checked',lambda _:v),patch.object(conf,'source_identity',lambda:identity['source']),patch.object(conf,'validate',normalize):result=inference.load_model(path,device)
    assert len(result[0].text_targets)==1632 and list(result[0].state_dict())==list(v['model']);save(WORK/'model_acceptance'/f'{key}_{rank}.json',dict(**record,strict_tensors=4809,targets=1632,original_sha256=p['arms'][0]['original_sha256']));return result
   native.STEP=step;native.ARM=key;native.WORK=w
   with patch.object(old,'load_model',load):native.render(w,items,0,p)
  elif stage=='decode':
   with rt.cached_loaders(['load_video_vae_decoder','load_audio_vae_decoder','load_vocoder']):
    for combo in plans():
     if stopped():break
     if combo['reused']:continue
     w=Path(combo['work']);p=read(w/'protocol.json');name=combo['key']
     if any(rt.ready(w/'latents'/name/x['key']) and not rt.ready(w/'generated'/name/x['key']) for x in items):rt.decode(w,items,rank,p,p['arms'])
 save(WORK/'worker_complete'/f'{stage}_{rank}_{key}_{q}_{shard}.json',dict(**record,finished=time.time()))

def environment(stage,gpu):
 return dict(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONPATH='/private/liqi_workspace/.cache/paired_video_metrics/python' if stage in ('metric','merge') else '')
def launch(stage,rank,key='',q=0,shard=0,attempt=1):
 from listen_cluster import transport
 c=config();s=slots()[rank];n=next(n for n in c['nodes'] if n['name']==s['node'])
 if stage=='metric':cmd=[str(METRIC_PYTHON),'-u',str(ROOT/'scripts/p2_pipelined_testsets_eval.py'),'metrics','--work',str(child(key)),'--rank',str(shard),'--arms',key]
 else:cmd=[str(PYTHON),'-u',str(SELF),'worker','--stage',stage,'--rank',str(rank),'--key',key,'--qwen',str(q),'--shard',str(shard)]
 command=['env',*[k+'='+v for k,v in environment(stage,s['gpu']).items()],*cmd];guard=[str(PYTHON),str(GUARD),'--guard',*command]
 log=WORK/'logs'/f'{stage}_{rank}_{key}_{q}_{shard}_{time.time_ns()}.log';log.parent.mkdir(exist_ok=True)
 with log.open('ab',buffering=0) as f:p=subprocess.Popen(transport(c,n,guard),stdin=subprocess.PIPE,stdout=f,stderr=subprocess.STDOUT)
 rec=dict(at=time.time(),pid=p.pid,stage=stage,rank=rank,key=key,qwen=q,shard=shard,attempt=attempt,log=str(log),node=s['node'],gpu=s['gpu']);save(WORK/'processes'/f'{stage}_{rank}_{key}_{q}_{shard}.json',rec);return dict(process=p,**rec)
def publish(key,shard):
 combo=next(x for x in plans() if x['key']==key)
 with patch.object(base,'child',lambda _:child(key)):base.publish(combo['ltx'],shard)
def snapshot():
 from p2_testsets_eval.runtime import ready
 items=read(WORK/'shared_items.json');state={}
 for x in plans():
  w=Path(x['work']);name=read(w/'protocol.json')['arms'][0]['name'];done={i['key'] for i in items if ready(w/'generated'/name/i['key'])};latent={i['key'] for i in items if ready(w/'latents'/name/i['key'])}|done
  scores={i for i in range(4) if (p:=w/'progress/metrics'/f'{name}_{i:02d}.json').exists() and read(p).get('status')=='complete'}
  state[x['key']]=dict(done=done,latent=latent,remaining=64-len(latent),scores=scores,merged=(w/'COMPLETE.json').exists(),reused=x['reused'],cache_ready=x['reused'] or all((cache_dir(x['qwen'])/f'CACHE_COMPLETE_{i}.json').exists() for i in range(4)))
 return state

def results():
 rows={}
 for x in plans():
  f=Path(x['work'])/'RESULTS.json'
  if f.exists():
   if x['reused']:assert sha(f)==x['results_sha256']
   rows[x['key']]=dict(qwen=x['qwen'],ltx=x['ltx'],reused=x['reused'],scores=next(iter(read(f).values())))
 save(WORK/'RESULTS.json',dict(at=time.time(),completed_combinations=len(rows),total_combinations=16,results=rows))
 cols=['psnr','ssim','lpips','fid','fvd','emo_acc','delta_resp'];lines=['# Qwen×LTX固定test64联合测评','',f'已汇总{len(rows)}/16组；每组64条，seed137；测试调参范围。','', '| Qwen | LTX | '+' | '.join(cols)+' | 漏回应率 | 静默误声率 |','|---|---|'+'---:|'*9]
 for k,v in rows.items():
  r=v['scores'];ac=r['audio']['activity']['acoustic'];values=[r['metrics'].get(c) for c in cols]+[ac.get('missed_response_rate'),ac.get('false_sound_clip_rate')];lines.append(f"| {v['qwen'] or 'base'} | {v['ltx']} | "+' | '.join('N/A' if x is None else f'{x:.4f}' for x in values)+' |')
 lines+=['','沿用原指标有效覆盖与N/A规则，部分表情／反应测量可能缺失；完整覆盖信息见RESULTS.json。未设置任意加权综合分，不据64条宣称完整test最优。'];(WORK/'RESULTS.md').write_text('\n'.join(lines)+'\n');return rows

def queue():
 from p2_progressive_eval_20261108 import ResourcePoller
 verify();assert '172.29.252.37' in subprocess.check_output(['hostname','-I'],text=True).split();assert (WORK/'ACTIVATED.json').exists()
 signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
 with (WORK/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save(WORK/'CONTROLLER.json',dict(at=time.time(),pid=os.getpid(),host=socket.gethostname()))
  poller=ResourcePoller(config(),probe,interval=10);active={};attempts=collections.Counter();failures=collections.Counter();merges={};published=set();next_update=0
  def start(stage,rank,key='',q=0,shard=0):
   task=(stage,key,q,shard) if stage in ['cache','metric'] else (stage,rank,key)
   attempts[task]+=1;active[task]=launch(stage,rank,key,q,shard,attempts[task])
  try:
   while not stopped():
    for task,j in list(active.items()):
     rc=j['process'].poll()
     if rc is None:continue
     if j['process'].stdin:j['process'].stdin.close()
     del active[task]
     if rc:
      failures[task]+=1
      if failures[task]>=(6 if rc==255 else 3):raise RuntimeError('Repeated worker error:'+j['log'])
    for key,p in list(merges.items()):
     if p.poll() is None:continue
     del merges[key]
     if p.returncode:raise RuntimeError('Merge failed:'+key)
    state=snapshot();reports=poller.snapshot();used={j['rank'] for j in active.values()};started=collections.Counter()
    for s in slots():
     rank=s['rank'];rep=reports.get(s['node'],{})
     if rank in used or started[s['node']]>=2 or 'gpus' not in rep:continue
     g=rep['gpus'][str(s['gpu'])]
     if g['used']>2048 or g['uuid'] in rep['apps']:continue
     pending_cache=[(q,i) for q in QWENS for i in range(4) if not (cache_dir(q)/f'CACHE_COMPLETE_{i}.json').exists() and ('cache','',q,i) not in active]
     counts=collections.Counter(j['key'] for j in active.values() if j['stage']=='render')
     candidates=[k for k,v in state.items() if not v['reused'] and v['cache_ready'] and v['remaining']>counts[k] and counts[k]<2]
     chosen=max(candidates,key=lambda k:(-counts[k],state[k]['remaining'])) if candidates else None
     score=None
     for k,v in state.items():
      if v['reused'] or v['merged']:continue
      for i in range(4):
       if i not in v['scores'] and ('metric',k,0,i) not in active and all(x['key'] in v['done'] for x in read(child(k)/'shards'/f'{i:02d}.json')):score=(k,i);break
      if score:break
     backlog=sum(len(v['latent']-v['done']) for v in state.values());decoders=sum(j['stage']=='decode' for j in active.values());scorers=sum(j['stage']=='metric' for j in active.values())
     if pending_cache:q,i=pending_cache[0];start('cache',rank,q=q,shard=i)
     elif score and (scorers<2 or chosen is None):
      k,i=score
      if score not in published:publish(k,i);published.add(score)
      start('metric',rank,k,shard=i)
     elif backlog>max(0,decoders*8) and (decoders<2 or chosen is None):start('decode',rank)
     elif chosen:start('render',rank,chosen)
     elif backlog>decoders:start('decode',rank)
     else:continue
     used.add(rank);started[s['node']]+=1
    for key,v in state.items():
     if not v['reused'] and not v['merged'] and len(v['scores'])==4 and key not in merges:
      env=dict(os.environ,**environment('merge',0));env['CUDA_VISIBLE_DEVICES']='';log=WORK/'logs'/f'merge_{key}.log'
      with log.open('a') as f:merges[key]=subprocess.Popen([str(METRIC_PYTHON),str(ROOT/'scripts/p2_testsets_eval.py'),'merge','--work',str(child(key))],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    finished=all(v['merged'] for v in state.values()) and not active and not merges
    if time.time()>=next_update or finished:
     info=dict(at=time.time(),stage='complete_with_coverage_report' if finished else 'cache_render_decode_score',generated=sum(len(v['done']) for v in state.values()),latent=sum(len(v['latent']) for v in state.values()),total=1024,reused=320,new=704,completed_combinations=[k for k,v in state.items() if v['merged']],active_gpus=len(used),max_gpus=16,active=[{k:v for k,v in j.items() if k!='process'} for j in active.values()]);save(WORK/'status.json',info);summaries=results();next_update=time.time()+15
    if finished:
     assert len(summaries)==16
     for v in summaries.values():
      r=v['scores'];assert all(r['coverage'][k]['pairs']==64 for k in ['psnr','ssim','lpips','fid','fvd']) and r['audio']['valid_pairs']==64 and r['audio']['pairs_with_errors']==0
     save(WORK/'COMPLETE.json',dict(success=True,at=time.time(),outputs=1024,combinations=16,physical_gpus=16,nodes=['node12','node8'],metrics_coverage='See per-combination coverage; partial emotion/deltaResp possible'));return
    time.sleep(3)
  finally:
   poller.close()
   for j in active.values():
    if j['process'].stdin:j['process'].stdin.close()
   for p in merges.values():
    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)

def activate():
 from concurrent.futures import ThreadPoolExecutor
 verify();reports={}
 with ThreadPoolExecutor(max_workers=2) as pool:
  jobs=[(n,pool.submit(probe,n)) for n in config()['nodes']]
  for n,f in jobs:reports[n['name']]=f.result()
 for s in slots():
  r=reports[s['node']];g=r['gpus'][str(s['gpu'])];assert g['used']<2048 and g['uuid'] not in r['apps'],(s,g)
 save(WORK/'ACTIVATED.json',dict(at=time.time(),uuids={str(s['rank']):reports[s['node']]['gpus'][str(s['gpu'])]['uuid'] for s in slots()},nodes=['node12','node8'],gpus=16));queue()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','activate','queue','worker','report']);p.add_argument('--stage',choices=['cache','render','decode']);p.add_argument('--rank',type=int,default=0);p.add_argument('--key',default='');p.add_argument('--qwen',type=int,default=0);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
 try:
  if a.action=='prepare':prepare()
  elif a.action=='activate':activate()
  elif a.action=='queue':queue()
  elif a.action=='worker':worker(a.stage,a.rank,a.key,a.qwen,a.shard)
  else:results()
 except BaseException:
  save(WORK/f'FAILURE_{a.action}_{a.stage}_{a.rank}_{a.key}_{a.qwen}_{a.shard}.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
