"""Four nested-test256 combinations, two-node16GPU native inference and scoring."""
import argparse,collections,copy,math,fcntl,importlib.util,json,os,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
SELF=Path(__file__).resolve();WORK=SELF.parent;ROOT=WORK.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
PARENT=ROOT/'experiments/p2_ltx15_test64_sft2752_20261113'
spec=importlib.util.spec_from_file_location('verified_ltx_screening',PARENT/'run.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
read,save,sha=base.read,base.save,base.sha;PYTHON=base.PYTHON;METRIC_PYTHON=base.METRIC_PYTHON
STEPS=[6000,6800];QWENS=[2752,8256,7500,0]
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
 if q==2752:return read(base.child(6000)/'protocol.json')['planner_adapter_identity']
 from p2_j_sft_hidden import adapter_identity
 return adapter_identity(ROOT/f'experiments/p2_planning_sft_continue2epochs20_20261113/training/checkpoint-{q}')

def prepare():
 from prepare_impl import prepare as build
 build(sys.modules[__name__])
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
  scores={i for i in range(10) if (p:=w/'progress/metrics'/f'{name}_{i:02d}.json').exists() and read(p).get('status')=='complete'}
  state[x['key']]=dict(done=done,latent=latent,remaining=256-len(latent),scores=scores,merged=(w/'COMPLETE.json').exists(),reused=x['reused'],cache_ready=x['reused'] or all((cache_dir(x['qwen'])/f'CACHE_COMPLETE_{i}.json').exists() for i in range(4)))
 return state

def results():
 rows={}
 for x in plans():
  f=Path(x['work'])/'RESULTS.json'
  if f.exists():
   if x['reused']:assert sha(f)==x['results_sha256']
   rows[x['key']]=dict(qwen=x['qwen'],ltx=x['ltx'],reused=x['reused'],scores=next(iter(read(f).values())))
 save(WORK/'RESULTS.json',dict(at=time.time(),completed_combinations=len(rows),total_combinations=4,results=rows))
 cols=['psnr','ssim','lpips','fid','fvd','emo_acc','delta_resp'];lines=['# Qwen×LTX固定test256联合测评','',f'已汇总{len(rows)}/4组；每组256条，seed137；测试调参范围。','', '| Qwen | LTX | '+' | '.join(cols)+' | 漏回应率 | 静默误声率 |','|---|---|'+'---:|'*9]
 for k,v in rows.items():
  r=v['scores'];ac=r['audio']['activity']['acoustic'];values=[r['metrics'].get(c) for c in cols]+[ac.get('missed_response_rate'),ac.get('false_sound_clip_rate')];lines.append(f"| {v['qwen'] or 'base'} | {v['ltx']} | "+' | '.join('N/A' if x is None else f'{x:.4f}' for x in values)+' |')
 lines+=['','沿用原指标有效覆盖与N/A规则，部分表情／反应测量可能缺失；完整覆盖信息见RESULTS.json。未设置任意加权综合分，不据256条宣称完整test最优。'];(WORK/'RESULTS.md').write_text('\n'.join(lines)+'\n');return rows

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
     unfinished=sum(not v['reused'] and v['remaining']>0 for v in state.values())
     cap=max(4,math.ceil(14/max(1,unfinished)))
     candidates=[k for k,v in state.items() if not v['reused'] and v['cache_ready'] and v['remaining']>counts[k] and counts[k]<cap] if sum(counts.values())<14 else []
     chosen=max(candidates,key=lambda k:(-counts[k],state[k]['remaining'])) if candidates else None
     score=None
     for k,v in state.items():
      if v['reused'] or v['merged']:continue
      for i in range(10):
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
     if not v['reused'] and not v['merged'] and len(v['scores'])==10 and key not in merges:
      env=dict(os.environ,**environment('merge',0));env['CUDA_VISIBLE_DEVICES']='';log=WORK/'logs'/f'merge_{key}.log'
      with log.open('a') as f:merges[key]=subprocess.Popen([str(METRIC_PYTHON),str(ROOT/'scripts/p2_testsets_eval.py'),'merge','--work',str(child(key))],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    finished=all(v['merged'] for v in state.values()) and not active and not merges
    if time.time()>=next_update or finished:
     info=dict(at=time.time(),stage='complete_with_coverage_report' if finished else 'cache_render_decode_score',generated=sum(len(v['done']) for v in state.values()),latent=sum(len(v['latent']) for v in state.values()),total=1024,reused=256,new=768,completed_combinations=[k for k,v in state.items() if v['merged']],active_gpus=len(used),max_gpus=16,active=[{k:v for k,v in j.items() if k!='process'} for j in active.values()]);save(WORK/'status.json',info);summaries=results();next_update=time.time()+15
    if finished:
     assert len(summaries)==4
     for v in summaries.values():
      r=v['scores'];assert all(r['coverage'][k]['pairs']==256 for k in ['psnr','ssim','lpips','fid','fvd']) and r['audio']['valid_pairs']==256 and r['audio']['pairs_with_errors']==0
     save(WORK/'COMPLETE.json',dict(success=True,at=time.time(),outputs=1024,combinations=4,physical_gpus=16,nodes=['node12','node8'],metrics_coverage='See per-combination coverage; partial emotion/deltaResp possible'));return
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
