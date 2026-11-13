"""Six majority-inferior test64 candidates on matched test256; two nodes16 GPUs."""
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
STEPS=[5500,6000,6500];QWENS=[7500]
GUARD=ROOT.parent/'audio-driven-v4-p3-autoselect-20261013/scripts/pipeline3_cluster.py'
def config():return read(WORK/'cluster.json')
def slots(c=None):
 c=c or config();assert {n['name']:n['gpus'] for n in c['nodes']}=={'node12':list(range(8)),'node11':list(range(8))} and len(c['nodes'])==2
 return [dict(rank=i,node=n['name'],host=n['host'],gpu=g) for i,(n,g) in enumerate((n,g) for n in c['nodes'] for g in n['gpus'])]
def plans():return read(WORK/'PLAN.json')['combinations']
def child(key):return WORK/'combinations'/key
def cache_dir(q):
 assert q==7500
 return ROOT/'experiments/p2_joint_qwen_ltx_test256_16gpu_20261113/conditioning/qwen7500'
def stopped():return (WORK/'STOP').exists()
def link(src,dst):base.link(src,dst)
def adapter(q):
 if q==0:return None
 from p2_j_sft_hidden import adapter_identity
 return adapter_identity(ROOT/f'experiments/p2_planning_sft_continue2epochs20_20261113/training/checkpoint-{q}')

def prepare():
 from preparation import prepare as impl
 impl(sys.modules[__name__])

def verify():
 p=read(WORK/'PREPARED.json');assert all(sha(k)==v for k,v in p['files'].items());slots();return p

PROBE=r'''
import json,subprocess
lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpus={}
for line in lines.splitlines():
 a=[x.strip() for x in line.split(',')];gpus[a[0]]=dict(uuid=a[1],used=float(a[2]),free=float(a[3]),util=float(a[4]))
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
  if stage=='cache':raise RuntimeError('This run reuses frozen Qwen7500 conditions; never rewrite upstream caches')
  elif stage=='render':
   combo=next(x for x in plans() if x['key']==key);step=combo['ltx'];w=child(key);p=read(w/'protocol.json')
   assert all((cache_dir(combo['qwen'])/f'CACHE_COMPLETE_{i}.json').exists() for i in range(4))
   from model_loading import load_native
   import render as native
   native.STEP=step;native.ARM=key;native.WORK=w
   native.load_model=lambda path,device='cuda:0':load_native(combo,record,device)
   native.render(w,items,0,p)
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
   rows[x['key']]=dict(arm=x['component'],qwen=x['qwen'],ltx=x['ltx'],reused=x['reused'],scores=next(iter(read(f).values())))
 save(WORK/'RESULTS.json',dict(at=time.time(),completed_combinations=len(rows),total_combinations=6,results=rows))
 cols=['psnr','ssim','lpips','fid','fvd','emo_acc','delta_resp'];lines=['# 三组全LoRA消融×2权重：Qwen7500固定test256','',f'已汇总{len(rows)}/6组；每组256条，seed137；测试调参范围。','', '| Pipeline | Qwen | LTX | '+' | '.join(cols)+' | 漏回应率 | 静默误声率 |','|---|---|---|'+'---:|'*9]
 for k,v in rows.items():
  r=v['scores'];ac=r['audio']['activity']['acoustic'];values=[r['metrics'].get(c) for c in cols]+[ac.get('missed_response_rate'),ac.get('false_sound_clip_rate')];lines.append(f"| {v['arm']} | {v['qwen'] or 'base'} | {v['ltx']} | "+' | '.join('N/A' if x is None else f'{x:.4f}' for x in values)+' |')
 lines+=['','沿用原指标有效覆盖与N/A规则，部分表情／反应测量可能缺失；完整覆盖信息见RESULTS.json。未设置任意加权综合分，不据256条宣称完整test最优。'];(WORK/'RESULTS.md').write_text('\n'.join(lines)+'\n');return rows

def per_checkpoint_limit(state):
 pending=sum(v['remaining']>0 and not v['merged'] for v in state.values())
 return max(2,12//max(1,pending))

def queue():
 from p2_progressive_eval_20261108 import ResourcePoller
 verify();assert '172.29.252.37' in subprocess.check_output(['hostname','-I'],text=True).split();assert (WORK/'ACTIVATED.json').exists()
 signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
 with (WORK/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save(WORK/'CONTROLLER.json',dict(at=time.time(),pid=os.getpid(),host=socket.gethostname(),entry=str(Path(__file__).resolve()),scheduler='dynamic-per-checkpoint-share-v2'))
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
    if (WORK/'INTEGRITY_FAILED.json').exists():raise RuntimeError('Checkpoint integrity failed')
    state=snapshot();reports=poller.snapshot();used={j['rank'] for j in active.values()};started=collections.Counter()
    for s in sorted(slots(),key=lambda s:({'node12':0,'node11':1,'node8':2}[s['node']],s['gpu'])):
     rank=s['rank'];rep=reports.get(s['node'],{})
     if rank in used or started[s['node']]>=2 or 'gpus' not in rep:continue
     g=rep['gpus'][str(s['gpu'])]
     free=g['free']/1024
     if free<44:continue
     pending_cache=[(q,i) for q in QWENS for i in range(4) if not (cache_dir(q)/f'CACHE_COMPLETE_{i}.json').exists() and ('cache','',q,i) not in active]
     counts=collections.Counter(j['key'] for j in active.values() if j['stage']=='render')
     candidates=[k for k,v in state.items() if not v['reused'] and v['cache_ready'] and v['remaining']>counts[k] and counts[k]<per_checkpoint_limit(state) and free>=102 and (WORK/'integrity'/f'{k}.json').exists()]
     chosen=max(candidates,key=lambda k:(-counts[k],state[k]['remaining'])) if candidates else None
     score=None
     for k,v in state.items():
      if v['reused'] or v['merged']:continue
      for i in range(10):
       if i not in v['scores'] and ('metric',k,0,i) not in active and all(x['key'] in v['done'] for x in read(child(k)/'shards'/f'{i:02d}.json')):score=(k,i);break
      if score:break
     backlog=sum(len(v['latent']-v['done']) for v in state.values());decoders=sum(j['stage']=='decode' for j in active.values());scorers=sum(j['stage']=='metric' for j in active.values())
     if pending_cache:q,i=pending_cache[0];start('cache',rank,q=q,shard=i)
     elif score and (scorers<4 or chosen is None):
      k,i=score
      if score not in published:publish(k,i);published.add(score)
      start('metric',rank,k,shard=i)
     elif backlog>max(0,decoders*8) and (decoders<4 or chosen is None):start('decode',rank)
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
     info=dict(at=time.time(),stage='complete_with_coverage_report' if finished else 'cache_render_decode_score',generated=sum(len(v['done']) for v in state.values()),latent=sum(len(v['latent']) for v in state.values()),total=1536,reused=384,new=1152,completed_combinations=[k for k,v in state.items() if v['merged']],active_gpus=len(used),max_gpus=16,per_checkpoint_worker_limit=per_checkpoint_limit(state),scheduler='dynamic-per-checkpoint-share-v2',active=[{k:v for k,v in j.items() if k!='process'} for j in active.values()]);save(WORK/'status.json',info);summaries=results();next_update=time.time()+15
    if finished:
     assert len(summaries)==6
     for v in summaries.values():
      r=v['scores'];assert all(r['coverage'][k]['pairs']==256 for k in ['psnr','ssim','lpips','fid','fvd']) and r['audio']['valid_pairs']==256 and r['audio']['pairs_with_errors']==0
     save(WORK/'COMPLETE.json',dict(success=True,at=time.time(),outputs=1536,combinations=6,physical_gpus=16,nodes=['node12','node11'],metrics_coverage='See per-combination coverage; partial emotion/deltaResp possible'));return
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
  r=reports[s['node']];g=r['gpus'][str(s['gpu'])];assert g['free']/1024>=44,(s,g)
 save(WORK/'ACTIVATED.json',dict(at=time.time(),uuids={str(s['rank']):reports[s['node']]['gpus'][str(s['gpu'])]['uuid'] for s in slots()},nodes=['node12','node11'],gpus=16));queue()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','activate','queue','worker','report','integrity']);p.add_argument('--stage',choices=['cache','render','decode']);p.add_argument('--rank',type=int,default=0);p.add_argument('--key',default='');p.add_argument('--qwen',type=int,default=0);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
 try:
  if a.action=='integrity':verify()
  elif a.action=='prepare':prepare()
  elif a.action=='activate':activate()
  elif a.action=='queue':queue()
  elif a.action=='worker':worker(a.stage,a.rank,a.key,a.qwen,a.shard)
  else:results()
 except BaseException:
  save(WORK/f'FAILURE_{a.action}_{a.stage}_{a.rank}_{a.key}_{a.qwen}_{a.shard}.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
