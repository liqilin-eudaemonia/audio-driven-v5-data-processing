"""Two GLARE checkpoints on the exact Qwen/LTX256, native inference/scoring."""
import argparse,copy,fcntl,json,os,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

SELF=Path(__file__).resolve();BASE=SELF.parent;ROOT=BASE.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
import eval_glare_seamless_test_20261112 as native
from p2_testsets_eval.protocol import read,rows,sha,METRIC_PYTHON
from listen_cluster import transport
core=native.core;save=core.save;GENPY=core.GENPY
GUARD=ROOT.parent/'audio-driven-v4-p3-autoselect-20261013/scripts/pipeline3_cluster.py'
SOURCE=ROOT/'experiments/p2_joint_qwen_ltx_test256_16gpu_20261113'
CURRENT=ROOT/'experiments/glare10618_seamless_test6624_20261113'
HISTORY={19050:ROOT/'experiments/glare19050_seamless_test6624_20261112',10618:CURRENT}
NAMES={19050:'before_extension30epochs.pt',10618:'before_extension20epochs.pt'}
STEP=19050;WORK=BASE/'step19050'

def configure(step):
 global STEP,WORK
 STEP=step;WORK=BASE/f'step{step}'
 native.WORK=WORK;core.WORK=WORK;core.SELF=SELF;native.SELF=SELF
 core.CHECKPOINT=ROOT/'experiments/glare_paper_settings_node11_gpu7_20261111'/NAMES[step]
 core.ARM=f'glare_step{step}';core.MODEL=f'GLARE step{step}'

def link(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists():dst.symlink_to(src,target_is_directory=src.is_dir())

def prepare():
 import torch,numpy as np
 from PIL import Image
 WORK.mkdir(parents=True,exist_ok=True)
 if (WORK/'PREPARED.json').exists():
  assert read(WORK/'protocol.json')['entry_sha256']==sha(SELF);return
 weights=torch.load(core.CHECKPOINT,map_location='cpu',weights_only=False,mmap=True)
 assert weights['step']==STEP;del weights
 prior=read(HISTORY[STEP]/'protocol.json');cur=read(CURRENT/'protocol.json')
 items=read(SOURCE/'shared_items.json');assert len(items)==len({x['id'] for x in items})==256
 current={s['id']:s for s in cur['samples']};samples=[];reuse=[]
 link(CURRENT/'prosody',WORK/'prosody');_,cachekey=core.prosody_id()
 for item in items:
  uid=item['id'];s=copy.deepcopy(current[uid]);a=s['item']['reference'];b=item['reference']
  assert all(a['gt_reference'][k]==b['gt_reference'][k] for k in ['row_sha256','crop_config_sha256','reader_contract'])
  assert all(a[k]==b[k] for k in ['frames','fps','duration_s','listening_intervals','source_start','source_end'])
  image=SOURCE/'combinations/qwen7500_ltx6800/conditions'/item['key']/'first_frame.png'
  with Image.open(image) as im,Image.open(s['reference']) as old:assert np.array_equal(np.asarray(im.convert('RGB')),np.asarray(old.convert('RGB'))),uid
  assert sha(s['reference'])==s['reference_sha256'] and sha(s['audio'])==s['audio_sha256'] and core.cache_path(s,cachekey).exists()
  s.update(item=item,evaluation_reference=b,ready=str(WORK/'inputs'/uid/'READY.json'),quick200=False)
  samples.append(s)
  source=HISTORY[STEP]/'seamless/generated'/uid;receipt=source/'complete.json'
  if not receipt.exists():continue
  r=read(receipt)
  if r['reference_sha256']!=s['reference_sha256'] or r['audio_sha256']!=s['audio_sha256']:continue
  assert r['step']==STEP and r['checkpoint_sha256']==prior['checkpoint_sha256'] and r['seed']==137 and r['nfe']==10
  assert r['score_frames']==s['frames'] and r['score_fps']==25 and r['source_only'] and not r['listener_audio_generated']
  score=HISTORY[STEP]/'seamless/scoring'/core.ARM/(uid+'.listener.mp4')
  assert sha(source/'generated.mp4')==r['native_sha256'] and sha(score)==r['score_sha256']
  link(source/'generated.mp4',WORK/'seamless/generated'/uid/'generated.mp4')
  link(score,WORK/'seamless/scoring'/core.ARM/score.name)
  reuse.append(dict(id=uid,receipt=r,source=str(receipt)))
 p=copy.deepcopy(cur)
 for key in ['physical_gpu','gpu_uuid','quick200_ids','quick200_references_sha256','source_updated_ids','historical_comparability','acceleration']:
  p.pop(key,None)
 p.update(schema='glare-current-test256-three-node-v1',created=time.time(),step=STEP,model=core.MODEL,
  checkpoint=str(core.CHECKPOINT),checkpoint_sha256=sha(core.CHECKPOINT),seed=137,nfe=10,audio_guidance=2.,
  samples=samples,total=256,metric_shards=24,physical_gpus=list(range(8)),nodes=['node12','node8','node11'],
  shared_items_sha256=sha(SOURCE/'shared_items.json'),entry_sha256=sha(SELF),reused_generated=len(reuse),
  references=str(ROOT/'experiments/seamless_test_tuning256_20261113/prepared_256/references.jsonl'),
  audio_metrics='N/A: GLARE generates no listener audio; conditioning soundtrack is not a prediction',
  execution='One scorer per GPU, >=48GiB free at launch, torch allocator cap24GiB. Four generators on idle node12 GPUs, cap22GiB. Native model/math unchanged.',
  comparison='seed137, exact same256 and current GT as Qwen/LTX seed137; Qwen/LTX two-seed mean is a separate statistic')
 p.update(references_sha256=sha(p['references']),seal_sha256=sha(Path(p['references']).parent/'seal.json'),
  total_duration_s=sum(s['duration'] for s in samples),
  code={str(f):sha(f) for f in [SELF,ROOT/'scripts/eval_glare_seamless_test_20261112.py',ROOT/'scripts/eval_glare_final_20261112.py',ROOT/'scripts/glare_resident_inference.py',ROOT/'scripts/infer_glare_checkpoint.py',ROOT/'scripts/p2_pipelined_video_metrics.py']})
 assert p['checkpoint_sha256']==prior['checkpoint_sha256']
 save(WORK/'protocol.json',p)
 for s in samples:save(Path(s['ready']),dict(sample=s,source_only=True,protocol_sha256=sha(WORK/'protocol.json')))
 for r in reuse:save(WORK/'seamless/generated'/r['id']/'complete.json',dict(r['receipt'],reused_from=r['source'],protocol_sha256=sha(WORK/'protocol.json')))
 missing=[s for s in samples if not (WORK/'seamless/generated'/s['id']/'complete.json').exists()]
 for i in range(4):save(WORK/'generation_shards'/f'{i}.json',missing[i::4])
 for i in range(24):save(WORK/'blocks'/f'{i:03d}.json',samples[i::24])
 save(WORK/'PROSODY_COMPLETE.json',dict(clips=256,reused=True))
 save(WORK/'PREPARED.json',dict(success=True,step=STEP,total=256,reused=len(reuse),new=len(missing),checkpoint_sha256=p['checkpoint_sha256'],same256_manifest_sha256=p['shared_items_sha256']))

def gpu(rank):
 slots=[(n,g) for n in read(BASE/'cluster.json')['nodes'] for g in n['gpus']];n,g=slots[rank]
 assert os.environ['CUDA_VISIBLE_DEVICES']==str(g)
 assert n['host'] in subprocess.check_output(['hostname','-I'],text=True).split()
 a=subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=uuid,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
 assert a[0].strip()==read(BASE/'ACTIVATED.json')['uuids'][str(rank)]
 return dict(uuid=a[0].strip(),free_mib=int(a[1]),total_mib=int(a[2]),utilization=int(a[3]))

def worker(stage,rank,slot):
 p=read(WORK/'protocol.json');assert sha(SELF)==p['entry_sha256'];g=gpu(rank)
 save(WORK/'devices'/f'{stage}_{slot}.json',dict(at=time.time(),pid=os.getpid(),host=socket.gethostname(),rank=rank,slot=slot,stage=stage,gpu=g))
 if stage=='generate':
  import torch
  torch.cuda.set_per_process_memory_fraction(22*1024/g['total_mib']);core.gpu=lambda:g
  pending=iter([s for s in read(WORK/'generation_shards'/f'{slot}.json') if not (WORK/'seamless/generated'/s['id']/'complete.json').exists()])
  core.claim=lambda _:next(pending,None)
  core.generate(slot,4)
 else:
  original=subprocess.call
  def capped(cmd,**kw):
   assert Path(cmd[1]).name=='p2_pipelined_video_metrics.py'
   return original([cmd[0],str(SELF),'metric_cli',*cmd[1:]],**kw)
  with patch.object(subprocess,'call',capped):native.score(slot)

def metric_cli():
 import torch,runpy
 torch.cuda.set_per_process_memory_fraction(24*2**30/torch.cuda.get_device_properties(0).total_memory)
 sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name='__main__')

def merge():
 import numpy as np
 from evaluation.video.paired_metrics.run import summarize,METRICS
 p=read(WORK/'protocol.json');actual=[];contracts=[];dist={k:[[],[]] for k in ['fid','fvd']};coverage={k:0 for k in dist}
 for i in range(24):
  f=WORK/'metrics'/f'{i:03d}';assert (f/'SUCCESS.json').exists();s=read(f/'summary.json');contracts.append(read(f/'contract.json'));actual+=rows(f/'pairs.jsonl')
  with np.load(f/'features.npz') as a:
   for k in dist:
    dist[k][0].append(a[k+'_gt']);dist[k][1].append(a[k+'_generated']);coverage[k]+=s['coverage'][k]['pairs']
 assert len(actual)==len({r['id'] for r in actual})==256 and {r['id'] for r in actual}=={s['id'] for s in p['samples']}
 assert all(c['assets']==contracts[0]['assets'] and c['code']==contracts[0]['code'] for c in contracts)
 result=summarize(actual,dist,set(METRICS),contracts[0]['schema'],contracts[0]['datasets'],256,{})
 for k,v in coverage.items():result['coverage'][k]['pairs']=v
 result.update(model=core.MODEL,step=STEP,seed=137,checkpoint_sha256=p['checkpoint_sha256'],audio_metrics=p['audio_metrics'],aggregation='full256 pooled features; not mean shard FID/FVD')
 for s in p['samples']:
  r=read(WORK/'seamless/generated'/s['id']/'complete.json');assert r['step']==STEP and r['checkpoint_sha256']==p['checkpoint_sha256'] and r['reference_sha256']==s['reference_sha256'] and r['audio_sha256']==s['audio_sha256'] and r['seed']==137
 save(WORK/'RESULTS.json',result);(WORK/'pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in actual))
 cols=['psnr','ssim','lpips','fid','fvd','emo_acc','emo_div','delta_resp'];lines=[f'# GLARE step{STEP}：相同Seamless256，seed137','', '| 方法 | '+' | '.join(cols)+' |','|---|'+'---:|'*len(cols),f'| GLARE {STEP} | '+' | '.join('N/A' if result['metrics'].get(k) is None else f"{result['metrics'][k]:.4f}" for k in cols)+' |','','音频指标N/A：不生成listener音频。FID/FVD全256条特征汇总；实际有效覆盖见RESULTS.json。']
 (WORK/'RESULTS.md').write_text('\n'.join(lines)+'\n');save(WORK/'COMPLETE.json',dict(success=True,at=time.time(),step=STEP,samples=256,seed=137,coverage=result['coverage']))

def probe(n):
 c=read(BASE/'cluster.json');code="import subprocess,json;print(json.dumps(dict(ips=subprocess.check_output(['hostname','-I'],text=True).split(),gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True))))"
 r=subprocess.run(transport(c,n,[str(GENPY),'-c',code]),capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr
 x=json.loads(r.stdout);assert n['host'] in x['ips'];x['gpus']={int(a[0]):dict(uuid=a[1].strip(),free=int(a[2]),util=int(a[3])) for line in x['gpus'].splitlines() if (a:=line.split(','))};return x

def launch(stage,rank,slot):
 c=read(BASE/'cluster.json');n,g=[(n,g) for n in c['nodes'] for g in n['gpus']][rank]
 python=GENPY if stage=='generate' else METRIC_PYTHON
 native_env=core.env_for('gen' if stage=='generate' else 'response')
 cmd=[str(GENPY),str(GUARD),'--guard','env',f'CUDA_VISIBLE_DEVICES={g}','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=2','WANDB_MODE=disabled','PYTHONPATH='+native_env['PYTHONPATH'],'PYTORCH_ALLOC_CONF=expandable_segments:True',str(python),'-u',str(SELF),'worker','--step',str(STEP),'--stage',stage,'--rank',str(rank),'--slot',str(slot)]
 logfile=WORK/'logs'/f'{stage}_{rank}_{slot}_{time.time_ns()}.log';logfile.parent.mkdir(exist_ok=True)
 with logfile.open('ab',buffering=0) as f:p=subprocess.Popen(transport(c,n,cmd),stdin=subprocess.PIPE,stdout=f,stderr=subprocess.STDOUT)
 return dict(proc=p,pid=p.pid,node=n['name'],gpu=g,rank=rank,stage=stage,slot=slot,log=str(logfile),at=time.time())

def run_step():
 active={};attempts={};donegen=set();p=read(WORK/'protocol.json');started=time.time();nodes=read(BASE/'cluster.json')['nodes']
 try:
  while not (BASE/'STOP').exists():
   for rank,j in list(active.items()):
    rc=j['proc'].poll()
    if rc is None:continue
    j['proc'].stdin.close();del active[rank]
    task=(j['stage'],j['slot'])
    if rc and attempts[task]>=3:raise RuntimeError('Repeated worker failure: '+j['log'])
    if rc and j['stage']=='score':
     folder=WORK/'metrics'/f"{j['slot']:03d}"
     if folder.exists() and not (folder/'SUCCESS.json').exists():folder.rename(folder.with_name(folder.name+f'.failed.{time.time_ns()}'))
    if not rc and j['stage']=='generate':donegen.add(j['slot'])
   generated=sum((WORK/'seamless/generated'/s['id']/'complete.json').exists() for s in p['samples'])
   scored={i for i in range(24) if (WORK/'metrics'/f'{i:03d}/SUCCESS.json').exists()}
   if generated==256 and len(scored)==24 and not active:break
   with ThreadPoolExecutor(3) as pool:reports=dict(zip([n['name'] for n in nodes],pool.map(probe,nodes)))
   for rank,(n,g) in enumerate((n,g) for n in nodes for g in n['gpus']):
    if rank in active or reports[n['name']]['gpus'][g]['free']<48*1024:continue
    job=None
    if rank<4 and rank not in donegen and read(WORK/'generation_shards'/f'{rank}.json'):
     job=('generate',rank)
    elif rank not in scored and all((WORK/'seamless/generated'/s['id']/'complete.json').exists() for s in read(WORK/'blocks'/f'{rank:03d}.json')):job=('score',rank)
    if job:
     attempts[job]=attempts.get(job,0)+1;active[rank]=launch(*[job[0],rank,job[1]])
   save(WORK/'status.json',dict(at=time.time(),step=STEP,generated=generated,total=256,scored_shards=len(scored),total_shards=24,active=[{k:v for k,v in j.items() if k!='proc'} for j in active.values()],elapsed=time.time()-started))
   time.sleep(10)
  if (BASE/'STOP').exists():return False
  with (WORK/'merge.log').open('a') as log:
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='/private/liqi_workspace/.cache/paired_video_metrics/python',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2')
   subprocess.run([str(METRIC_PYTHON),str(SELF),'merge','--step',str(STEP)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  save(WORK/'status.json',dict(at=time.time(),stage='complete',step=STEP,generated=256,total=256,scored_shards=24,active=[]));return True
 finally:
  for j in active.values():j['proc'].stdin.close()

def run():
 signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
 with (BASE/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save(BASE/'CONTROLLER.json',dict(pid=os.getpid(),at=time.time()))
  nodes=read(BASE/'cluster.json')['nodes']
  with ThreadPoolExecutor(3) as pool:reports=dict(zip([n['name'] for n in nodes],pool.map(probe,nodes)))
  save(BASE/'ACTIVATED.json',dict(at=time.time(),uuids={str(i):reports[n['name']]['gpus'][g]['uuid'] for i,(n,g) in enumerate((n,g) for n in nodes for g in n['gpus'])},reports=reports))
  for step in [19050,10618]:
   configure(step);assert read(WORK/'PREPARED.json')['success']
   if not (WORK/'COMPLETE.json').exists() and not run_step():return
  save(BASE/'COMPLETE.json',dict(success=True,steps=[19050,10618],at=time.time()))

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='metric_cli':metric_cli();sys.exit(0)
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run','worker','merge']);ap.add_argument('--step',type=int,choices=[19050,10618],default=19050);ap.add_argument('--stage',choices=['generate','score']);ap.add_argument('--rank',type=int,default=0);ap.add_argument('--slot',type=int,default=0);a=ap.parse_args();configure(a.step)
 try:
  if a.action=='prepare':prepare()
  elif a.action=='run':run()
  elif a.action=='worker':worker(a.stage,a.rank,a.slot)
  else:merge()
 except BaseException:
  save(WORK/f'FAILURE_{a.action}_{a.stage}_{a.slot}.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
