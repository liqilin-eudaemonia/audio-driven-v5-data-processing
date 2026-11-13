"""Independent evaluation queue on the user's inference host; idle GPUs only."""
import concurrent.futures,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from .protocol import ROOT,OMNI,METRIC_PYTHON,read,save,rows
from .cluster import CLI,environment,stopped


def metrics_worker(work,rank,arms=None,smoke=False):
 work=Path(work);p=read(work/'protocol.json');env=environment()
 chosen=[a for a in p['arms'] if arms is None or a['name'] in arms]
 if smoke:utmos_backend_smoke(work)
 (work/'locks').mkdir(exist_ok=True)
 lock=(work/'locks'/('metrics_'+('_'.join(arms or ['all']))+f'_{rank}_{smoke}.lock')).open('a')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for arm in chosen:
  for dataset,spec in p['datasets'].items():
   folder=work/('metric_smoke' if smoke else 'metric_shards')/arm['name']/dataset/f'{rank:02d}'
   manifest=work/('smoke_metric_manifests' if smoke else 'metric_manifests')/arm['name']/dataset/f'{rank:02d}.jsonl'
   if not smoke and arm['name'] in p.get('metrics_manifest_overrides',{}):
    manifest=Path(p['metrics_manifest_overrides'][arm['name']])/dataset/f'{rank:02d}.jsonl'
   for kind in ['video','audio']:
    out=folder if kind=='video' else folder/'audio'
    if (out/'summary.json').exists():
     result=read(out/'summary.json')
     if kind=='audio' or ((out/'features.npz').exists() and not result['initialization_errors']):
      if smoke:validate_smoke(result,kind,out)
      continue
    if kind=='video' and out.exists():out.rename(out.with_name(out.name+f'.failed.{time.time_ns()}'))
    elif kind=='audio' and out.exists():out.rename(out.with_name(out.name+f'.failed.{time.time_ns()}'))
    folder.mkdir(parents=True,exist_ok=True)
    common=['--manifest',str(manifest),'--assets',p['metric_assets'],'--output',str(out)]
    if kind=='video':
     cmd=[str(METRIC_PYTHON),'-m','evaluation.video.paired_metrics.run',*common,'--device','cuda:0','--threads','4',
          '--batch-size',str(p['metric_batch_size']),'--face-padding',str(spec['face_padding']),
          '--feature-output',str(out/'features.npz'),'--allow-partial']
     if p.get('metrics_face_device'):
      cmd+=['--face-device',p['metrics_face_device'],'--emotion-device',p.get('metrics_emotion_device','cpu'),
            '--pixel-workers',str(p.get('metrics_pixel_workers',1))]
     if p.get('metrics_gt_cache'):cmd+=['--gt-cache',p['metrics_gt_cache']]
     if dataset=='seamless' and p.get('metrics_decoded_gt_cache'):
      cmd=[str(METRIC_PYTHON),str(ROOT/'scripts/p2_cached_video_metrics.py'),
           '--decoded-gt-cache',p['metrics_decoded_gt_cache'],*cmd[3:]]
    else:
     cmd=[str(METRIC_PYTHON),'-m','evaluation.listener_metrics.run',*common,'--stages','audio',
          '--quality-kinds','utmos','--audio-device','cuda:0']
    with (folder/(kind+'.log')).open('w') as log:
     code=subprocess.run(cmd,cwd=OMNI,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    if not (out/'summary.json').exists():raise RuntimeError(f'{kind} failed {code}: {out}')
    result=read(out/'summary.json')
    if result.get('initialization_errors'):raise RuntimeError(str(result['initialization_errors']))
    if smoke:validate_smoke(result,kind,out)
 save(work/'progress/metrics'/('smoke_' if smoke else '')/(chosen[0]['name']+f'_{rank:02d}.json'),dict(status='complete',at=time.time()))


def validate_smoke(result,kind,out):
 if kind=='video':
  for metric in ['psnr','ssim','lpips','fid','fvd']:
   if result['coverage'][metric]['pairs']!=2 or result['metrics'][metric] is None:raise RuntimeError('Video smoke failed: '+str(out))
 elif result['pairs_with_errors']:raise RuntimeError('Audio smoke failed: '+str(out))


def utmos_backend_smoke(work):
 # Existing known long speech is ONLY an isolated backend self-check, never a benchmark result.
 import shutil
 from evaluation.listener_metrics.bridge import Worker
 folder=work/'utmos_backend_gpu_smoke';folder.mkdir(exist_ok=True)
 if (folder/'result.json').exists():return
 source=OMNI/'evaluation/listener_metrics/backend_self_check_20261102/speech.wav'
 for side in ['gt','generated']:shutil.copyfile(source,folder/(side+'.wav'))
 worker=Worker('audio',folder,'cuda:0')
 try:
  result=worker.call(dict(kind='utmos',gt_audio=str(folder/'gt.wav'),generated_audio=str(folder/'generated.wav')))
  assert result['status']=='success' and abs(result['scores']['gt']['value']-result['scores']['generated']['value'])<1e-5
  save(folder/'result.json',dict(self_check_only=True,source=str(source),result=result))
 finally:worker.close()


def prepare_smoke(work):
 from evaluation.listener_metrics.suite import enrich
 p=read(work/'protocol.json');arm=p['arms'][0]['name']
 items=read(work/'shards/00.json')[:2]+read(work/'shards/01.json')[:2]
 for dataset,spec in p['datasets'].items():
  dest=work/'smoke_metric_manifests'/arm/dataset/'00.jsonl'
  if dest.exists():
   existing=rows(dest)
   if len(existing)==2 and all(Path(x['generated']).exists() and Path(x['generated_audio']).exists() and (dataset!='responsenet' or 'source_start' in x) for x in existing):continue
  selected=[x for x in items if x['dataset']==dataset][:2];assert len(selected)==2
  paired=[];mapping=[]
  for x in selected:
   r=x['reference'];folder=work/'generated'/arm/x['key']
   row=dict(id=r['id'],dataset=dataset,identity='listener',generated=str(folder/'generated.mp4'),listening_intervals=r['listening_intervals'])
   if dataset=='seamless':row['gt_reference']=r['gt_reference']
   else:row.update(gt=r['gt'],gt_resize=[512,512],gt_resize_method='pil_bicubic',source_start=r['source_start'],source_end=r['source_end'])
   paired.append(row);mapping.append(dict(id=r['id'],generated_audio=str(folder/'listener.wav')))
  paired=enrich(paired,[x['reference'] for x in selected],mapping,work)
  dest=work/'smoke_metric_manifests'/arm/dataset/'00.jsonl';dest.parent.mkdir(parents=True,exist_ok=True)
  dest.write_text(''.join(json.dumps(x)+'\n' for x in paired))


def idle_gpus(report):
 result=[]
 for line in report.splitlines():
  index,used,total,util=[float(x.strip()) for x in line.split(',')]
  if used<4096 and util<10:result.append(int(index))
 return result


def balanced_metric_slots(reports, active):
 """Use every available GPU before adding more jobs to an already busy GPU."""
 caps={(name,gpu):cap for name,cap,gpus in reports for gpu in gpus}
 counts={key:sum(v[1]==key[0] and v[2]==key[1] for v in active.values()) for key in caps}
 slots=[]
 while True:
  eligible=[key for key in caps if counts[key]<caps[key]]
  if not eligible:return slots
  key=min(eligible,key=lambda key:(counts[key],key))
  slots.append(key);counts[key]+=1


def metric_hosts(work, protocol):
 """Inference GPUs become eligible only after the complete generation marker."""
 cfg=read(protocol['metrics_cluster'])
 hosts={n['name']:(cfg,n,protocol['metrics_workers_per_gpu']) for n in cfg['nodes']}
 if protocol.get('metrics_use_released_inference_nodes') and (work/'GENERATION_COMPLETE.json').exists():
  inference=read(protocol['cluster'])
  for node in inference['nodes']:
   hosts[node['name']]=(inference,node,protocol.get('metrics_inference_workers_per_gpu',2))
 return hosts


class AdoptedMetricProcess:
 """Observe a surviving local SSH/worker after a controller-only restart."""
 def __init__(self,pid,marker):
  self.pid=pid;self.marker=marker;self.returncode=None
 def poll(self):
  if self.returncode is not None:return self.returncode
  try:
   state=Path(f'/proc/{self.pid}/stat').read_text().split(') ',1)[1].split()[0]
   if state!='Z':return None
  except FileNotFoundError:pass
  self.returncode=0 if self.marker.exists() and read(self.marker).get('status')=='complete' else 1
  return self.returncode


def acquire_queue_or_observe_completion(work, lock, poll_s=3):
 """A later controller joins a live scorer instead of starting duplicate jobs."""
 while True:
  if (work/'COMPLETE.json').exists() or stopped(work):return False
  try:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   return True
  except BlockingIOError:
   if (work/'COMPLETE.json').exists() or stopped(work):return False
   time.sleep(poll_s)


def queue(work):
 sys.path.insert(0,str(ROOT/'scripts'));from listen_cluster import transport
 work=Path(work);lock=(work/'metrics_queue.lock').open('a')
 if not acquire_queue_or_observe_completion(work,lock):
  lock.close();return
 p=read(work/'protocol.json');primary=read(p['metrics_cluster']);primary_node=primary['nodes'][0]
 status=lambda stage,**kw:save(work/'metrics_status.json',dict(stage=stage,host=primary_node['host'],at=time.time(),**kw))
 active={};completed=set();attempts={};smoke_ready=False
 # Recover completed and still-running jobs without killing/restarting workers.
 for arm in p['arms']:
  for rank in range(p['workers']):
   name=arm['name']+f'_{rank:02d}';job=(arm['name'],rank,False)
   marker=work/'progress/metrics'/(name+'.json')
   if marker.exists() and read(marker).get('status')=='complete':
    completed.add(job);continue
   process_file=work/'processes/metrics_remote'/(name+'.json')
   if not process_file.exists():continue
   info=read(process_file);pid=info['pid']
   try:command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
   except FileNotFoundError:continue
   if str(work) not in command or str(CLI) not in command:continue
   if f'--rank {rank} ' not in command or f'--arms {arm["name"]}' not in command:continue
   active[job]=(AdoptedMetricProcess(pid,marker),info.get('node','evaluation'),info['gpu'],Path(info['log']))
   attempts[job]=1
 def launch(job,host_name,gpu,hosts):
  cfg,node,_=hosts[host_name]
  arm,rank,smoke=job;name=('smoke_' if smoke else '')+arm+f'_{rank:02d}';attempts[job]=attempts.get(job,0)+1
  folder=work/'logs/metrics_remote';folder.mkdir(parents=True,exist_ok=True);log=folder/f'{name}_{time.time_ns()}.log'
  cmd=['env',f'CUDA_VISIBLE_DEVICES={gpu}','PYTHONPATH=/private/liqi_workspace/.cache/paired_video_metrics/python',
       'OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=2',str(METRIC_PYTHON),'-u',str(CLI),'metrics','--work',str(work),'--rank',str(rank),'--arms',arm]
  if smoke:cmd+=['--metric-smoke']
  with log.open('w') as stream:proc=subprocess.Popen(transport(cfg,node,cmd),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
  active[job]=(proc,host_name,gpu,log)
  save(work/'processes/metrics_remote'/(name+'.json'),dict(pid=proc.pid,gpu=gpu,host=node['host'],node=host_name,command=cmd,log=str(log)))
 try:
  while True:
   if stopped(work):status('stopped');return
   for job,(proc,host_name,gpu,log) in list(active.items()):
    if proc.poll() is None:continue
    del active[job]
    if proc.returncode:
     connection_drop=proc.returncode==255 and 'kex_exchange_identification' in log.read_text(errors='replace')
     if attempts[job]>=(8 if connection_drop else 3):raise RuntimeError('Metric job failed after bounded retries: '+str(log))
    else:
     completed.add(job)
     if job[2]:save(work/'METRICS_SMOKE_COMPLETE.json',dict(status='passed',host=host_name,gpu=gpu,at=time.time()))
   wanted=[]
   if (work/'SMOKE_COMPLETE.json').exists():
    if not smoke_ready:prepare_smoke(work);smoke_ready=True
    if not (work/'METRICS_SMOKE_COMPLETE.json').exists():wanted=[(p['arms'][0]['name'],0,True)]
    else:
     for arm in p['arms']:
      if (work/'stages'/('bind_'+arm['name']+'.json')).exists():
       wanted.extend((arm['name'],rank,False) for rank in range(p['workers']))
   pending=[j for j in wanted if j not in completed and j not in active]
   hosts=metric_hosts(work,read(work/'protocol.json'))
   if pending:
    def report_one(item):
     name,(cfg,node,capacity)=item
     try:
      report=subprocess.check_output(transport(cfg,node,['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']),text=True,timeout=30)
     except (subprocess.CalledProcessError,subprocess.TimeoutExpired,OSError) as exc:
      save(work/'resource_errors'/(name+'.json'),dict(at=time.time(),error=repr(exc),action='skip host this poll; retry next poll'))
      return name,capacity,[]
     available=sorted(set(idle_gpus(report))|{v[2] for v in active.values() if v[1]==name})
     allowed=set(node.get('gpus',range(8)))
     return name,capacity,[gpu for gpu in available if gpu in allowed]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as pool:
     reports=list(pool.map(report_one,hosts.items()))
    for name,gpu in balanced_metric_slots(reports,active):
     if not pending:break
     launch(pending.pop(0),name,gpu,hosts)
   production=sum(not job[2] for job in completed)
   by_host={name:sum(v[1]==name for v in active.values()) for name in hosts}
   status('running' if active else 'waiting_gpu_or_generated_outputs',active=len(active),active_by_host=by_host,
          completed_shards=production,total_shards=len(p['arms'])*p['workers'],pending=len(pending),
          inference_gpus_released=(work/'GENERATION_COMPLETE.json').exists())
   if production==len(p['arms'])*p['workers']:
    status('merging')
    cmd=['env','PYTHONPATH=/private/liqi_workspace/.cache/paired_video_metrics/python','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',str(METRIC_PYTHON),str(CLI),'merge','--work',str(work)]
    for attempt in range(1,9):
     with (work/'metrics_merge.log').open('a') as log:result=subprocess.run(transport(primary,primary_node,cmd),stdout=log,stderr=subprocess.STDOUT)
     if result.returncode==0:break
     if result.returncode!=255 or attempt==8:raise RuntimeError('Metric merge failed; see metrics_merge.log')
     time.sleep(min(attempt*2,15))
    status('complete');return
   time.sleep(15)
 except Exception as e:
  status('failed',error=repr(e));raise
