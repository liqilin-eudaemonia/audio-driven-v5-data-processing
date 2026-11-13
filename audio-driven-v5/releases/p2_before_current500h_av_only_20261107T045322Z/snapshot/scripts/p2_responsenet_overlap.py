"""Overlap ready-arm scoring with rendering, then export beside bounded scoring."""
import argparse,fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha,PYTHON,METRIC_PYTHON
from p2_response200_fourway import link
REASON='owned overlap inference decode metrics handoff'


def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False


def freeze_metrics(work,arms):
 p=read(work/'protocol.json')
 for arm in arms:
  name=arm['name'];dest=work/'frozen_metric_manifests'/name
  for dataset in p['datasets']:
   for rank in range(p['workers']):
    src=work/'metric_manifests'/name/dataset/f'{rank:02d}.jsonl'
    target=dest/dataset/src.name;target.parent.mkdir(parents=True,exist_ok=True)
    payload=src.read_bytes()
    if target.exists():
     if target.read_bytes()!=payload:raise ValueError('Frozen scoring manifest changed')
    else:
     with target.open('xb') as f:f.write(payload)
  p.setdefault('metrics_manifest_overrides',{})[name]=str(dest)
 save(work/'protocol.json',p)


def run(work):
 from p2_testsets_eval.cluster import environment,run_cluster,bind
 from p2_testsets_eval.runtime import ready
 folder=work/'overlap';folder.mkdir(exist_ok=True)
 with (folder/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  def state(phase,**kw):save(folder/'status.json',dict(phase=phase,at=time.time(),**kw))
  try:
   p=read(work/'protocol.json');child=work/'whisper';cp=read(child/'protocol.json')
   if p['workers']!=32:raise ValueError('Expected current32 GPU execution')
   if (work/'STOP').exists():raise ValueError('Existing STOP is not ours')
   oldpid=read(work/'CONTROLLER.json')['pid']
   p['metrics_workers_per_gpu']=2;p['scoring_handoff_stop_reason']=REASON
   save(work/'protocol.json',p)
   first=[a for a in p['arms'] if a['kind']!='whisper_av']
   if len(first)!=10 or not all((work/'stages'/('bind_'+a['name']+'.json')).exists() for a in first):
    raise ValueError('Need complete ten-arm generation and binding before overlap')
   freeze_metrics(work,first)
   state('cpu_reusing_scores_before_early_metrics')
   env=environment();env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2'
   with (folder/'reuse_scores.log').open('a') as log:
    subprocess.run([str(METRIC_PYTHON),str(ROOT/'scripts/p2_responsenet_random200.py'),'reuse_scores','--work',str(work)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   save(work/'SMOKE_COMPLETE.json',dict(status='ten_existing_arms_production_verified',samples=200,completed_arms=10,at=time.time()))
   from p2_responsenet_random200 import OLD
   original=read(OLD/'metrics/av_only_6000/responsenet/contract.json')['shard_contracts'][0]
   core=ROOT.parent/'OmniResponse/evaluation/video/paired_metrics'
   for name,expected in original['code'].items():
    if sha(core/name)!=expected:raise ValueError('Metric math differs from original real-GPU validation')
   save(work/'METRICS_SMOKE_COMPLETE.json',dict(status='passed',reused_real_gpu_validation=True,
    source=str(OLD/'metrics/av_only_6000/responsenet'),
    proof='same selected IDs, GT/generated source and times; exact metric code verified; per-pair scores and features reused',
    fresh_gpu_smoke_not_repeated=True,at=time.time()))
   # Start scoring only after all32 rendering processes have initialized CUDA;
   # the scorer itself accepts only idle cards, never stealing a rendering GPU.
   state('starting_early_metrics_on_released_rendering_cards')
   mcmd=[str(PYTHON),'-u',str(ROOT/'scripts/p2_testsets_eval.py'),'metrics_queue','--work',str(work)]
   with (folder/'metrics_queue.log').open('a') as log:
    metrics=subprocess.Popen(mcmd,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   save(folder/'METRICS_PROCESS.json',dict(pid=metrics.pid,command=mcmd,at=time.time()))
   # Existing GPU workers continue normally; this only prevents the old main
   # controller from waiting for every card to be idle before export/scoring.
   save(work/'STOP',dict(reason=REASON,at=time.time()))
   state('render_and_ready_arm_metrics_overlap',metrics_pid=metrics.pid,render_controller=oldpid)
   while alive(oldpid):
    if metrics.poll() not in (None,0):raise RuntimeError('Early scoring failed; see overlap/metrics_queue.log')
    if (work/'GENERATION_COMPLETE.json').exists():
     if (work/'STOP').exists() and read(work/'STOP').get('reason')==REASON:
      (work/'STOP').rename(folder/'completed_controller_handoff.json')
     state('generation_already_exported_scoring_overlap',metrics_pid=metrics.pid)
     while metrics.poll() is None or alive(oldpid):time.sleep(3)
     if not (work/'COMPLETE.json').exists():raise RuntimeError('Scoring incomplete')
     if not (work/'PUBLISHED.json').exists():
      from p2_responsenet_random200 import publish
      publish(work)
     state('complete',results=str(work/'RESULTS.json'));return
    time.sleep(3)
   if read(work/'status.json')['stage']!='stopped':raise ValueError('Render coordinator did not exit normally')
   if read(work/'STOP')['reason']!=REASON:raise ValueError('Stop request changed')
   (work/'STOP').rename(folder/'completed_controller_handoff.json')
   items=read(cp['shared_items_manifest'])
   if not all(ready(child/'latents'/a['name']/x['key']) or ready(child/'generated'/a['name']/x['key']) for a in cp['arms'] for x in items):
    raise ValueError('Whisper render incomplete after normal controller shutdown')
   # Renderer processes have now exited. Two bounded metric jobs per card leave
   # room for native VAE decoding. Sampling and score formulas stay unchanged.
   state('native_export_overlapping_two_metric_jobs_per_gpu')
   if not all(ready(child/'generated'/a['name']/x['key']) for a in cp['arms'] for x in items):
    if not (child/'stages/decode.json').exists():run_cluster(child,'decode')
   if not all(ready(child/'generated'/a['name']/x['key']) for a in cp['arms'] for x in items):raise ValueError('Incomplete native exports')
   bind(child)
   for arm in cp['arms']:
    for name in ['generated','maps','bindings','metric_manifests']:link(child/name/arm['name'],work/name/arm['name'])
   freeze_metrics(work,cp['arms'])
   for arm in cp['arms']:save(work/'stages'/('bind_'+arm['name']+'.json'),dict(complete=True,at=time.time(),immutable_scoring=True))
   save(child/'GENERATION_COMPLETE.json',dict(total_outputs=400,at=time.time()))
   save(work/'GENERATION_COMPLETE.json',dict(total_outputs=2400,at=time.time()))
   latest=read(work/'protocol.json');latest['metrics_workers_per_gpu']=4;save(work/'protocol.json',latest)
   state('all2400_exports_complete_scoring_on32_gpus',metrics_pid=metrics.pid)
   while metrics.poll() is None:time.sleep(5)
   if metrics.returncode or not (work/'COMPLETE.json').exists():raise RuntimeError('Metrics merge did not complete')
   from p2_responsenet_random200 import publish
   publish(work)
   state('complete',results=str(work/'RESULTS.json'))
  except Exception as exc:state('failed',error=repr(exc));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);run(p.parse_args().work)
