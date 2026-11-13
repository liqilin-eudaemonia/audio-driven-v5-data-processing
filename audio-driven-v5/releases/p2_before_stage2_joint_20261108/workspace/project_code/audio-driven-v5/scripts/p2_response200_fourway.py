"""The confirmed ResponseNet200: base-H Feature and SFT-H Event, 6000/final."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT.parent/'OmniResponse'),str(ROOT/'scripts')]
from p2_testsets_eval.protocol import read,save,sha,rows,geometry,partition,PYTHON
DEFAULT=ROOT/'experiments/p2_response200_feature_sft_event_20261103'
OLD=ROOT/'experiments/p2_testsets_4ckpt_24gpu_20261102'
CLI=ROOT/'scripts/p2_testsets_eval.py'

class JoinedMetrics:
 def __init__(self,pid,work):self.pid=pid;self.work=work;self.returncode=None
 def poll(self):
  if self.returncode is not None:return self.returncode
  try:
   if Path('/proc',str(self.pid),'stat').read_text().split(') ',1)[1].split()[0]!='Z':return None
  except FileNotFoundError:pass
  self.returncode=0 if (self.work/'COMPLETE.json').exists() else 1
  return self.returncode


def link(src,dst):
 dst=Path(dst);src=Path(src)
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.is_symlink():
  if dst.resolve()!=src.resolve():raise ValueError('Symlink target changed: '+str(dst))
 elif dst.exists():raise FileExistsError(dst)
 else:dst.symlink_to(src,target_is_directory=True)


def prepare(work):
 selection=ROOT/'experiments/responsenet_initial200_20261102'
 refs=rows(selection/'references.jsonl');seal=read(selection/'seal.json')
 if len(refs)!=200 or len({r['id'] for r in refs})!=200:raise ValueError('Expected exactly 200 fixed pairs')
 for name in ('references.jsonl','selection.csv'):
  if sha(selection/name)!=seal['outputs_sha256'][str(selection/name)]:raise ValueError('Confirmed selection changed')
 work.mkdir(exist_ok=False)
 prepared=work/'references';prepared.mkdir()
 (prepared/'references.jsonl').write_bytes((selection/'references.jsonl').read_bytes())
 save(prepared/'seal.json',dict(references_sha256=sha(prepared/'references.jsonl'),source=str(selection),
      original_seal_sha256=sha(selection/'seal.json'),selection_csv_sha256=sha(selection/'selection.csv')))
 old=read(OLD/'protocol.json');p=copy.deepcopy(old)
 p.update(created_at=time.time(),schema='response200-feature-vs-sft-event-v1',datasets=dict(responsenet=dict(
     prepared=str(prepared),references_sha256=sha(prepared/'references.jsonl'),count=200,face_padding=64)),
     arms=[],total_samples=200,total_outputs=800,planner='per_arm_training_matched_Qwen',planner_adapter=None,
     enforce_planner_identity=True,cluster=str(work/'cluster.json'),metrics_cluster=str(work/'metrics_cluster.json'),
     metrics_workers_per_gpu=4,metrics_inference_workers_per_gpu=4)
 for key in ('supersedes','scope_history','deferred_datasets'):p.pop(key,None)
 cluster=read(ROOT/'configs/p2_j_av_feature_base025_32gpu_20261102.json')
 save(work/'cluster.json',cluster);save(work/'metrics_cluster.json',read(OLD/'metrics_cluster.json'))
 items=[dict(key='responsenet/'+r['id'],id=r['id'],dataset='responsenet',reference=r,geometry=geometry(r,'responsenet')) for r in refs]
 shards=partition(items,32)
 for i,shard in enumerate(shards):save(work/'shards'/f'{i:02d}.json',shard)
 base=Path(read(ROOT/'experiments/p2_j_guided_av_feature_base025_20261102/CURRENT_RUN.json')['output'])
 event=ROOT/'experiments/p2_j_sft_h_av_event_ready_20261102'
 adapter=read(ROOT/'experiments/p2_j_train500h_sft_hidden_20261102/protocol.json')['adapter']
 groups=[]
 for name,folder,kind in [('base_feature',base,'av_feature'),('sft_event',event,'av_event')]:
  child=work/name;child.mkdir();cp=copy.deepcopy(p)
  cp.update(total_outputs=400,arms=[],planner='unmodified_Qwen2.5-Omni-3B' if name=='base_feature' else 'planning_SFT_Qwen2.5-Omni-3B',
      cluster=str(work/'cluster.json'),metrics_cluster=str(work/'metrics_cluster.json'))
  cp['planner_adapter']=None if name=='base_feature' else adapter['path']
  if name=='sft_event':cp.update(planner_adapter_identity=adapter,sft_hidden_cache=str(ROOT/'experiments/p2_j_train500h_sft_hidden_20261102'))
  for step in (6000,8927):
   arm=dict(name=f'{name}_{step}',kind=kind,step=step,checkpoint=str(folder/f'step_{step:06d}.pt'),planner=cp['planner'])
   cp['arms'].append(arm);p['arms'].append(arm)
  save(child/'protocol.json',cp)
  for i,shard in enumerate(shards):save(child/'shards'/f'{i:02d}.json',shard)
  if name=='base_feature':
   # Frozen reference IDs, pixels, planner, and prompts are identical to prior generation.
   oldrefs={r['id']:r for r in rows(Path(old['datasets']['responsenet']['prepared'])/'references.jsonl')}
   for ref in refs:
    if ref!=oldrefs[ref['id']]:raise ValueError('Selected reference differs from previous inference')
    src=OLD/'conditions/responsenet'/ref['id']
    from p2_testsets_eval.runtime import ready
    if not ready(src/'qwen') or not ready(src/'text'):raise ValueError('Previous base condition missing')
    plan=read(src/'qwen/plan.json')
    if plan['adapter'] is not None or plan['target_description_access']:raise ValueError('Not a source-only base plan')
    for folder_name in ('qwen','text'):
     for rel,receipt in read(src/folder_name/'complete.json')['files'].items():
      if sha(src/folder_name/rel)!=receipt['sha256']:raise ValueError('Base condition checksum mismatch')
    link(src,child/'conditions/responsenet'/ref['id'])
   link(OLD/'negative',child/'negative')
  groups.append(str(child))
 save(work/'protocol.json',p);save(work/'groups.json',groups)
 save(work/'status.json',dict(stage='prepared_waiting_training',total_outputs=800))
 return p


def queue(work):
 from p2_testsets_eval.cluster import queue as generate
 from p2_j_sft_event_queue import resources
 lock=(work/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 p=read(work/'protocol.json');groups=[Path(g) for g in read(work/'groups.json')]
 def state(stage,**kw):save(work/'status.json',dict(stage=stage,at=time.time(),**kw))
 try:
  while True:
   if (work/'STOP').exists():state('stopped');return
   event=ROOT/'experiments/p2_j_sft_h_av_event_ready_20261102'
   status=read(event/'status.json');controller=read(event/'queue/state.json')
   if status['complete'] and status['step']==8927 and controller['state']=='complete':break
   state('waiting_training',training=status);time.sleep(15)
  # Verify all four exact checkpoints before allocating any inference GPUs.
  hashes={}
  for arm in p['arms']:
   receipt=read(Path(arm['checkpoint']).with_suffix('.complete.json'))
   if receipt['step']!=arm['step'] or sha(arm['checkpoint'])!=receipt['sha256']:raise ValueError('Checkpoint mismatch')
   hashes[arm['name']]=receipt['sha256']
  save(work/'checkpoint_hashes.json',hashes)
  while True:
   if (work/'STOP').exists():state('stopped');return
   try:gpu=resources(read(p['cluster']))
   except (subprocess.TimeoutExpired,OSError) as exc:
    state('waiting_gpu_query_retry',error=repr(exc));time.sleep(15);continue
   save(work/'resources.json',dict(at=time.time(),nodes=gpu))
   if all(x['free'] for x in gpu):break
   state('waiting_gpu_release');time.sleep(15)
  # One shared metric queue scores completed arms while later inference continues.
  metrics=None
  if (work/'metrics_launch.json').exists():
   pid=read(work/'metrics_launch.json')['pid'];proc=Path('/proc',str(pid))
   try:
    command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if str(work) in command and 'metrics_queue' in command and (proc/'stat').read_text().split(') ',1)[1].split()[0]!='Z':
     metrics=JoinedMetrics(pid,work)
   except FileNotFoundError:pass
  if metrics is None:
   with (work/'metrics_queue.log').open('a') as log:
    metrics=subprocess.Popen([str(PYTHON),str(CLI),'metrics_queue','--work',str(work)],cwd=ROOT,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   save(work/'metrics_launch.json',dict(pid=metrics.pid))
  for child in groups:
   if (work/'STOP').exists():(child/'STOP').touch();state('stopped');return
   state('generating',group=child.name)
   if not (child/'GENERATION_COMPLETE.json').exists():generate(child)
   if not (child/'GENERATION_COMPLETE.json').exists():raise ValueError('Child generation incomplete')
   for arm in read(child/'protocol.json')['arms']:
    for folder in ('generated','metric_manifests','bindings','maps'):
     link(child/folder/arm['name'],work/folder/arm['name'])
    save(work/'stages'/('bind_'+arm['name']+'.json'),dict(complete=True,at=time.time()))
   if not (work/'SMOKE_COMPLETE.json').exists():save(work/'SMOKE_COMPLETE.json',read(child/'SMOKE_COMPLETE.json'))
  save(work/'GENERATION_COMPLETE.json',dict(total_outputs=p['total_outputs'],at=time.time()))
  state('scoring',total_outputs=p['total_outputs'])
  while metrics.poll() is None:
   if (work/'STOP').exists():state('stopped');return
   time.sleep(15)
  if metrics.returncode or not (work/'COMPLETE.json').exists():raise RuntimeError('Metrics queue failed')
 except Exception as e:state('failed',error=repr(e));raise


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','queue'])
 parser.add_argument('--work',type=Path,default=DEFAULT);a=parser.parse_args()
 prepare(a.work.resolve()) if a.action=='prepare' else queue(a.work.resolve())
