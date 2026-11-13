"""Add node8 GPU4-7 immediately; safely adopt a full32 shared queue at a boundary."""
import argparse,copy,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha,partition,PYTHON
from p2_response200_fourway import link
from p2_responsenet_random200 import OLD,PREVIOUS


def code_identity():
 paths=[ROOT/'scripts/p2_responsenet_random200.py',ROOT/'scripts/p2_responsenet_expand32.py',
  ROOT/'src/p2_testsets_eval/reuse_scores.py',ROOT/'src/p2_testsets_eval/whisper_runtime.py',
  ROOT/'src/p2_testsets_eval/cluster.py',ROOT/'src/p2_testsets_eval/runtime.py',ROOT/'src/p2_testsets_eval/shared_items.py']
 return {str(p):sha(p) for p in paths}


def shared_manifest(work,protocol):
 items=[x for rank in range(protocol['workers']) for x in read(work/'shards'/f'{rank:02d}.json')]
 assert len(items)==len({x['key'] for x in items})==200
 path=work/'shared_items.json';save(path,items)
 return items,path


def prepare_aux(work):
 child=work/'whisper';aux=work/'node8_extra4';aux.mkdir(exist_ok=True)
 p=read(child/'protocol.json');items,path=shared_manifest(child,p)
 cluster=read(work/'cluster.json');node=copy.deepcopy(cluster['nodes'][-1]);assert node['host']=='172.29.104.237'
 node['gpus']=[4,5,6,7];cluster['nodes']=[node];save(aux/'cluster.json',cluster)
 cp=copy.deepcopy(p);cp.update(workers=4,cluster=str(aux/'cluster.json'),shared_item_queue=True,
  shared_items_manifest=str(path),shared_items_sha256=sha(path),execution_code=code_identity())
 save(aux/'protocol.json',cp)
 for rank,part in enumerate(partition(items,4)):save(aux/'shards'/f'{rank:02d}.json',part)
 for name in ['latents','generated','whisper','sample_claims']:
  (child/name).mkdir(exist_ok=True);link(child/name,aux/name)
 link(child/'conditions',aux/'conditions')
 for rank in range(4):link(child/'negative'/f'{rank:02d}',aux/'negative'/f'{rank:02d}')
 return aux


def aux_run(aux):
 from p2_testsets_eval.cluster import run_cluster
 for stage in ['cache','render','decode']:
  if (aux/'STOP').exists():break
  if not (aux/'stages'/(stage+'.json')).exists():run_cluster(aux,stage)
 save(aux/'AUX_FINISHED.json',dict(at=time.time(),partial=(aux/'STOP').exists(),
  reason='Completed individual outputs retained; canonical200 completion is checked by main controller'))


def live(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False


def migrate(work):
 from p2_seamless200_whisper_eval import idle_snapshot
 parent=read(work/'protocol.json');oldcluster=read(work/'cluster.json');groups=list(map(Path,read(work/'groups.json')))
 # All original workers must have committed and exited at their normal boundary.
 oldpid=read(work/'CONTROLLER.json')['pid']
 if live(oldpid):raise ValueError('Original coordinator is still running')
 stop=read(work/'STOP')
 if stop['reason']!='owned coordinator safe-boundary expansion from28 to32 GPUs':
  raise ValueError('STOP is not this expansion request')
 cluster=copy.deepcopy(oldcluster);cluster['nodes'][-1]['gpus']=list(range(8))
 while True:
  snap=idle_snapshot(cluster);save(work/'expansion32/resources.json',snap)
  if snap['all_idle']:break
  time.sleep(10)
 archive=work/'expansion32/original28';archive.mkdir(parents=True,exist_ok=True)
 shutil.copy2(work/'protocol.json',archive/'protocol.json');shutil.copy2(work/'cluster.json',archive/'cluster.json')
 before=dict(references=parent['datasets'],selection_sha256=parent['selection_sha256'],
  generation=parent['generation'],seed=parent['seed'],arms=parent['arms'])
 # The full list is kept byte-identical; only execution and source-verified routing change.
 items,path=shared_manifest(work,parent)
 save(work/'cluster.json',cluster)
 parent.update(workers=32,authorized_scope='User added node8 GPU4-7; all32 authorized across four hosts',
  execution_code=code_identity(),shared_item_queue=True,shared_items_manifest=str(path),shared_items_sha256=sha(path),
  gpu_expansion_at=time.time(),original_workers=28)
 for rank,part in enumerate(partition(items,32)):save(work/'shards'/f'{rank:02d}.json',part)
 for child in groups:
  cp=read(child/'protocol.json');shutil.copy2(child/'protocol.json',archive/(child.name+'_protocol.json'))
  cp.update(workers=32,shared_item_queue=True,shared_items_manifest=str(path),shared_items_sha256=sha(path),
   execution_code=parent['execution_code'],original_workers=28)
  for rank,part in enumerate(partition(items,32)):save(child/'shards'/f'{rank:02d}.json',part)
  for rank in range(28,32):
   source=PREVIOUS/'sft_event/negative'/f'{rank:02d}' if child.name=='sft_event' else OLD/'negative'/f'{rank:02d}'
   link(source,child/'negative'/f'{rank:02d}')
  save(child/'protocol.json',cp)
 # Per-pair scores stay intact in the archive; their distribution features will
 # be exactly repartitioned to the new32 score shards by the existing CPU path.
 for rel in ['metric_shards','progress/metrics','REUSED_SCORES.json']:
  src=work/rel
  if src.exists():
   dest=archive/rel;dest.parent.mkdir(parents=True,exist_ok=True);src.rename(dest)
 save(work/'protocol.json',parent)
 after=dict(references=parent['datasets'],selection_sha256=parent['selection_sha256'],
  generation=parent['generation'],seed=parent['seed'],arms=parent['arms'])
 if before!=after:raise ValueError('Expansion changed a frozen experiment input')
 (work/'STOP').rename(archive/'owned_expansion_STOP.json')
 save(work/'expansion32/MIGRATED.json',dict(at=time.time(),from_workers=28,to_workers=32,
  fixed200_unchanged=True,all12_checkpoints_unchanged=True,sampling_math_unchanged=True,
  completed_outputs_preserved=True,shared_render_decode_claims=True,reference_identity=before))
 env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
 cmd=[str(PYTHON),'-u',str(ROOT/'scripts/p2_responsenet_random200.py'),'queue','--work',str(work)]
 with (work/'controller.log').open('a') as log:
  proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(work/'CONTROLLER.json',dict(pid=proc.pid,command=cmd,at=time.time(),expanded_to32=True))
 return proc.pid


def run(work):
 folder=work/'expansion32';folder.mkdir(exist_ok=True)
 with (folder/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  def state(stage,**kw):save(folder/'status.json',dict(stage=stage,at=time.time(),**kw))
  try:
   aux=prepare_aux(work)
   env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
   cmd=[str(PYTHON),'-u',str(Path(__file__).resolve()),'aux','--work',str(aux)]
   with (aux/'controller.log').open('a') as log:
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   save(aux/'CONTROLLER.json',dict(pid=proc.pid,command=cmd,at=time.time()))
   state('extra4_running_whisper_while_original28_finish',auxiliary_pid=proc.pid)
   oldpid=read(work/'CONTROLLER.json')['pid']
   while live(oldpid):
    if proc.poll() not in (None,0):raise RuntimeError('Extra4 helper failed; see node8_extra4/controller.log')
    time.sleep(5)
   if read(work/'status.json')['stage']!='stopped':raise ValueError('Original coordinator did not stop normally')
   save(aux/'STOP',dict(reason='Finish current sample and hand off to canonical32 queue',at=time.time()))
   state('draining_extra4_at_sample_boundary')
   code=proc.wait(timeout=600)
   if code:raise RuntimeError('Extra4 helper exited with failure')
   state('migrating_frozen_inputs_and_execution_only')
   pid=migrate(work)
   state('complete_32gpu_controller_running',pid=pid)
  except Exception as e:state('failed',error=repr(e));raise


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['run','aux']);p.add_argument('--work',type=Path,required=True);a=p.parse_args()
 run(a.work) if a.action=='run' else aux_run(a.work)
