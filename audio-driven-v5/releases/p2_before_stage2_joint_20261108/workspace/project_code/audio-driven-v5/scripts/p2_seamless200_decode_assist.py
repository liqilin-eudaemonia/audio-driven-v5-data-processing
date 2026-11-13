"""Use node8 GPU0-3 to export a finished sampling arm before its whole group ends."""
import argparse
import copy
import fcntl
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,partition,PYTHON
from p2_response200_fourway import link


def run(work):
 from p2_testsets_eval.runtime import ready
 from p2_testsets_eval.cluster import run_cluster,bind
 from listen_cluster import transport
 work=work.resolve();child=work/'sft_event';aux=work/'node8_decode_assist'
 aux.mkdir(exist_ok=True)
 with (aux/'controller.lock').open('a') as control:
  fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB)
  cp=read(child/'protocol.json');arm=cp['arms'][0]
  items=[x for rank in range(cp['workers']) for x in read(child/'shards'/f'{rank:02d}.json')]
  if len(items)!=200 or not all(ready(child/'latents'/arm['name']/x['key']) for x in items):
   raise ValueError('Assistance requires the complete fixed200 sampling arm')
  cfg=read(work/'metrics_cluster.json');node=cfg['nodes'][0]
  if node['host']!='172.29.104.237' or node['gpus']!=[0,1,2,3]:raise ValueError('Wrong authorized GPUs')
  dc=copy.deepcopy(cfg);dc['nodes'][0]['gpus']=[0,1,2,3]
  save(aux/'cluster.json',dc)
  ap=copy.deepcopy(cp);ap.update(workers=4,arms=[arm],total_outputs=200,cluster=str(aux/'cluster.json'))
  save(aux/'protocol.json',ap)
  for i,part in enumerate(partition(items,4)):save(aux/'shards'/f'{i:02d}.json',part)
  (child/'generated').mkdir(exist_ok=True)
  for folder in ('latents','generated'):link(child/folder,aux/folder)
  # Pre-publish matching directory links, but no bind/readiness marker. The
  # original controller can later publish the identical targets without races.
  for candidate in cp['arms']:
   for folder in ('generated','maps','bindings','metric_manifests'):
    link(child/folder/candidate['name'],work/folder/candidate['name'])
  def state(phase,**kw):save(aux/'status.json',dict(phase=phase,at=time.time(),arm=arm['name'],**kw));print(phase,kw,flush=True)
  # Remove only new scoring leases. Existing scoring jobs finish normally.
  scoring_cfg=copy.deepcopy(cfg);scoring_cfg['nodes'][0]['gpus']=[]
  save(work/'metrics_cluster.json',scoring_cfg)
  try:
   state('waiting_for_scoring_release')
   while True:
    if (work/'STOP').exists():return
    p=subprocess.run(transport(dc,node,['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']),capture_output=True,text=True,timeout=30)
    if p.returncode:time.sleep(3);continue
    values={int(s[0]):[int(v) for v in s[1:]] for line in p.stdout.splitlines() if (s:=line.split(','))}
    if all(values[g][0]<1024 and values[g][1]<10 for g in [0,1,2,3]):break
    time.sleep(3)
   state('decoding_on_four_gpus')
   if not (aux/'stages/decode.json').exists():run_cluster(aux,'decode')
   if not all(ready(child/'generated'/arm['name']/x['key']) for x in items):raise ValueError('Incomplete assisted export')
   state('binding_completed_arm')
   bind(child,[arm['name']])
   # The main controller may rewrite equal manifests when binding all arms.
   # Give scoring immutable snapshots so it cannot read during a truncate.
   snapshot=aux/'frozen_metric_manifests'/arm['name']
   for dataset in cp['datasets']:
    for rank in range(cp['workers']):
     source=child/'metric_manifests'/arm['name']/dataset/f'{rank:02d}.jsonl'
     dest=snapshot/dataset/source.name;dest.parent.mkdir(parents=True,exist_ok=True)
     payload=source.read_bytes()
     if dest.exists():
      if dest.read_bytes()!=payload:raise ValueError('Immutable scoring manifest changed')
     else:
      with dest.open('xb') as f:f.write(payload)
   parent=read(work/'protocol.json')
   parent.setdefault('metrics_manifest_overrides',{})[arm['name']]=str(snapshot)
   save(work/'protocol.json',parent)
   save(work/'stages'/('bind_'+arm['name']+'.json'),dict(complete=True,at=time.time(),assisted=True,immutable_manifests=str(snapshot)))
   state('complete',samples=200)
  finally:
   # Restore only the four authorized cards; all other metadata stays current.
   latest=read(work/'metrics_cluster.json');latest['nodes'][0]['gpus']=[0,1,2,3]
   save(work/'metrics_cluster.json',latest)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True)
 run(p.parse_args().work)
