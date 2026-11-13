import sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent;R=W.parents[1];sys.path.insert(0,str(R/'src'))
from listen_benchmark.j_annotation import read,write,locked,acquire
C=read(W/'config.json');O=Path(C['nodes']['node17']['old_work']);Q=O/'queues/nod'
assert read(D/'early_replay.json')['passed']
start=min(x['at'] for x in read(D/'nod_roll.json').values())
with locked(Q/'state.lock'):
 state=read(Q/'state.json');changes={};handles=[]
 try:
  for jid,failed in list(state['failed'].items()):
   if failed['at']<start:continue
   handle=acquire(Q/'claims'/f'{jid}.lock',False)
   if handle is None:continue
   handles.append(handle)
   assert not (O/'nod/completed'/f'{jid}.json').exists()
   changes[jid]=dict(previous_failure=failed,previous_attempts=state['attempts'].get(jid))
   del state['failed'][jid];state['attempts'].pop(jid,None)
   state.setdefault('deferred',{})[jid]=dict(at=time.time(),reason='early_crop_receipt_adapter_repaired')
  record=dict(at=time.time(),sources=len(changes),changes=changes,action='defer_for_normal_source_claims_no_output_rewrite')
  write(D/'early_queue_recovery.json',record);write(Q/'state.json',state)
 finally:
  for handle in handles:handle.close()
print('Recovered',len(changes),'source jobs; remaining failed',len(state['failed']))
