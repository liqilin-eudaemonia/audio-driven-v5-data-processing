import json,os,sys,time
from pathlib import Path
W=Path(__file__).resolve().parent.parent;R=W.parents[1];sys.path.insert(0,str(R/'src'))
from listen_benchmark.j_annotation import write
c=json.loads((W/'config.json').read_text())
write(W/'nod_cpu_share/config.json',dict(schema='j-nod-cpu-share-v1',created_at=time.time(),source_work=c['nodes']['node17']['old_work'],python=str(W/'nod_cpu_share/runtime_env/bin/python'),nodes={n:dict(workers=0,max_workers=m,slot_base=b) for n,m,b in [('node8',80,10000),('node4',64,20000),('node12',64,30000)]}))
for n in ['node4','node12']:
 p=Path(c['nodes'][n]['resource_work'])/'selected.json'
 assert not p.exists();write(p,dict(decode_slots=c['decode_slots'],purpose='Borrowed nod workers share existing host-local Omni decoder locks'))
write(W/'nod_cpu_share/initial_counts.json',dict(at=time.time(),caption=json.loads((W/'queues/caption/state.json').read_text()),nod=json.loads((Path(c['nodes']['node17']['old_work'])/'queues/nod/state.json').read_text()),quality=json.loads((W/'visual_quality/queues/measure/state.json').read_text())))
