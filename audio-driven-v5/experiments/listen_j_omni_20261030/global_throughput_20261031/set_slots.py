"""Hot-adjust only CPU nod concurrency; retain every input and model setting."""
import sys,time,json,hashlib
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent;R=W.parents[1]
sys.path.insert(0,str(R/'src'))
from listen_benchmark.j_annotation import read,write,locked
value=int(sys.argv[1]);assert value in (16,24,32)
with locked(W/'config-edit.lock'):
    cfg=read(W/'config.json');old=cfg['bounded_crop_input']['nod_slots_per_host']
    cfg['bounded_crop_input']['nod_slots_per_host']=value
    write(W/'config.json',cfg)
    record=dict(at=time.time(),old=old,new=value,changed_field='bounded_crop_input.nod_slots_per_host')
    write(D/f'change-{time.time_ns()}.json',record)
print(json.dumps(record))
