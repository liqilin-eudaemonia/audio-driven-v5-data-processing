import json,sys,time,hashlib
from pathlib import Path
W=Path(__file__).resolve().parent.parent;R=W.parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
from listen_benchmark.j_annotation import read,write,result_path
import listen_j_annotation_nod as original
from run_glare_nodding_pilot import module,FACE,NOD,MODEL,PROFILES,intervals
node=sys.argv[1];c=read(W/'config.json');old=Path(c['nodes']['node17']['old_work']);resource=Path(c['nodes'][node]['resource_work'])
face=module('parity_face',FACE);nod=module('parity_nod',NOD)
items=read(W/'nod_cpu_share/parity_inputs.json');results=[]
for item in items:
 row=item['row'];reference=read(result_path(old,'nod',row['sample_id']));t=time.time()
 actual=original.annotate(row,face,nod,PROFILES['recall'],MODEL,intervals,resource,fast_crop=True)
 keys=['sample_id','source','status','fps','frames','parameters','model_sha256','pose','scores','events','media']
 same={k:actual.get(k)==reference.get(k) for k in keys}
 results.append(dict(sample_id=row['sample_id'],seconds=time.time()-t,same=same,passed=all(same.values())))
import importlib.metadata
out=dict(node=node,at=time.time(),passed=all(r['passed'] for r in results),results=results,packages={k:importlib.metadata.version(k) for k in ['av','numpy','mediapipe','opencv-contrib-python','pillow','tqdm']})
write(W/'nod_cpu_share'/f'parity-{node}.json',out);print(json.dumps(out));assert out['passed']
