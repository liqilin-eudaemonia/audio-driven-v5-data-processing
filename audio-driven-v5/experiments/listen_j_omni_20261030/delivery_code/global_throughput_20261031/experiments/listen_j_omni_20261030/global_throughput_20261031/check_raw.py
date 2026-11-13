import sys,tempfile,time,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent;R=W.parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from listen_benchmark.j_annotation import read,write
from listen_benchmark.full_raw import prepare
from listen_benchmark.j_omni_crop import _measure,_services
C=read(W/'config.json');results=[]
for i in [1,0,2,3,4,5]:
 src=W/'input_speed_v1_20261031/shared_replay_guarded'/str(i);row=read(src/'row.json');out=D/'raw_parity'/str(i);out.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='j-raw-parity-',dir='/dev/shm') as tmp:
  start=time.monotonic();media=prepare(row['source'],'visual',fast_crop=True,native_dump=tmp,raw_native_dump=True);prep_s=time.monotonic()-start
  original=prepare(row['source'],'visual',fast_crop=True)
  assert media['meta']==original['meta'];assert all(a.tobytes()==b.tobytes() for a,b in zip(media['images'],original['images']))
  manifest=read(Path(tmp)/'manifest.json') if (Path(tmp)/'manifest.json').exists() else None
  req=dict(row=str(src/'row.json'),output=str(out/'nod.json'),resources=C['nodes']['node12']['resource_work'])
  if manifest:req['decoded_dir']=tmp
  t=time.monotonic();_measure(req,C['bounded_crop_input'],C,req['resources'],Path(C['bounded_crop_input']['nod_cache']));nod_s=time.monotonic()-t
  old=read(src/'independent_nod.json');new=read(out/'nod.json')
  for k in ['source','media','pose','scores','events','status','model_sha256','parameters']:assert old[k]==new[k],(i,k)
  r=dict(index=i,sample_id=row['sample_id'],schema=manifest['schema'] if manifest else None,bytes=manifest.get('bytes') if manifest else None,prep_s=prep_s,nod_s=nod_s,passed=True);results.append(r);write(D/'raw_parity.json',dict(complete=len(results)==6,samples=results));print(json.dumps(r),flush=True)
for s in _services.values():s['process'].stdin.close();s['process'].wait(timeout=30)
