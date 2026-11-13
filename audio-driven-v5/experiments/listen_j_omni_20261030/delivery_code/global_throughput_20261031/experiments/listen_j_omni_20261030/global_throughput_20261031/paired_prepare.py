"""Matched input-only ABBA; complements, never replaces, 32-GPU windows."""
import sys,time,tempfile,json
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent;R=W.parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from listen_benchmark.j_annotation import read,write
from listen_benchmark.full_raw import prepare
from listen_benchmark.j_omni_crop import _measure,_services
cfg=read(W/'config.json');res=cfg['nodes']['node12']['resource_work'];records=[]
# Warm the persistent nod runtime outside the four timed phases.
source=W/'input_speed_v1_20261031/shared_replay_guarded'
warm=D/'paired_prepare/warm.json';warm.parent.mkdir(exist_ok=True)
_measure(dict(row=str(source/'2/row.json'),output=str(warm),resources=res),cfg['bounded_crop_input'],cfg,res,Path(cfg['bounded_crop_input']['nod_cache']))
for phase,roi in enumerate([False,True,True,False]):
 rows=[];start=time.monotonic()
 for index in [0,2]:
  row=read(source/str(index)/'row.json');out=D/'paired_prepare'/f'{phase}-{index}.json'
  with tempfile.TemporaryDirectory(prefix='j-paired-input-',dir='/dev/shm') as tmp:
   tick=time.monotonic();m=prepare(row['source'],'visual',fast_crop=True,native_dump=tmp,raw_native_dump=roi,roi_crop=roi)
   req=dict(row=str(source/str(index)/'row.json'),output=str(out),resources=res)
   if (Path(tmp)/'manifest.json').exists():req['decoded_dir']=tmp
   _measure(req,cfg['bounded_crop_input'],cfg,res,Path(cfg['bounded_crop_input']['nod_cache']))
   elapsed=time.monotonic()-tick;a=read(source/str(index)/'independent_nod.json');b=read(out)
   for k in ['media','pose','scores','events','source']:assert a[k]==b[k],(phase,index,k)
   rows.append(dict(index=index,seconds=elapsed,jpegs=[f['jpeg_sha256'] for f in m['meta']['video']['frames']]))
 record=dict(phase=phase,roi=roi,seconds=time.monotonic()-start,samples=rows);records.append(record);write(D/'paired_prepare_progress.json',records);print(json.dumps({k:v for k,v in record.items() if k!='samples'}),flush=True)
for k in range(2):assert len({tuple(r['samples'][k]['jpegs']) for r in records})==1
old=sum(r['seconds'] for r in records if not r['roi']);new=sum(r['seconds'] for r in records if r['roi'])
write(D/'paired_prepare_result.json',dict(old_seconds=old,new_seconds=new,speedup=old/new,all_inputs_equal=True,scope='input preparation only; four-host production measured separately',phases=records))
for service in _services.values():service['process'].stdin.close();service['process'].wait(timeout=30)
