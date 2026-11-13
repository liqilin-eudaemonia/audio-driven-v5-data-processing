import sys,tempfile,time,os,hashlib
from pathlib import Path
R=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from listen_benchmark.j_annotation import read,write
from listen_benchmark.full_raw import prepare
from listen_benchmark.media import uniform_video
from listen_benchmark.j_omni_crop import _measure,_services
from listen_benchmark.j_omni import fingerprint
W=R/'experiments/listen_j_omni_20261030';D=W/'input_speed_v1_20261031';C=read(W/'config.json');results=[]
items=read(W/'corrected_crop_rollout_20261031/deployment_audit.json')['new_samples'];selected=[x for x in items if x['crop_changed']][:4]+[x for x in items if not x['crop_changed']][:2]
for i,item in enumerate(selected):
 r=read(item['path']);row=dict(sample_id=r['sample_id'],source=r['source']);out=D/'shared_replay_guarded'/str(i);out.mkdir(parents=True,exist_ok=True);write(out/'row.json',row)
 with tempfile.TemporaryDirectory(prefix='j-shared-parity-',dir='/dev/shm') as tmp:
  t=time.monotonic();before=prepare(row['source'],'visual');old_seconds=time.monotonic()-t
  t=time.monotonic();after=prepare(row['source'],'visual',fast_crop=True,native_dump=tmp);new_seconds=time.monotonic()-t
  a,ae=uniform_video(before,8);b,be=uniform_video(after,8)
  assert (a==b).all() and before['meta']==after['meta'] and ae==be
  settings=C['bounded_crop_input'];res=C['nodes']['node12']['resource_work']
  t=time.monotonic();_measure(dict(row=str(out/'row.json'),output=str(out/'independent_nod.json'),resources=res),settings,C,res,Path(settings['nod_cache']));nod_seconds=time.monotonic()-t
  shared=(Path(tmp)/'manifest.json').exists();req=dict(row=str(out/'row.json'),output=str(out/'shared_nod.json'),resources=res)
  if shared:req['decoded_dir']=tmp
  t=time.monotonic();_measure(req,settings,C,res,Path(settings['nod_cache']));shared_seconds=time.monotonic()-t
  old=read(out/'independent_nod.json');new=read(out/'shared_nod.json')
  for k in ['source','media','pose','scores','events','status','model_sha256','parameters']:assert old[k]==new[k],(row['sample_id'],k)
  result=dict(shared_decode_used=shared,sample_id=row['sample_id'],crop_changed=item['crop_changed'],source_size=row['source']['video_transform']['source_size'],duration=row['source']['duration_s'],pixels_equal=True,metadata_equal=True,nod_equal=True,independent_seconds=old_seconds+nod_seconds,shared_seconds=new_seconds+shared_seconds,video_sha256=old['media']['video_sha256'],model_frames_sha256=hashlib.sha256(a).hexdigest())
  results.append(result);write(D/'shared_replay_guarded.json',dict(passed=True,samples=results,complete=len(results)==len(selected)));print(result,flush=True)
for service in _services.values():service['process'].stdin.close();service['process'].wait(timeout=30)
