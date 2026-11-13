import sys,time
from pathlib import Path
R=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from listen_benchmark.j_annotation import read,write,result_path
from listen_benchmark.j_visual_quality_schedule import ensure_nod_quality
from listen_benchmark.j_omni_crop import effective_row,_measure,_services
W=R/'experiments/listen_j_omni_20261030';D=W/'corrected_crop_rollout_20261031';C=read(W/'config.json');O=Path(C['nodes']['node17']['old_work'])
failed=read(D/'upstream_issue.json')['failed_snapshot'];selected=None
for jid in failed:
 if (Path(C['visual_quality_gate']['work'])/'measure/completed'/f'{jid}.json').exists():continue
 job=read(O/'jobs'/f'{jid}.json');quality=ensure_nod_quality(C['visual_quality_gate'],O/'jobs'/f'{jid}.json')
 for row in job['rows']:
  d=quality[row['sample_id']]
  if d['decision']!='retain':continue
  applied,prov=effective_row(row,C,decision=d)
  if prov['coordinates_changed']:selected=(row,applied,prov);break
 if selected:break
assert selected
row,applied,prov=selected;request=D/'early_replay_request.json';target=D/'early_replay_nod.json';write(request,applied)
_measure(dict(row=str(request),output=str(target),resources=C['nodes']['node12']['resource_work']),C['bounded_crop_input'],C,C['nodes']['node12']['resource_work'],Path(C['bounded_crop_input']['nod_cache']))
nod=read(target);assert nod['source']==applied['source'] and nod['media']['video_transform']==applied['source']['video_transform']
for service in _services.values():service['process'].stdin.close();service['process'].wait(timeout=30)
write(D/'early_replay.json',dict(passed=True,job_id=jid,sample_id=row['sample_id'],downstream_receipt_absent=True,crop=prov,status=nod['status']));print('PASSED',row['sample_id'])
