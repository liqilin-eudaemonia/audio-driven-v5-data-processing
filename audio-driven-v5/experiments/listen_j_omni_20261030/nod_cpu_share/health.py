import concurrent.futures,json,sys,time
from probe import remote,W,C
code_template='''import json,sys,time,socket
from pathlib import Path
w=Path(WORK);sys.path[:0]=[str(w.parents[1]/'scripts'),str(w.parents[1]/'src')]
from listen_benchmark.j_annotation import read
from listen_j_annotation_supervisor import alive,identity
c=read(w/'config.json');cfg=read(w/'nod_cpu_share/config.json');source=Path(cfg['source_work']);r=dict(node=NODE,at=time.time(),gpu=[])
sup=read(w/'nodes'/NODE/'supervisor.json')
for g in c['nodes'][NODE]['gpu_ids']:
 child=sup['children'][f'gpu-{g}'];h=read(w/'workers'/f'{NODE}-gpu-{g}.json');r['gpu'].append(dict(gpu=g,pid=child['pid'],start_ticks=child['start_ticks'],live=alive(child),fresh=time.time()-h['heartbeat']<60,errors=h.get('failed_clips'),completed=h.get('completed_clips'),stage=h['stage']))
if NODE=='node17':
 old=Path(c['nodes'][NODE]['old_work']);oldsup=read(old/'supervisor.json');r['original_nod']=[]
 for slot in range(read(old/'config.json')['nod_workers']):
  child=oldsup['children'][f'nod-{slot}'];h=read(old/'workers'/f'nod-{slot}.json')
  r['original_nod'].append(dict(slot=slot,live=alive(child),fresh=h['pid']==child['pid'] and time.time()-h['heartbeat']<60,errors=h['errors'],completed=h['completed_clips'],stage=h['stage']))
q=read(w/'visual_quality/nodes'/NODE/'controller.json');r['quality']=dict(live=alive(q),workers=q['workers'],age=time.time()-q['heartbeat'],stage=q['stage'],errors=q['execution_errors_this_process'],completed=q['completed_clips'])
if NODE in cfg['nodes']:
 local=w/'nod_cpu_share'/NODE
 if (local/'supervisor.json').exists():
  s=read(local/'supervisor.json');r['supervisor']=dict(live=alive(s),stage=s['stage'],age=time.time()-s['heartbeat'],blocked=s.get('restart_blocked',{}));r['workers']=[]
  for slot,child in s['children'].items():
   p=source/'workers'/f'nod-{slot}.json';h=read(p) if p.exists() else {}
   r['workers'].append(dict(slot=slot,pid=child['pid'],live=alive(child),fresh=h.get('pid')==child['pid'] and time.time()-h.get('heartbeat',0)<60,stage=h.get('stage'),completed=h.get('completed_clips',0),errors=h.get('errors',0),job_id=h.get('job')))
print(json.dumps(r))'''
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as p:
 fs={n:p.submit(remote,n,f'WORK={str(W)!r}\nNODE={n!r}\n'+code_template) for n in C['nodes']};nodes={n:f.result() for n,f in fs.items()}
out=dict(at=time.time(),nodes=nodes);label=sys.argv[1] if len(sys.argv)>1 else 'health';(W/'nod_cpu_share'/f'{label}.json').write_text(json.dumps(out,indent=2))
for n,r in nodes.items():print(n,json.dumps(dict(gpu_live=sum(x['live'] and x['fresh'] for x in r['gpu']),gpu_errors=sum(x['errors'] or 0 for x in r['gpu']),quality=r['quality'],supervisor=r.get('supervisor'),workers=len(r.get('workers',[])),live=sum(x['live'] and x['fresh'] for x in r.get('workers',[])),completed=sum(x['completed'] for x in r.get('workers',[])),errors=sum(x['errors'] for x in r.get('workers',[])),stages={v:sum(x['stage']==v for x in r.get('workers',[])) for v in set(x['stage'] for x in r.get('workers',[]))})))
