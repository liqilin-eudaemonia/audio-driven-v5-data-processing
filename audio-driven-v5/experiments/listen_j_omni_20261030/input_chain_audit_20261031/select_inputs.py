import json,sys,time,hashlib
from pathlib import Path
W=Path(__file__).resolve().parent.parent;R=W.parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from listen_benchmark.j_annotation import read,write,result_path
c=read(W/'config.json');selected=[];gate_counts=dict(retain=0,exclude=0,review=0,legacy=0);seen=set();sources=[]
for n,nc in c['nodes'].items():
 choices=[]
 for g in nc['gpu_ids']:
  h=read(W/'workers'/f'{n}-gpu-{g}.json');assert time.time()-h['heartbeat']<60 and not h['failed_clips']
  jid=h.get('job')
  if not jid or jid in seen:continue
  seen.add(jid);job=read(W/'jobs'/f'{jid}.json');sources.append(jid)
  quality=read(W/'visual_quality/measure/completed'/f'{jid}.json');ds={d['sample_id']:d for d in quality['decisions']}
  for row in job['rows']:
   d=ds[row['sample_id']];p=result_path(W,'caption',row['sample_id'])
   if p.exists():
    result=read(p)
    if result.get('visual_quality'):
     assert d==result['visual_quality'] and d['decision']=='retain';gate_counts['retain']+=1
     if result['node']==n:choices.append(dict(node=n,sample_id=row['sample_id'],job_id=jid,path=str(p),result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),silent=result['input_evidence']['listener_audio_is_silent']))
    else:gate_counts['legacy']+=1
   for b,decision in [('visual_excluded','exclude'),('visual_review','review')]:
    p=result_path(W,b,row['sample_id'])
    if p.exists():
     rr=read(p);assert rr['visual_quality']==d and d['decision']==decision;gate_counts[decision]+=1
  if len(choices)>=8:break
 assert len(choices)>=2,(n,len(choices))
 first=choices[0];second=next((x for x in choices if x['silent']!=first['silent']),choices[1]);selected.extend([first,second])
write(W/'input_chain_audit_20261031/selected.json',dict(at=time.time(),samples=selected,gate_counts=gate_counts,sources=sources,scope='Bounded current-production sources; two retained outputs per GPU host, including silent/non-silent when available.'))
print(json.dumps(dict(samples=len(selected),gate_counts=gate_counts,source_count=len(sources),silent=sum(x['silent'] for x in selected))))
