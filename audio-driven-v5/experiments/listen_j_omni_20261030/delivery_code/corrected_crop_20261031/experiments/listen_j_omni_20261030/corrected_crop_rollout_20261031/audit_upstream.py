import sys,time,collections
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent;R=W.parents[1];sys.path.insert(0,str(R/'src'))
from listen_benchmark.j_annotation import read,write,result_path
C=read(W/'config.json');O=Path(C['nodes']['node17']['old_work']);groups={'node17':range(48),'node8':range(10000,10048),'node4':range(20000,20032),'node12':range(30000,30040)}
summary={};samples=[]
for n,slots in groups.items():
 states=[read(O/'workers'/f'nod-{i}.json') for i in slots];new=[s for s in states if s.get('revision')==6]
 summary[n]=dict(expected=len(states),revision_counts=dict(collections.Counter(x.get('revision') for x in states)),corrected_active=sum(s.get('crop_revision')=='quality-bounded-crop-v1-20261030' for s in new),new_completed=sum(s['completed_clips'] for s in new),new_errors=sum(s['errors'] for s in new))
 assert new and summary[n]['new_errors']==0,n
 for state in sorted(new,key=lambda x:-x['completed_clips'])[:4]:
  if not state.get('job'):continue
  job=read(O/'jobs'/f"{state['job']}.json")
  for row in job['rows']:
   p=result_path(O,'nod',row['sample_id'])
   if not p.exists():continue
   result=read(p)
   if not result.get('crop_provenance',{}).get('coordinates_changed'):continue
   assert result['original_source']==row['source']
   assert result['source']['video_transform']['crop']==result['crop_provenance']['actual_crop']==result['media']['video_transform']['crop']
   samples.append(dict(node_worker_group=n,sample_id=row['sample_id'],path=str(p),crop=result['crop_provenance']['actual_crop']));break
  else:continue
  break
q=read(O/'queues/nod/state.json');re=read(D/'early_queue_recovery.json')
outcomes={j:'completed' if (O/'nod/completed'/f'{j}.json').exists() else 'active' if j in q['inflight'] else 'deferred' if j in q.get('deferred',{}) else 'failed' if j in q['failed'] else 'unknown' for j in re['changes']}
assert not q['failed'];assert all(v in ['completed','active','deferred'] for v in outcomes.values())
assert samples
out=dict(at=time.time(),passed=True,nodes=summary,corrected_samples=samples,remaining_failed_sources=len(q['failed']),recovered_source_states=dict(collections.Counter(outcomes.values())),recovered_source_details=outcomes,early_receipt_replay=read(D/'early_replay.json'),note='Older CPU workers may still be draining their already-owned source; corrected-caption consumers independently enforce exact crop consistency.')
write(D/'upstream_audit.json',out);print({k:out[k] for k in ['passed','nodes','remaining_failed_sources','recovered_source_states']})
