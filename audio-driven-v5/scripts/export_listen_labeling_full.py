#!/usr/bin/env python3
"""Publish auditable snapshots; incomplete and unreviewed labels remain explicit."""
import argparse,collections,datetime,fcntl,json,os,sys,time
from pathlib import Path

def write(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def emit(handle,value):handle.write(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n')
def snapshot(root,remaining=False):
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=root/'exports'/stamp;out.mkdir(parents=True)
 files=list((root/'production'/'results').glob('*/*.json'));processed=set();counts=collections.Counter();split=collections.Counter();began=time.monotonic()
 handles={name:(out/(name+'.jsonl')).open('w',buffering=1024*1024) for name in ['schema_pass','supported_both_classes','failed','open_events','remaining']}
 try:
  for path in files:
   row=json.loads(path.read_text());sid=row['sample_id'];processed.add(sid);source=row['source'];counts['processed']+=1;split[source['split']]+=1
   base=dict(sample_id=sid,split=source['split'],result_file=str(path.resolve()),source_manifest=source['source_manifest'],source_line_1based=source['source_line_1based'],listener_id=source['listener_id'],source_start_s=source['source_start_s'],source_end_s=source['source_end_s'],video_transform=source['video_transform'],training_admitted=False,human_verified=False)
   if row['status']=='schema_pass':
    counts['schema_pass']+=1;support=row['usable_seconds_by_class'];label=dict(base,usable_seconds_by_class=support,canonical_draft=row['canonical_draft'])
    emit(handles['schema_pass'],label)
    if all(v>0 for v in support.values()):counts['supported_both_classes']+=1;emit(handles['supported_both_classes'],label)
    else:counts['schema_pass_without_both_class_support']+=1
   else:
    counts['failed']+=1;emit(handles['failed'],dict(base,preparation_or_fusion_error=row.get('preparation_or_fusion_error'),errors={k:v.get('error') for k,v in row['stages'].items() if v['status']!='schema_pass'}))
   observations={branch:row['stages'][branch+'.discovery']['parsed'] for branch in ['visual','audio'] if row['stages'].get(branch+'.discovery',{}).get('status')=='schema_pass'}
   if observations:emit(handles['open_events'],dict(base,complete_sample=row['status']=='schema_pass',observations=observations))
  if remaining:
   for path in sorted((root/'queue'/'shards').glob('*.jsonl')):
    with path.open() as f:
     for line in f:
      row=json.loads(line)
      if row['sample_id'] in processed:continue
      counts['remaining']+=1;emit(handles['remaining'],dict(sample_id=row['sample_id'],split=row['split'],queue_shard=str(path.resolve()),source_manifest=row['source_manifest'],source_line_1based=row['source_line_1based'],reason='not_in_this_completed_result_snapshot'))
 finally:
  for handle in handles.values():handle.close()
 index=json.loads((root/'queue'/'index.json').read_text())
 if not remaining:counts['remaining']=index['counts']['eligible']-counts['processed']
 summary=dict(snapshot_utc=stamp,counts=counts,processed_by_split=split,eligible=index['counts']['eligible'],excluded_no_crop=index['counts']['excluded'],remaining_manifest_written=remaining,
  remaining_manifest_note=None if remaining else 'Periodic snapshot: remaining count only; full remaining manifest is emitted in the deadline snapshot.',
  human_accuracy_evaluated=False,training_admitted=False,elapsed_seconds=time.monotonic()-began,directory=str(out.resolve()),
  note='schema_pass is structural validity, not verified event accuracy; supported_both_classes means model-claimed nonzero inspected support outside unknown/censored intervals.')
 write(out/'summary.json',summary);write(root/'exports'/'latest.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True);return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--remaining',action='store_true');p.add_argument('--watch',action='store_true');p.add_argument('--deadline',type=float);a=p.parse_args();root=Path(a.root).resolve();(root/'exports').mkdir(exist_ok=True)
 lock=(root/'exports'/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if not a.watch:snapshot(root,a.remaining);return
 if not a.deadline:raise ValueError('watch requires deadline')
 write(root/'exports'/'process.json',dict(pid=os.getpid(),deadline=a.deadline,interval_seconds=1200,command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]]))
 while time.time()<a.deadline:
  if (root/'queue'/'index.json').exists():
   try:snapshot(root)
   except Exception as exc:write(root/'exports'/'error.json',dict(error=str(exc),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
  time.sleep(min(1200,max(0,a.deadline-time.time())))
 snapshot(root,remaining=True)
if __name__=='__main__':main()
