#!/usr/bin/env python3
from pathlib import Path
import sys,json,time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,sha,digest,atomic_json


def audit(root):
 spec=read(root/'pipeline.json');manifest=read(root/'view/COMPLETE.json');protocol=read(root/'plans/protocol.json')
 prepared=Path(spec['inputs']).parent/'PREPARED.json'
 if sha(prepared)!=spec['prepared_sha256'] or sha(spec['inputs'])!=read(prepared)['inputs_file_sha256']:raise ValueError('Original preparation manifest/inputs changed')
 def one(job):
  record=job['record'];sid=record['sample_id'];a=read(root/'recovery'/(sid+'.json'));item=read(root/'plans/samples'/(sid+'.json'));events=item['guidance']['source_record']['events']
  if digest(item)!=protocol['entries'][sid] or digest(events)!=a['events_sha256'] or a['original_record_sha256']!=digest(record):raise ValueError('Content/provenance mismatch')
  originals={e['event_id']:e for e in record['events'] if e['modality']=='visual'};safe=set(a['safe_source_ids']);fallback=set(a['fallback_source_ids'])
  if safe|fallback!=set(originals) or safe&fallback:raise ValueError('Visual source coverage lost')
  actual={e['event_id']:e for e in events}
  for sid0 in fallback:
   if actual.get(sid0)!=originals[sid0]:raise ValueError('Fallback altered original event')
  represented={sid0 for e in events if e['modality']=='visual' for sid0 in e.get('source_event_ids',[]) if e['event_id'] not in fallback}
  if represented!=safe:raise ValueError('Safe source coverage differs from output evidence')
  audio=lambda es:sorted((e for e in es if e['modality']=='audio'),key=lambda e:e['event_id'])
  if audio(events)!=audio(record['events']):raise ValueError('Audio changed')
  if a['review_policy']!='record_only':raise ValueError('Review still gates membership')
  return record['sample_id'],record['duration_s']
 ids=[];seconds=0.
 with ThreadPoolExecutor(max_workers=8) as pool:
  pending=deque()
  for line in Path(spec['inputs']).open():
   if not line.strip():continue
   pending.append(pool.submit(one,json.loads(line)))
   if len(pending)>=64:
    sid,d=pending.popleft().result();ids.append(sid);seconds+=d
  while pending:
   sid,d=pending.popleft().result();ids.append(sid);seconds+=d
 if len(ids)!=29975 or len(set(ids))!=29975 or set(ids)!=set(protocol['sample_ids']) or abs(seconds-manifest['duration_s'])>1e-6:raise ValueError('Full membership/duration mismatch')
 result=dict(success=True,samples=len(ids),duration_s=seconds,excluded_samples=0,checks=['prepared_input_sha','all_29975_unique_members','all_visual_sources_accounted_for','fallback_events_exactly_original','safe_output_evidence_coverage','all_audio_fields_unchanged','review_record_only','cache_item_entry_digest'],script_sha256=sha(__file__))
 atomic_json(root/'RESTORE_AUDIT.json',result,immutable=True);print(json.dumps(result),flush=True)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();audit(a.root.resolve())
