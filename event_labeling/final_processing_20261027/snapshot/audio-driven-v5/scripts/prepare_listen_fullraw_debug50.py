"""Freeze 50 fully completed raw samples and adapt existing text to semantic/fusion inputs."""
import argparse,collections,hashlib,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.full_raw import views

def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=20261025);a=p.parse_args();run=a.run.resolve();out=a.out.resolve()
 if out.exists():raise ValueError('Use new output')
 branches=['visual','audio','asr'];common=[]
 for f in sorted((run/'raw/visual/state').glob('*.json')):
  if all((run/f'raw/{b}/state/{f.name}').exists() and json.loads((run/f'raw/{b}/state/{f.name}').read_text()).get('done') for b in branches):common.append(f.stem)
 rng=random.Random(a.seed);rng.shuffle(common);selected=[];seen_pairs=set();audit=[];frozen={b:[] for b in branches}
 old=json.loads((ROOT/'experiments/listen_existing_fusion_verbatim_20261025/review.json').read_text());exclude={s['id'] for s in old['samples']}
 for shard in common:
  records={};latest={}
  for b in branches:
   records[b]=collections.defaultdict(list);latest[b]=collections.defaultdict(dict)
   f=run/f'raw/{b}/results/{shard}.jsonl'
   with f.open('rb') as stream:
    for line_no,line in enumerate(stream,1):
     d=json.loads(line);sid=d.get('sample_id')
     if not sid:continue
     records[b][sid].append((line_no,line,d))
     if d.get('record_kind')=='result':latest[b][sid][d['request_id']]=d
  eligible=[]
  for sid,rv in latest['visual'].items():
   if sid in exclude:continue
   full=rv.get(sid+'@full')
   if not full:continue
   source=full['source'];duration=full['parent_duration_s']
   if not 2<=duration<=8.04 or source['crop_proposal']['size']<384:continue
   expected={sid+'@'+name for name,_,_ in views(duration)}
   if any(set(latest[b].get(sid,{}))!=expected for b in branches):continue
   if any(r.get('status')!=('complete' if b=='asr' else 'schema_pass') for b in branches for r in latest[b][sid].values()):continue
   if any(r['source']['source_line_sha256']!=source['source_line_sha256'] or r['parent_duration_s']!=duration for b in branches for r in latest[b][sid].values()):continue
   eligible.append(sid)
  rng.shuffle(eligible);taken=[]
  for sid in eligible:
   source=latest['visual'][sid][sid+'@full']['source']
   if source['pair'] in seen_pairs:continue
   seen_pairs.add(source['pair']);taken.append(sid)
   selected.append(dict(sample_id=sid,shard=shard,source=source,duration_s=latest['visual'][sid][sid+'@full']['parent_duration_s']))
   for b in branches:
    frozen[b].extend(records[b][sid])
   if len(taken)>=10 or len(selected)>=50:break
  audit.append(dict(shard=shard,eligible=len(eligible),selected=taken));print(json.dumps(audit[-1]),flush=True)
  if len(selected)>=50:break
 assert len(selected)==50
 out.mkdir(parents=True);raw=out/'raw';raw.mkdir();inputs=out/'open_input';inputs.mkdir();align=out/'word_alignment';align.mkdir()
 lookup={s['sample_id']:s for s in selected};receipts=[]
 for b in branches:
  with (raw/(b+'.jsonl')).open('wb') as f:
   for no,line,d in frozen[b]:
    f.write(line);receipts.append(dict(branch=b,sample_id=d['sample_id'],request_id=d['request_id'],kind=d['record_kind'],source_file=str(run/f'raw/{b}/results/{lookup[d["sample_id"]]["shard"]}.jsonl'),source_line_1based=no,sha256=sha(line)))
 latest={b:{} for b in branches}
 for b in branches:
  for _,_,d in frozen[b]:
   if d['record_kind']=='result':latest[b][d['request_id']]=d
 labels=[];provenance={};media=[]
 for row in selected:
  sid=row['sample_id'];events=[];candidates=[];origins={}
  for b in ['visual','audio']:
   for rid,d in sorted(latest[b].items()):
    if d['sample_id']!=sid:continue
    for i,e in enumerate(d['parsed']['events']):
     eid=f'{b}-{d["view"]}-{i:04d}';cid='source-'+eid;start=round(d['window_start_s']+e['start'],9);end=round(d['window_start_s']+e['end'],9)
     assert 0<=start<end<=row['duration_s']
     event=dict(event_id=eid,start=start,end=end,modality=b,behavior=e['behavior'],review_flags=['raw_model_unverified'])
     src=dict(start=start,end=end,modality=b,behavior=e['behavior'],source='qwen3_vl' if b=='visual' else 'omni_audio',window_id=rid,window=dict(start_s=d['window_start_s'],end_s=d['window_end_s']),request_id=rid)
     candidates.append(dict(id=cid,start=start,end=end,modality=b,behavior=e['behavior'],source=src['source'],sources=[src],valid=True,human_verified=False))
     origins[eid]=dict(source_candidate_ids=[cid],boundary_method='raw_observed_interval_no_union');events.append(event)
  labels.append(dict(sample_id=sid,duration_s=row['duration_s'],time_origin='clip_start_seconds',events=events,human_ground_truth=False,training_admitted=False))
  provenance[sid]=dict(source_candidates=candidates,events=origins,excluded=[])
  media.append(dict(id=sid,duration=row['duration_s'],video=f'../debug50_fullraw_media/{sid}/video.mp4',audio=f'../debug50_fullraw_media/{sid}/listener.wav'))
  for rid,d in sorted(latest['asr'].items()):
   if d['sample_id']!=sid:continue
   record=dict(d['raw_response'],sample_id=sid if d['view']=='full' else sid+'__'+d['view'],status=d['status'],input_evidence={} if d['view']=='full' else dict(window=dict(start_s=d['window_start_s'],end_s=d['window_end_s'])),raw_request_id=rid)
   dump(align/(record['sample_id']+'.json'),record)
 (inputs/'labels.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in labels));dump(inputs/'provenance.json',dict(samples=provenance));dump(inputs/'input_review.json',dict(samples=media))
 dump(out/'selection.json',dict(seed=a.seed,scope='debug_train_only_not_representative_evaluation',samples=selected,shards=audit,previous50_excluded=True,all_required_views_successful=True,raw_run=str(run)))
 dump(out/'raw_receipts.json',receipts)
 print(json.dumps(dict(out=str(out),samples=len(selected),events=sum(len(x['events']) for x in labels),alignment_records=len(list(align.glob('*.json'))))))
if __name__=='__main__':main()
