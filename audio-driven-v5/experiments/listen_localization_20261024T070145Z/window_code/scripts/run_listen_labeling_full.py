#!/usr/bin/env python3
"""Resumable source-backed listener annotation; each shard has exactly one writer."""
import argparse,datetime,fcntl,hashlib,json,multiprocessing,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import atomic_json,digest,sha
from p2_listen.annotations import union,intersection_length
from listen_labeling.full_source import prepare_source
from listen_labeling.client import prompt
from listen_labeling.guided import response_format
from listen_labeling.schema import parse,validate_response,canonical

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path,default=None):
 try:return json.loads(path.read_text())
 except FileNotFoundError:return default

def invoke(session,endpoint,branch,stage,duration,content,guided):
 payload=dict(model=endpoint['model'],messages=[dict(role='user',content=[dict(type='text',text=prompt(branch,duration,stage)),*content])],temperature=0,max_tokens=endpoint.get('max_tokens',1536),chat_template_kwargs=dict(enable_thinking=False))
 if guided:payload['response_format']=response_format(branch,duration,stage)
 record=dict(endpoint=endpoint,request_sha256=digest(payload),prompt=prompt(branch,duration,stage),attempts=[])
 semantic_retry_used=False
 for retry in range(3):
  record['request_sha256']=digest(payload)
  began=time.monotonic()
  try:
   r=session.post(endpoint['base_url'].rstrip('/')+'/chat/completions',json=payload,timeout=(15,endpoint.get('timeout_s',300)))
   record['attempts'].append(dict(status=r.status_code,body=r.text,seconds=time.monotonic()-began,request_sha256=record['request_sha256'],prompt=payload['messages'][0]['content'][0]['text']))
   if r.status_code in (408,429,500,502,503,504):
    if retry<2:time.sleep(2**retry);continue
   r.raise_for_status();body=r.json();choice=body['choices'][0]
   if choice.get('finish_reason')!='stop':raise ValueError('Truncated/non-stop response')
   record['parsed']=validate_response(parse(choice['message']['content']),branch,duration,stage);record['status']='schema_pass';record.pop('error',None);return record
  except Exception as exc:
   record['error']=dict(type=type(exc).__name__,message=str(exc));record['status']='failed'
   # Retry transport faults only, never turn malformed labels into implicit negatives.
   import requests
   if isinstance(exc,ValueError) and not semantic_retry_used and retry<2:
    semantic_retry_used=True
    record['attempts'][-1]['validation_error']=str(exc)
    payload['messages'][0]['content'][0]['text']=prompt(branch,duration,stage)+'\nYour preceding independent attempt failed validation: '+str(exc)+'. Reinspect the original supplied media and return a fresh response. reviewed means all inspected intervals, including intervals containing reported events, not only quiet spans. Every reported event must lie inside reviewed. All interval endpoints must be numeric and strictly increasing. Keep genuinely unjudgeable spans unknown; never invent events or certainty to satisfy the schema.'
    continue
   if isinstance(exc,(requests.Timeout,requests.ConnectionError)) and retry<2:
    record['attempts'].append(dict(transport_error=type(exc).__name__,seconds=time.monotonic()-began));time.sleep(2**retry);continue
   return record
 return record

def draft(row,duration,stages,media,algorithm):
 provenance=dict(human_verified=False,training_admitted=False,algorithm=algorithm,
  input_representation='source_member_timestamped_listener_frames_and_isolated_audio',source=row,
  video_payload_sha256=digest(media['visual']),audio_payload_sha256=media['audio']['audio']['wav_sha256'],
  model_requests={key:dict(endpoint=v['endpoint'],request_sha256=v['request_sha256']) for key,v in stages.items()})
 label=canonical(stages['audio.audit']['parsed'],'audio',row['sample_id'],duration,provenance)
 visual=canonical(stages['visual.audit']['parsed'],'visual',row['sample_id'],duration,provenance)
 for field in ['coverage','unknown']:label[field]['head_nod']=visual[field]['head_nod']
 label['events']+=visual['events'];label['events'].sort(key=lambda e:(e['start'],e['type']))
 support={}
 for name,spans in label['coverage'].items():
  unknown=label['unknown'][name]+[[e['start'],e['end']] for e in label['events'] if e['type']==name and e.get('boundary_censored')]
  support[name]=sum(b-a-intersection_length(unknown,a,b) for a,b in union(spans))
 return label,support

def worker(root_name,rank,total,stop_at):
 import requests
 root=Path(root_name);config=read(root/'production_config.json');out=root/'production';queue=root/'queue';session=requests.Session()
 code=digest({str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),*sorted((ROOT/'src/listen_labeling').glob('*.py'))]})
 algorithm=digest(dict(revision=config['revision'],media=config['media'],guided_json=config['guided_json'],models=config['model_revisions']))
 shard_paths=sorted((queue/'shards').glob('*.jsonl'));shard_paths=shard_paths[rank%len(shard_paths):]+shard_paths[:rank%len(shard_paths)]
 for path in shard_paths:
  if time.time()>=stop_at:break
  shard=path.stem;lock=(out/'locks'/f'{shard}.lock').open('a')
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:lock.close();continue
  try:
   statepath=out/'state'/f'{shard}.json';state=read(statepath,dict(shard=shard,offset=0,processed=0,schema_pass=0,failed=0,supervision_supported=0,done=False))
   if state['done']:continue
   if state.get('algorithm',algorithm)!=algorithm:raise ValueError('Shard algorithm mismatch; use a new versioned production directory')
   state.update(algorithm=algorithm,worker_pid=os.getpid(),rank=rank,heartbeat=utc());atomic_json(statepath,state)
   with path.open() as source:
    source.seek(state['offset'])
    while time.time()<stop_at:
     if rank>=read(out/'concurrency.json',{'workers':total})['workers']:return
     if (out/'PAUSE').exists():time.sleep(10);continue
     line=source.readline()
     if not line:state['done']=True;state['heartbeat']=utc();atomic_json(statepath,state);break
     next_offset=source.tell();row=json.loads(line);sid=row['sample_id'];destination=out/'results'/shard/(sid+'.json')
     state.update(current_sample=sid,heartbeat=utc());atomic_json(statepath,state)
     result=read(destination)
     if result is not None:
      if result['algorithm']!=algorithm or result['source']['source_row_sha256']!=row['source_row_sha256']:raise ValueError('Completed result identity mismatch')
     else:
      began=time.monotonic();inflight_path=out/'inflight'/f'{shard}.json'
      old=read(inflight_path,{})
      stages=old.get('stages',{}) if old.get('algorithm')==algorithm and old.get('sample_id')==sid and old.get('source_sha')==row['source_row_sha256'] else {}
      result=dict(schema='listen-labeling-source-batch-v1',sample_id=sid,source=row,algorithm=algorithm,code_sha256=code,training_admitted=False,stages=stages,started_at=utc())
      try:
       prepared=prepare_source(row,config['media']);duration=prepared['duration_s'];media={b:prepared[b][1] for b in ['visual','audio']}
       result.update(duration_s=duration,media=media)
       for stage in ['discovery','audit']:
        for branch in ['visual','audio']:
         key=branch+'.'+stage
         if stages.get(key,{}).get('status')=='schema_pass':continue
         endpoints=read(root/'production_config.json')['endpoints'][branch];endpoint=endpoints[rank%len(endpoints)]
         state.update(stage=key,heartbeat=utc());atomic_json(statepath,state)
         prior=stages.get(key)
         stages[key]=invoke(session,endpoint,branch,stage,duration,prepared[branch][0],config['guided_json'])
         if prior:stages[key]['previous_stage_attempts']=[prior]
         atomic_json(inflight_path,dict(sample_id=sid,source_sha=row['source_row_sha256'],algorithm=algorithm,stages=stages))
       result['status']='schema_pass' if all(stages[k]['status']=='schema_pass' for k in ['visual.discovery','audio.discovery','visual.audit','audio.audit']) else 'failed'
       if result['status']=='schema_pass':result['canonical_draft'],result['usable_seconds_by_class']=draft(row,duration,stages,media,algorithm)
      except Exception as exc:result.update(status='failed',preparation_or_fusion_error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()[-5000:]))
      result.update(ended_at=utc(),elapsed_seconds=time.monotonic()-began)
      atomic_json(destination,result,immutable=True)
     state['processed']+=1;state['schema_pass']+=result['status']=='schema_pass';state['failed']+=result['status']!='schema_pass'
     state['supervision_supported']+=result['status']=='schema_pass' and all(v>0 for v in result.get('usable_seconds_by_class',{}).values()) and bool(result.get('usable_seconds_by_class'))
     state.update(offset=next_offset,heartbeat=utc(),current_sample=None,stage=None);atomic_json(statepath,state)
  finally:lock.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=32);p.add_argument('--stop-at',type=float,required=True);a=p.parse_args()
 root=Path(a.root).resolve();out=root/'production'
 for name in ['locks','state','results','inflight']: (out/name).mkdir(parents=True,exist_ok=True)
 lock=(out/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if not (root/'queue'/'index.json').exists():raise ValueError('Index incomplete; refusing partial queue')
 atomic_json(out/'process.json',dict(pid=os.getpid(),command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],started_at=utc(),stop_at=a.stop_at,workers=a.workers))
 os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
 ctx=multiprocessing.get_context('spawn');children={};restarts={}
 while time.time()<a.stop_at:
  states=[read(p) for p in (out/'state').glob('*.json')]
  if len(states)==len(list((root/'queue'/'shards').glob('*.jsonl'))) and all(s['done'] for s in states):break
  worker_limit=max(1,min(64,read(out/'concurrency.json',{'workers':a.workers})['workers']))
  for rank in range(worker_limit):
   child=children.get(rank)
   if child is not None and child.is_alive():continue
   recent=[t for t in restarts.get(rank,[]) if time.time()-t<3600]
   if child is not None:
    child.join();recent.append(time.time());restarts[rank]=recent
   if len(recent)>3:
    atomic_json(out/'worker_alert.json',dict(rank=rank,reason='repeated worker exits',utc=utc()));continue
   child=ctx.Process(target=worker,args=(str(root),rank,a.workers,a.stop_at));child.start();children[rank]=child
  atomic_json(out/'supervisor_heartbeat.json',dict(utc=utc(),workers={rank:dict(pid=c.pid,alive=c.is_alive(),exitcode=c.exitcode) for rank,c in children.items()},stop_at=a.stop_at))
  time.sleep(10)
 for child in children.values():child.join(timeout=1)
 atomic_json(out/'supervisor_finished.json',dict(utc=utc(),note='Deadline or queue completion reached; in-flight requests may finish and commit.'))
if __name__=='__main__':main()
