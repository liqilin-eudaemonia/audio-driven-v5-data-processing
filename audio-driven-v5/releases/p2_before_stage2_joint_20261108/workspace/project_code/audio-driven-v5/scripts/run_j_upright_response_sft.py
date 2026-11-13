#!/usr/bin/env python3
"""Frozen-policy, GPU7-only screening of the current full listener captions.

This is a new run adapter. Historical scripts and ongoing training are untouched.
"""
import argparse
import collections
import contextlib
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
WORK=ROOT/'experiments/listen_j_upright_response_sft_20261105'
LABELS=ROOT/'experiments/listen_j_labels_upright_20261105/labels.jsonl'
EXPECTED='0851714c52e17dec6c7b5145b82dc53998ab09af0cc7ceda0b9c807192c2b389'
COUNTS={'train':1086207,'dev':8822,'test':12284}
UUID='GPU-d5cb2cac-0f7a-e5f8-1870-c2efcae7aedc'
PY='/private/software/conda/envs/vllm/bin/python'
SPLITS=('dev','test','train')

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def ensure(ok,msg):
 if not ok:raise ValueError(msg)
def line(v):return (json.dumps(v,ensure_ascii=False)+'\n').encode()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def prepare():
 WORK.mkdir(exist_ok=True);(WORK/'logs').mkdir(exist_ok=True)
 if (WORK/'PREPARED.json').exists():
  ensure(read(WORK/'PREPARED.json')['labels_sha256']==EXPECTED,'Wrong prepared input');return
 save(WORK/'RUN_STATUS.json',dict(stage='preparing_full_reference_index',at=time.time()))
 vendor=WORK/'vendor';vendor.mkdir(exist_ok=True)
 # Exact historical policy/parser snapshots. Worker hook adds the historical
 # short-quote and sentence-enum retries before committing any unresolved result.
 for name in ['filter_full_response_sft.py','recover_voiced_response_sft.py','retry_prefix_quote_failures.py']:
  shutil.copy2(ROOT/'scripts'/name,vendor/name)
 base=vendor/'filter_full_response_sft.py';code=base.read_text()
 old='                elapsed=time.monotonic()-t\n'
 ensure(code.count(old)==1,'Worker hook location changed')
 code=code.replace(old,'                from quote_retry_hook import recover\n                recover(llm,tokenizer,prompt,batch,results,cfg,schema,parse_answer,stage)\n'+old)
 base.write_text(code)
 shutil.copy2(ROOT/'scripts/j_upright_sft_quote_retry.py',vendor/'quote_retry_hook.py')
 shutil.copy2(Path(__file__),vendor/'controller.py')
 source=WORK/'source';source.mkdir(exist_ok=True)
 oldroot=ROOT/'experiments/listen_j_full_response_sft_20261101'
 diskdb=sqlite3.connect(f'file:{oldroot}/index.sqlite?mode=ro',uri=True)
 olddb=sqlite3.connect(':memory:');diskdb.backup(olddb);diskdb.close();olddb.row_factory=sqlite3.Row
 avroot=ROOT/'experiments/listen_j_upright_av500h_train_dev_v3_20261105/selection'
 avdb=sqlite3.connect(f'file:{avroot}/index.sqlite?mode=ro',uri=True)
 av_ids={r[0] for r in avdb.execute('SELECT sid FROM samples')};avdb.close()
 dbfile=source/'index.sqlite';dbfile.unlink(missing_ok=True);db=sqlite3.connect(dbfile)
 db.execute('PRAGMA journal_mode=MEMORY');db.execute('PRAGMA synchronous=OFF')
 db.execute('CREATE TABLE samples(sid TEXT PRIMARY KEY,split TEXT,pair TEXT,listener TEXT,group_id TEXT,start REAL,end REAL,duration REAL,face_p10 REAL,file TEXT,offset INTEGER,bytes INTEGER)')
 counts=collections.Counter();hours=collections.Counter();groups={};prefix=collections.Counter();audio=collections.Counter();canaries={};digest=hashlib.sha256();pending=[]
 with contextlib.ExitStack() as stack:
  outputs={s:stack.enter_context((source/(s+'.jsonl')).open('wb')) for s in SPLITS}
  f=stack.enter_context(LABELS.open('rb'))
  while True:
   off=f.tell();raw=f.readline()
   if not raw:break
   digest.update(raw);r=json.loads(raw);sid=r['sample_id'];s=r['split'];src=r['source']
   ensure(s in COUNTS and src['split']==s and src['sample_id']==sid,'Label identity differs')
   ensure(r['finish_reason']=='stop' and 'repeated_generation' not in r['review_flags'],'Unrepaired generation')
   ensure(type(r['listener_audio_is_silent']) is bool,'Unknown processed waveform status')
   ensure(src['listener_id']==r['listener_id'] and src['speaker_id']==r['speaker_id'] and r['listener_id']!=r['speaker_id'],'Wrong participant')
   old=olddb.execute('SELECT * FROM samples WHERE sid=?',(sid,)).fetchone();ensure(old is not None,'Missing historical full-pool lineage')
   for k,v in [('split',s),('pair',r['pair']),('listener',r['listener_id']),('group_id',r['group_id']),('start',r['source_start_s']),('end',r['source_end_s']),('duration',r['duration_s'])]:ensure(old[k]==v,'Original split/time/role changed: '+k)
   group=r['group_id'];ensure(group not in groups or groups[group]==s,'Group crosses source splits');groups[group]=s
   ensure(r['duration_s']==src['duration_s'] and abs(r['duration_s']-(r['source_end_s']-r['source_start_s']))<=1/r['audio_recipe']['source']['sample_rate']+1e-8,'Duration differs beyond native audio sample precision')
   prefix[s]+=int(re.match(r'^\s*Assistant\s*:\s*',r['caption'],re.I) is not None)
   row=dict(schema='j-upright-response-sft-reference-v1',sample_id=sid,split=s,pair=r['pair'],group_id=group,duration_s=r['duration_s'],face=dict(short_side_p10_px=r['face_p10'],short_side_p50_px=r['face_p50']),selection_reason='all_latest_quality_retained_no_face_or_av500h_gate',
    inputs=dict(speaker_audio=dict(participant_id=r['speaker_id'],source_result_file=src['source_result'],source_result_sha256=src['source_result_sha256'],clip_index=src['source_clip_index'],start_s=r['source_start_s'],end_s=r['source_end_s'],role='speaker',processing='paired_normalized_track_no_listener_gate'),listener_first_frame=dict(participant_id=r['listener_id'],source=src['source'],source_time_s=r['source_start_s'],video_transform=src['video_transform'],frame_policy='nearest_source_pts_to_pair_start',output_size=[512,512])),
    target=dict(text=r['caption'],format='original_caption_verbatim',source_file=r['caption_file'],source_sha256=r['caption_file_sha256']),
    provenance=dict(labels_file=str(LABELS),labels_byte_offset=off,labels_line_sha256=hashlib.sha256(raw).hexdigest(),labels_sha256=EXPECTED,processed_listener_waveform_nonzero=not r['listener_audio_is_silent'],in_current_500h_trial=sid in av_ids,original_training_admitted=r['training_admitted'],human_ground_truth=False))
   data=line(row);out=outputs[s];pending.append((sid,s,r['pair'],r['listener_id'],group,r['source_start_s'],r['source_end_s'],r['duration_s'],r['face_p10'],s+'.jsonl',out.tell(),len(data)));out.write(data)
   counts[s]+=1;hours[s]+=r['duration_s']/3600;audio[s]+=int(not r['listener_audio_is_silent'])
   key=(s,'restored' if 'orientation_restoration' in r else 'upright','nonzero' if not r['listener_audio_is_silent'] else 'silent')
   canaries.setdefault(key,sid)
   if len(pending)>=4096:
    db.executemany('INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',pending);db.commit();pending=[]
    save(WORK/'RUN_STATUS.json',dict(stage='preparing_full_reference_index',processed=sum(counts.values()),total=sum(COUNTS.values()),at=time.time()))
  db.executemany('INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',pending);db.commit()
 ensure(digest.hexdigest()==EXPECTED and dict(counts)==COUNTS,'Latest label hash/count mismatch')
 ensure(not sum(prefix.values()),'Unexpected Assistant prefixes: run explicit historical prefix branches before publishing')
 ensure(db.execute('SELECT count(*) FROM samples').fetchone()[0]==sum(COUNTS.values()),'Coverage differs')
 db.close();olddb.close()
 shutil.copy2(oldroot/'crop_config.json',source/'crop_config.json')
 summary=dict(scope='full',schema='j-upright-response-sft-reference-v1',total_clips=sum(counts.values()),splits={s:dict(clips=counts[s],hours=hours[s],listener_nonzero=audio[s]) for s in SPLITS},manifests={s:dict(file=s+'.jsonl',bytes=(source/(s+'.jsonl')).stat().st_size,sha256=sha(source/(s+'.jsonl'))) for s in SPLITS},labels_sha256=EXPECTED,assistant_prefixes=dict(prefix),group_split_overlap=0,canary_ids=list(canaries.values()),reference_loader=str(ROOT/'scripts/j_upright_response_sft_data.py'),old_judgments_reused=0,reason='Recompute captions under frozen policies; never reuse by sample id alone')
 save(source/'summary.json',summary);save(source/'COMPLETE.json',dict(status='complete',summary_sha256=sha(source/'summary.json')))
 save(WORK/'PREFIX_CHECK.json',dict(status='complete',prefixes=dict(prefix),normalization_needed=False,body_unchanged=True))
 codehashes={str(p):sha(p) for p in vendor.iterdir() if p.suffix=='.py'}
 codehashes[str(ROOT/'scripts/j_upright_response_sft_data.py')]=sha(ROOT/'scripts/j_upright_response_sft_data.py')
 save(WORK/'PREPARED.json',dict(status='complete',labels_sha256=EXPECTED,total=sum(counts.values()),splits=dict(counts),code=codehashes,at=time.time()))
 print(json.dumps(dict(prepared=True,total=sum(counts.values()),splits=dict(counts))),flush=True)

def setup_job(name,source,splits,policy):
 job=WORK/name
 if (job/'config.json').exists():return job
 job.mkdir(exist_ok=True)
 b=module('filter_full_response_sft',WORK/'vendor/filter_full_response_sft.py')
 v=module('recover_voiced_response_sft',WORK/'vendor/recover_voiced_response_sft.py')
 template=ROOT/'experiments/listen_j_full_response_filter_20261101'
 prompt=(template/'system.txt').read_text() if policy=='strict' else v.SYSTEM+'\n'
 (job/'system.txt').write_text(prompt)
 summary=read(source/'summary.json');sources={}
 for s in splits:
  p=source/(s+'.jsonl');sources[s]=dict(path=str(p),file=p.name,bytes=p.stat().st_size,sha256=sha(p),count=summary['splits'][s]['clips'])
 cfg=read(template/'config.json');cfg.update(source=str(source),sources=sources,splits=list(splits),total=sum(x['count'] for x in sources.values()),physical_gpu=7,gpu_uuid=UUID,policy=policy,system_sha256=sha(job/'system.txt'),code_sha256=sha(WORK/'vendor/filter_full_response_sft.py'),source_summary_sha256=sha(source/'summary.json'),started_at=time.time(),validation_revision='upright-full-label-frozen-policy')
 if policy=='voiced':cfg.update(max_tokens=256,retry_max_tokens=512)
 save(job/'config.json',cfg);return job

def run_job(job):
 if (job/'COMPLETE.json').exists():return
 cfg=read(job/'config.json')
 if cfg['total']:
  for attempt in range(3):
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(WORK/'vendor')+':'+str(ROOT/'src'),LD_PRELOAD='/private/software/conda/envs/vllm/lib/libstdc++.so.6')
   cmd=[PY,str(WORK/'vendor/controller.py'),'worker','--job',str(job)]
   with (job/f'worker-{attempt}-{int(time.time())}.log').open('a') as f:
    child=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
    save(job/'process.json',dict(pid=child.pid,controller=os.getpid(),command=cmd,log=f.name,at=time.time()));code=child.wait()
   if code==0:break
   save(job/'failure.json',dict(exit_code=code,attempt=attempt,at=time.time()))
   if attempt==2:raise RuntimeError('Worker failed: '+str(job))
   time.sleep(20)
 b=module('filter_full_response_sft',WORK/'vendor/filter_full_response_sft.py');b.finalize(job)

def decisions(job):
 cfg=read(job/'config.json')
 for s in cfg['splits']:
  for mpath in sorted((job/'batches'/s).glob('*.meta.json')):
   m=read(mpath);p=job/m['result_file'];ensure(sha(p)==m['result_sha256'],'Changed judgment batch')
   for raw in p.open('rb'):yield json.loads(raw)

def recovery_source():
 source=WORK/'voiced_source'
 if (source/'COMPLETE.json').exists():return source
 source.mkdir(exist_ok=True);strict=WORK/'strict/filtered';counts=collections.Counter();hours=collections.Counter()
 for s in ('dev','train'):
  with (source/(s+'.jsonl')).open('wb') as out:
   for folder in ('rejected','needs_review'):
    for raw in (strict/folder/(s+'.jsonl')).open('rb'):
     r=json.loads(raw)
     if r['provenance']['processed_listener_waveform_nonzero'] is True:out.write(raw);counts[s]+=1;hours[s]+=r['duration_s']/3600
 shutil.copy2(WORK/'source/crop_config.json',source/'crop_config.json')
 save(source/'summary.json',dict(splits={s:dict(clips=counts[s],hours=hours[s]) for s in ('dev','train')},test_processed=False,candidate_gate='strict reject/review AND nonzero processed listener waveform'))
 save(source/'COMPLETE.json',dict(status='complete',summary_sha256=sha(source/'summary.json')));return source

def publish():
 if (WORK/'COMPLETE.json').exists():return
 temp=WORK/'final.partial';temp.mkdir(exist_ok=True)
 for folder in ('rejected','needs_review'):(temp/folder).mkdir(exist_ok=True)
 dbpath=temp/'index.sqlite';dbpath.unlink(missing_ok=True);db=sqlite3.connect(dbpath)
 db.execute('CREATE TABLE samples(sid TEXT PRIMARY KEY,split TEXT,pair TEXT,listener TEXT,group_id TEXT,start REAL,end REAL,duration REAL,face_p10 REAL,file TEXT,offset INTEGER,bytes INTEGER)')
 # Store compact decisions in SQLite instead of keeping a million full model outputs.
 ddpath=WORK/'decision_index.sqlite';ddpath.unlink(missing_ok=True);dd=sqlite3.connect(ddpath)
 dd.execute('CREATE TABLE decisions(sid TEXT PRIMARY KEY,split TEXT,strict TEXT,voiced TEXT,caption_sha TEXT)')
 for r in decisions(WORK/'strict'):
  dd.execute('INSERT INTO decisions VALUES(?,?,?,?,?)',(r['sample_id'],r['split'],r['decision'],None,r['caption_sha256']))
 dd.commit()
 for r in decisions(WORK/'voiced'):
  old=dd.execute('SELECT split,strict,caption_sha FROM decisions WHERE sid=?',(r['sample_id'],)).fetchone()
  ensure(old and old[0]==r['split']!='test' and old[1]!='keep' and old[2]==r['caption_sha256'],'Invalid voiced recovery origin')
  ensure(r['listener_waveform_nonzero'] is True,'Silent voiced recovery')
  dd.execute('UPDATE decisions SET voiced=? WHERE sid=?',(r['decision'],r['sample_id']))
 dd.commit();totals={};groups={};n=0;recovered=collections.Counter();source=WORK/'source';source_summary=read(source/'summary.json')
 for s in SPLITS:
  counts=collections.Counter();hours=collections.Counter();digest=hashlib.sha256();pending=[]
  with contextlib.ExitStack() as stack:
   outs={d:stack.enter_context((temp/p).open('wb')) for d,p in [('keep',s+'.jsonl'),('reject','rejected/'+s+'.jsonl'),('needs_review','needs_review/'+s+'.jsonl')]}
   for raw in (source/(s+'.jsonl')).open('rb'):
    digest.update(raw);r=json.loads(raw);sid=r['sample_id'];v=dd.execute('SELECT split,strict,voiced,caption_sha FROM decisions WHERE sid=?',(sid,)).fetchone()
    ensure(v and v[0]==r['split']==s and v[3]==hashlib.sha256(r['target']['text'].encode()).hexdigest(),'Decision/source differs')
    # Same historical final_decision precedence: any valid keep wins; any
    # still-unresolved applicable branch stays review; otherwise reject.
    decision='keep' if 'keep' in (v[1],v[2]) else ('needs_review' if 'needs_review' in (v[1],v[2]) else 'reject')
    ensure(s!='test' or v[2] is None,'Test entered relaxed recovery')
    if v[1]!='keep' and decision=='keep':recovered[s]+=1
    out=outs[decision];offset=out.tell();out.write(raw);counts[decision]+=1;hours[decision]+=r['duration_s']/3600;n+=1
    ensure(r['group_id'] not in groups or groups[r['group_id']]==s,'Final group leak');groups[r['group_id']]=s
    if decision=='keep':
     a=r['inputs']['speaker_audio'];fr=r['inputs']['listener_first_frame']
     pending.append((sid,s,r['pair'],fr['participant_id'],r['group_id'],a['start_s'],a['end_s'],r['duration_s'],r['face']['short_side_p10_px'],s+'.jsonl',offset,len(raw)))
    if len(pending)>=4096:db.executemany('INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',pending);db.commit();pending=[]
   db.executemany('INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',pending);db.commit()
  ensure(digest.hexdigest()==source_summary['manifests'][s]['sha256'] and sum(counts.values())==COUNTS[s],'Final source coverage changed')
  totals[s]=dict(counts=dict(counts),hours=dict(hours),voiced_recovered=recovered[s])
 ensure(n==sum(COUNTS.values())==dd.execute('SELECT count(*) FROM decisions').fetchone()[0],'Final partition lost samples')
 ensure(db.execute('SELECT count(*) FROM samples').fetchone()[0]==sum(t['counts'].get('keep',0) for t in totals.values()),'Final keep index differs')
 # The original pool has already audited no time overlap. Verify the exact
 # current accepted subset independently, allowing adjacency only.
 overlap=db.execute('SELECT count(*) FROM (SELECT start,MAX(end) OVER(PARTITION BY pair,listener ORDER BY start,end ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prior_end FROM samples) WHERE start<prior_end-0.000001').fetchone()[0]
 ensure(overlap==0,'Accepted source intervals overlap')
 db.close();dd.close();shutil.copy2(source/'crop_config.json',temp/'crop_config.json')
 manifests={str(p.relative_to(temp)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in temp.rglob('*.jsonl')}
 summary=dict(status='complete',total_inputs=n,splits=totals,labels_sha256=EXPECTED,group_split_overlap=0,source_interval_overlaps=overlap,manifests=manifests,original_rows_copied_verbatim=True,test_relaxed_recovery=False,assistant_prefix_check=read(WORK/'PREFIX_CHECK.json'),reference_loader=str(ROOT/'scripts/j_upright_response_sft_data.py'),at=time.time())
 save(temp/'summary.json',summary)
 from j_upright_response_sft_data import audit
 save(temp/'MEDIA_READBACK.json',audit(temp,source_summary['canary_ids']))
 temp.replace(WORK/'final')
 save(WORK/'COMPLETE.json',dict(status='complete',processed=n,summary_sha256=sha(WORK/'final/summary.json'),media_readback_sha256=sha(WORK/'final/MEDIA_READBACK.json'),export=str(WORK/'final'),at=time.time()))
 save(WORK/'RUN_STATUS.json',dict(stage='complete',splits=totals,at=time.time()))

def control():
 WORK.mkdir(exist_ok=True);lock=(WORK/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  prepare();p=read(WORK/'PREPARED.json')
  for path,h in p['code'].items():ensure(sha(path)==h,'Frozen code changed: '+path)
  sys.path[:0]=[str(WORK/'vendor'),str(ROOT/'scripts'),str(ROOT/'src')]
  save(WORK/'RUN_STATUS.json',dict(stage='strict_full_train_dev_test',at=time.time()));run_job(setup_job('strict',WORK/'source',SPLITS,'strict'))
  save(WORK/'RUN_STATUS.json',dict(stage='voiced_train_dev_recovery',at=time.time()));run_job(setup_job('voiced',recovery_source(),('dev','train'),'voiced'))
  save(WORK/'RUN_STATUS.json',dict(stage='export_and_audit',at=time.time()));publish()
 except BaseException as exc:
  save(WORK/'CONTROLLER_ERROR.json',dict(error=repr(exc),at=time.time()));raise
 finally:lock.close()

def worker(job):
 sys.path.insert(0,str(WORK/'vendor'))
 b=module('filter_full_response_sft',WORK/'vendor/filter_full_response_sft.py');v=module('recover_voiced_response_sft',WORK/'vendor/recover_voiced_response_sft.py')
 if read(job/'config.json')['policy']=='voiced':v.configure_policy()
 b.worker(job)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['control','prepare','worker','publish']);p.add_argument('--job',type=Path);a=p.parse_args()
 if a.action=='worker':worker(a.job)
 else:globals()[a.action]()
