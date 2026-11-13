"""Use the native stage2 evaluation on the approved28 GPUs, with shared claims."""
from pathlib import Path
import argparse, collections, copy, fcntl, inspect, json, os, subprocess, sys, time
W=Path(__file__).resolve().parent
sys.path[:0]=[str(W),str(W.parent/'p2_stage2_joint_full6000_trial28_20261112')]
import isolate_events
isolate_events.activate()
import runtime as R
R.activate()
from p2_listen.contracts import read,sha,digest,atomic_json
from p2_stage2_joint_v2 import eval_worker as N,evaluation as E
PY='/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python'

def code_identity():
 value=read(W/'EVALUATOR_BASE_SOURCE.json')
 for p in [W/'eval28.py',W/'evaluate_overlap.py',W/'EVALUATOR_BASE_SOURCE.json',W/'isolation/sitecustomize.py']:
  value[str(p)]=sha(p)
 return value

def folder(step):return R.RUN/'evaluation'/f'step_{step:06d}'

def values(path):return [json.loads(line)for line in Path(path).read_text().splitlines()if line.strip()]

def save_rows(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
 temp.write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n'for v in rows));temp.replace(path)

def export(c,step):
 out=folder(step);p=out/'model.pt';receipt=out/'MODEL_EXPORT.json'
 if receipt.exists():
  r=read(receipt)
  if r['checkpoint_sha256']!=read(R.RUN/f'step_{step:06d}.complete.json')['sha256']or r['model_stat']!=stat(p):raise ValueError('Evaluation model changed')
  return
 import torch
 from p2_stage2_joint_v2.checkpoint import checked
 from p2_listen.cache import atomic_torch
 saved=checked(R.RUN/f'step_{step:06d}.pt')
 if saved['identity']['experiment']!=c or saved['identity']['source']!=R.source():raise ValueError('Wrong joint checkpoint')
 atomic_torch(p,{k:saved[k]for k in ('schema','model','identity','step')})
 atomic_json(receipt,dict(checkpoint_sha256=read(R.RUN/f'step_{step:06d}.complete.json')['sha256'],model_sha256=sha(p),model_stat=stat(p)))

def stat(path):
 s=Path(path).stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)

def load_state(c,step):
 import torch
 out=folder(step);p=out/'model.pt';r=read(out/'MODEL_EXPORT.json')
 if r['model_stat']!=stat(p)or r['checkpoint_sha256']!=read(R.RUN/f'step_{step:06d}.complete.json')['sha256']:raise ValueError('Export identity changed')
 saved=torch.load(p,weights_only=True,map_location='cpu',mmap=True)
 if saved['step']!=step or saved['identity']['source']!=R.source()or saved['identity']['experiment']!=c:raise ValueError('Wrong evaluation export')
 return saved

def claimed_rows(c,step,phase):
 out=folder(step);contract=digest(read(out/'CONTRACT.json'))
 rows=sorted(NATIVE_ROWS(c),key=lambda x:(-x['duration_s'],x['sample_id']))
 for row in rows:
  dest=out/'claims'/phase/row['sample_id'];dest.mkdir(parents=True,exist_ok=True)
  with(dest/'lock').open('a')as lock:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:continue
   if(dest/'DONE.json').exists():
    if read(dest/'DONE.json')['contract_sha256']!=contract:raise ValueError('Claim identity changed')
    continue
   yield row
   atomic_json(dest/'DONE.json',dict(success=True,contract_sha256=contract,at=time.time()))

NATIVE_ROWS=N.rows

def publish(records,value,step,phase):
 records.append(value)
 key=value.get('sample_id',value.get('id'))
 atomic_json(folder(step)/'records'/phase/(key+'.json'),value)

def phase_function(phase,step):
 native=N.PHASES[phase];source=inspect.getsource(native)
 if phase in ('qwen','references'):
  old='records.append(dict'if phase=='qwen'else'pairs.append(dict'
  if source.count(old)!=1:raise ValueError('Native record publication changed')
  source=source.replace(old,('_publish(records,dict'if phase=='qwen'else'_publish(pairs,dict'))
 # Worker-local partial summaries must never race with another worker.
 if phase=='qwen':source=source.replace("atomic_json(out/'planning_metrics.json',records)","None")
 if phase=='references':source=source.replace("(out/'pairs.jsonl').write_text(''.join(json.dumps(v)+'\\n'for v in pairs))","None")
 namespace=dict(N.__dict__,rows=lambda c:claimed_rows(c,step,phase),load_state=load_state,
  _publish=lambda records,value:publish(records,value,step,phase))
 exec(compile(source,str(W/'eval28.py')+'::native_'+phase,'exec'),namespace)
 return namespace[phase]

def merge_phase(c,step,phase):
 expected={r['sample_id']for r in NATIVE_ROWS(c)};out=folder(step)
 actual={p.parent.name for p in(out/'claims'/phase).glob('*/DONE.json')}
 if actual!=expected:raise ValueError('Incomplete evaluation phase '+phase+': '+str(len(actual)))
 if phase in ('qwen','references'):
  records=[read(out/'records'/phase/(sid+'.json'))for sid in sorted(expected)]
  if phase=='qwen':atomic_json(out/'planning_metrics.json',records)
  else:save_rows(out/'pairs.jsonl',records)
 atomic_json(out/'stages'/(phase+'.json'),dict(success=True,samples=64,contract_sha256=digest(read(out/'CONTRACT.json')),at=time.time()))

def cluster_phase(c,step,phase,python=PY):
 from listen_cluster import launch
 out=folder(step)
 if(out/'stages'/(phase+'.json')).exists():return
 atomic_json(out/'STATUS.json',dict(stage=phase,at=time.time(),complete=False))
 cl=read(R.RUN/'cluster.json')
 commands=[[python,'-u',str(W/'eval28.py'),'node','--step',str(step),'--phase',phase,'--index',str(i)]for i in range(4)]
 launch(cl,commands,out/'launches'/(phase+'_'+str(time.time_ns())))
 if phase in N.PHASES:merge_phase(c,step,phase)
 elif phase=='metrics':merge_metrics(c,step)
 elif phase=='plans':merge_plans(c,step)

def node(c,step,phase,index):
 import socket
 cl=read(R.RUN/'cluster.json');n=cl['nodes'][index]
 if n['host']not in subprocess.check_output(['hostname','-I'],text=True).split():raise ValueError('Wrong evaluation host')
 children=[];streams=[];specs=[];offsets=[];attempts=[];ignored=set();out=folder(step)
 try:
  for gpu in n['gpus']:
   rank=index*8+gpu;dest=out/'workers'/phase;dest.mkdir(parents=True,exist_ok=True)
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=''if phase=='references'else str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True',TOKENIZERS_PARALLELISM='false')
   py=c['evaluation']['metric_python']if phase=='metrics'else read(out/'event_detector_config.json')['inference_python']if phase=='plans'else PY
   from p2_testsets_eval.cluster import environment
   paths=[str(R.ROOT/'src'),str(R.ROOT/'scripts'),str(R.ROOT.parent/'OmniResponse')]
   if phase=='metrics':paths.insert(0,'/private/liqi_workspace/.cache/paired_video_metrics/python')
   env['PYTHONPATH']=os.pathsep.join([str(W/'isolation'),*paths])
   env['LD_LIBRARY_PATH']=str(Path(py).parent.parent/'lib')+':'+env.get('LD_LIBRARY_PATH','')
   if phase=='plans':env['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
   stream=(dest/f'rank_{rank:02d}.log').open('ab',buffering=0);streams.append(stream)
   atomic_json(dest/f'rank_{rank:02d}.json',dict(rank=rank,node=n['name'],host=n['host'],gpu=gpu,mask=env['CUDA_VISIBLE_DEVICES'],at=time.time()))
   cmd=[py,'-u',str(W/'eval28.py'),'worker','--step',str(step),'--phase',phase,'--rank',str(rank)]
   specs.append((cmd,env,dest/f'rank_{rank:02d}.log',rank));offsets.append(stream.tell());attempts.append(0)
   children.append(subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT))
  while True:
   for i,p in enumerate(children):
    if i in ignored or p.poll()in(None,0):continue
    cmd,env,log,rank=specs[i];failure=log.read_bytes()[offsets[i]:].decode(errors='replace')
    if ('EADDRINUSE'in failure or'address already in use'in failure)and attempts[i]<3:
     if phase in N.PHASES or phase=='plans':
      if len(list((out/'claims'/phase).glob('*/DONE.json')))==64:ignored.add(i);continue
     attempts[i]+=1;offsets[i]=log.stat().st_size
     atomic_json(out/'port_retries'/f'{phase}_{rank}_{attempts[i]}.json',dict(at=time.time(),reason='transient port bind collision; retain peers and completed claims',rank=rank))
     children[i]=subprocess.Popen(cmd,env=env,stdout=streams[i],stderr=subprocess.STDOUT)
    else:raise RuntimeError('Evaluation worker failed: '+phase+' '+n['name'])
   if all(p.poll()is not None or i in ignored for i,p in enumerate(children)):break
   time.sleep(1)
  if any(p.returncode and i not in ignored for i,p in enumerate(children)):raise RuntimeError('Evaluation failed')
 finally:
  for p in children:
   if p.poll()is None:p.terminate()
  for p in children:
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  for stream in streams:stream.close()

def metric_worker(c,step,rank):
 out=folder(step);base=out/'metrics_shards'/f'{rank:02d}';manifest=out/'metric_manifests'/f'{rank:02d}.jsonl'
 omni=R.ROOT.parent/'OmniResponse';assets=c['evaluation']['metric_assets']
 for kind in ('video','audio'):
  dest=base/kind
  if(dest/'summary.json').exists():continue
  if dest.exists():dest.rename(dest.with_name(kind+'.failed.'+str(time.time_ns())))
  cmd=[sys.executable,'-m','evaluation.video.paired_metrics.run'if kind=='video'else'evaluation.listener_metrics.run','--manifest',str(manifest),'--assets',assets,'--output',str(dest)]
  if kind=='video':cmd+=['--device','cuda:0','--threads','4','--batch-size','64','--pixel-workers','4','--gt-cache',str(R.RUN/'evaluation/gt_features'),'--metrics','psnr,ssim,lpips','--allow-partial']
  else:cmd+=['--stages','audio','--quality-kinds','utmos','--audio-device','cuda:0']
  E.run_command(cmd,base/(kind+'.log'),cwd=omni,env=os.environ.copy(),allowed=(0,2))

def merge_metrics(c,step):
 from evaluation.video.paired_metrics.run import summarize
 from evaluation.listener_metrics.run import quality_summary
 from evaluation.listener_metrics.core import aggregate_activity
 out=folder(step);video=[];audio=[];contracts=[];errors={}
 for rank in range(28):
  base=out/'metrics_shards'/f'{rank:02d}'
  video+=values(base/'video/pairs.jsonl');audio+=values(base/'audio/pairs.jsonl')
  contracts.append(read(base/'video/contract.json'));errors.update(read(base/'video/summary.json').get('initialization_errors',{}))
 expected={r['sample_id']for r in NATIVE_ROWS(c)}
 for rows in (video,audio):
  if len(rows)!=64 or {r['id']for r in rows}!=expected:raise ValueError('Metric population mismatch')
 for contract in contracts:
  if contract['code']!=contracts[0]['code']or contract['assets']!=contracts[0]['assets']:raise ValueError('Metric implementation/assets differ')
 summary=summarize(video,{},set(('psnr','ssim','lpips')),contracts[0]['schema'],contracts[0]['datasets'],64,errors)
 atomic_json(out/'metrics/video/summary.json',summary);save_rows(out/'metrics/video/pairs.jsonl',video)
 valid=[r for r in audio if'audio'in r]
 a=dict(schema='listener-metrics-v1',requested_pairs=64,datasets=['stage2_dev'],pairs_with_errors=sum(bool(r['errors'])for r in audio),
  errors=dict(collections.Counter(k for r in audio for k in r['errors'])),audio=dict(valid_pairs=len(valid),requested_pairs=64,
  groups=dict(collections.Counter(r['audio']['group']for r in valid)),activity={k:aggregate_activity([r['audio']['comparison'][k]for r in valid])for k in ('acoustic','speech')},
  quality=quality_summary(audio,('utmos',)),failed_pair_ids=[r['id']for r in audio if'audio'not in r]))
 a['status']='partial'if a['pairs_with_errors']else'complete'
 atomic_json(out/'metrics/audio/summary.json',a);save_rows(out/'metrics/audio/pairs.jsonl',audio)
 atomic_json(out/'stages/metrics.json',dict(success=True,at=time.time(),samples=64))

def plan_worker(c,step):
 from p2_stage2_joint_v2 import plan_evidence as P
 source=inspect.getsource(P.extract)
 start=source.index(" policy=read(c['evaluation']['event_config']);expected=")
 end=source.index(" out=Path(c['output'])",start)
 source=source[:start]+" policy=read_policy()\n"+source[end:]
 # Text judging needs no multimodal capacity profiling. Use the already
 # validated native detector loader settings on the same frozen judge.
 needle="max_model_len=policy['max_model_len'],max_num_seqs=1,enforce_eager=True)"
 if source.count(needle)!=1:raise ValueError('Native plan-judge loader changed')
 source=source.replace(needle,"max_model_len=policy['max_model_len'],max_num_seqs=1,max_num_batched_tokens=policy['max_model_len'],enable_chunked_prefill=False,limit_mm_per_prompt={'image':128,'audio':1,'video':1},skip_mm_profiling=True,enforce_eager=True)")
 source=source.replace("records.append(dict", "_publish(records,dict")
 old="raise ValueError('Plan evidence quote not present verbatim')"
 if source.count(old)!=1:raise ValueError('Native plan evidence guard changed')
 source=source.replace(old,"value.update(state='unknown',evidence='verbatim_quote_not_found; original raw judge output retained')")
 source=source.replace("atomic_json(out/'plan_labels.json',records)","None")
 def read_adapter(path):
  if str(path)==c['validation_manifest']:return claimed_rows(c,step,'plans')
  return read(path)
 ns=dict(P.__dict__,read=read_adapter,read_policy=lambda:read(folder(step)/'event_detector_config.json'),
  _publish=lambda records,value:publish(records,value,step,'plans'))
 exec(compile(source,str(W/'eval28.py')+'::plan_evidence','exec'),ns)
 ns['extract'](c,step)

def merge_plans(c,step):
 expected={r['sample_id']for r in NATIVE_ROWS(c)};out=folder(step)
 if {p.parent.name for p in(out/'claims/plans').glob('*/DONE.json')}!=expected:raise ValueError('Missing plan evidence')
 atomic_json(out/'plan_labels.json',[read(out/'records/plans'/(sid+'.json'))for sid in sorted(expected)])
 rejected=[]
 for sid in sorted(expected):
  raw=read(out/'plan_labels'/sid/'raw.json');labels=json.loads(raw['text']);plan=read(out/'conditions'/sid/'plan.json')['text']
  for mod,classes in labels.items():
   for name,value in classes.items():
    if value['state']!='unknown'and value['evidence']not in plan:rejected.append(dict(sample_id=sid,modality=mod,category=name,original=value))
 atomic_json(out/'PLAN_EVIDENCE_AUDIT.json',dict(total_units=64*7,rejected_quote_units=len(rejected),rejected=rejected,policy='No supported verbatim quote => unknown; raw judge output retained; no inferred negative'))
 atomic_json(out/'stages/plans.json',dict(success=True,at=time.time(),samples=64))

def reuse_plan_raw(c,step):
 from p2_event_presence.contracts import validate_labels
 out=folder(step);contract=digest(read(out/'CONTRACT.json'))
 for row in NATIVE_ROWS(c):
  sid=row['sample_id'];raw=out/'plan_labels'/sid/'raw.json'
  if not raw.exists()or(out/'claims/plans'/sid/'DONE.json').exists():continue
  value=read(raw)
  if value['finish_reason']!='stop':continue
  labels=json.loads(value['text']);validate_labels(labels);plan=read(out/'conditions'/sid/'plan.json')['text']
  for classes in labels.values():
   for cell in classes.values():
    if cell['state']!='unknown'and cell['evidence']not in plan:cell.update(state='unknown',evidence='verbatim_quote_not_found; original raw judge output retained')
  publish([],dict(sample_id=sid,labels=labels,plan_sha256=sha(out/'conditions'/sid/'plan.json')),step,'plans')
  atomic_json(out/'claims/plans'/sid/'DONE.json',dict(success=True,contract_sha256=contract,at=time.time(),reused_raw_judge=True))

def freeze_parent_reference(c,step):
 from p2_event_presence.scoring import score
 parent=R.PARENT/'run/evaluation/step_000000'
 receipt=read(R.RUN/'CONTINUATION_ACCEPTED.json');report=read(parent/'REPORT.json')
 if sha(parent/'REPORT.json')!=receipt['parent_latest_reports']['0']:raise ValueError('Approved latest baseline changed')
 for name,value in report['metrics'].items():
  if sha(parent/name)!=value:raise ValueError('Baseline evidence changed: '+name)
 frozen=read(parent/'REFERENCE_EVENTS.json');out=folder(step)
 if c['validation_manifest_sha256']!=report['validation_sha256']:raise ValueError('Different validation population')
 atomic_json(R.RUN/'evaluation/step_000000/REFERENCE_EVENTS.json',frozen,immutable=True)
 raw=values(out/'events/event_pairs.jsonl');fixed=E.fixed_reference_records(raw,frozen)
 save_rows(out/'event_pairs_fixed_reference.jsonl',fixed)
 atomic_json(out/'fixed_reference_event_summary.json',score(fixed,values(out/'events/frozen_pairs.jsonl')))
 atomic_json(out/'BASELINE_SOURCE.json',dict(parent_report_sha256=sha(parent/'REPORT.json'),parent_checkpoint_sha256=report['checkpoint_sha256'],baseline_is_previous_run=True))

def evaluate(step):
 c=R.validate(read(R.RUN/'config.json'));out=folder(step);out.mkdir(parents=True,exist_ok=True)
 with(out/'evaluation.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  atomic_json(out/'CONTRACT.json',E.stage_contract(c,step),immutable=True)
  atomic_json(out/'EVAL_CODE.json',code_identity(),immutable=True)
  atomic_json(W/'PLAN.json',dict(cluster=read(R.RUN/'cluster.json'),source=code_identity()),immutable=True)
  reuse_plan_raw(c,step)
  export(c,step)
  for phase in ('qwen','gemma','whisper','render','decode','references'):cluster_phase(c,step,phase)
  import parallel_events as PE
  cl=read(R.RUN/'cluster.json')
  for rank,part in enumerate(PE.pair_shards(values(out/'pairs.jsonl'))):save_rows(out/'metric_manifests'/f'{rank:02d}.jsonl',part)
  cluster_phase(c,step,'metrics')
  policy=read(c['evaluation']['event_config']);policy.update(gpu_memory_utilization=.80,cpu_offload_gb=0,authorized_scope='selected28; actual GPU bindings recorded separately; canonical GPU7 policy identifier only')
  atomic_json(out/'event_detector_config.json',policy,immutable=True)
  from p2_event_presence.media import bind
  events=out/'events'
  if not(events/'contract.json').exists():bind([out/'pairs.jsonl'],events,out/'event_detector_config.json',method='stage2')
  PE.parallel_cpu(events,cl,'prepare');PE.parallel_cpu(events,cl,'candidates')
  if not(events/'DETECTION_COMPLETE.json').exists():PE.parallel_infer(events,cl)
  from p2_event_presence.reporting import report
  report(events)
  cluster_phase(c,step,'plans')
  freeze_parent_reference(c,step);result=E.summarize(c,step)
  result['metrics'].update({name:sha(out/name)for name in ('EVAL_CODE.json','event_detector_config.json','MODEL_EXPORT.json','PLAN_EVIDENCE_AUDIT.json','BASELINE_SOURCE.json')})
  if(out/'EVAL_CODE_HISTORY.json').exists():result['metrics']['EVAL_CODE_HISTORY.json']=sha(out/'EVAL_CODE_HISTORY.json')
  atomic_json(out/'REPORT.json',result)
  E.verify_report(c,step)
  atomic_json(out/'STATUS.json',dict(stage='finished',at=time.time(),complete=result['summary']['complete']))
  print(json.dumps(result['summary']),flush=True)
  if step in(100,300):print(json.dumps(E.decision(c,step)),flush=True)

def worker(c,step,phase,rank):
 from parallel_events import assignment
 n,gpu=assignment(read(R.RUN/'cluster.json'),rank)
 if n['host']not in subprocess.check_output(['hostname','-I'],text=True).split():raise ValueError('Wrong worker host')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=(''if phase=='references'else str(gpu)):raise ValueError('Wrong worker device')
 if read(folder(step)/'EVAL_CODE.json')!=code_identity():raise ValueError('Evaluation code changed')
 # Match the native evaluation CLI's bootstrap before its phase functions
 # import the reused omni_av/omni_av_p2 modules.
 from p2_stage2_joint_v2.data import data_config
 from p2_listen.config import extractor_config
 extractor_config(data_config(c['generator']))
 if phase in N.PHASES:phase_function(phase,step)(c,step,'cuda:0')
 elif phase=='metrics':metric_worker(c,step,rank)
 elif phase=='plans':plan_worker(c,step)

if __name__=='__main__':
 sys.path.insert(0,str(R.ROOT.parent/'OmniResponse'))
 p=argparse.ArgumentParser();p.add_argument('action',choices=['evaluate','node','worker']);p.add_argument('--step',type=int,required=True,choices=[1000]);p.add_argument('--phase');p.add_argument('--index',type=int);p.add_argument('--rank',type=int)
 a=p.parse_args();c=R.validate(read(R.RUN/'config.json'))
 if a.action=='evaluate':evaluate(a.step)
 elif a.action=='node':node(c,a.step,a.phase,a.index)
 else:worker(c,a.step,a.phase,a.rank)
