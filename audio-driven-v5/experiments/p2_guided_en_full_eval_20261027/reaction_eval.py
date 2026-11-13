"""RGB strict and no-quality-gate reaction evaluation for English guidance."""
import concurrent.futures,os,sys,subprocess,time,traceback
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(OUT)]
from generate_cluster import remote,PY,ARMS
from p2_glare_eval.common import read,sha,digest,atomic_json,artifact_valid
from p2_glare_eval.media import aligned
from p2_glare_rgb.detector import Engine,policy as strict_policy,code_identity as strict_code
from p2_glare_unfiltered.rules import process,policy,code_identity
WORK=OUT/'reactions';OLD=ROOT/'experiments/p2_glare_detector_fix_20261027'
DP=ROOT/'.venv-p2-glare-detector/bin/python'
MP=read(ROOT/'configs/p2_omni_runtime.lock.json')['paths']['metrics_python']
ENV={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
SEEDS=[137,251]
def state(phase,**kw):atomic_json(OUT/'reactions_status.json',dict(phase=phase,time=time.time(),**kw))
def prepare():
 videos={};pairs={}
 for kind in ['train','test']:
  pairs[kind]={}
  for arm in ARMS:
   for variant in ['fixed','guided']:
    base=ROOT/f'experiments/p2_{"guided_en_" if variant=="guided" else ""}{kind}24_fourway_eval_20261027'
    rows=read(base/arm/'glare/protocol.json')['samples'];pairs[kind][arm+'-'+variant]=rows
    for row in rows:
     for side in ['gt','generated']:
      v=row[side+'_video'];h=row['files'][v];videos.setdefault(h,dict(video=v,sha256=h))
 (WORK/'rgb').mkdir(parents=True,exist_ok=True);(WORK/'logs').mkdir(exist_ok=True)
 for h in videos:
  original=OLD/'detections'/h;target=WORK/'rgb'/h
  if original.exists() and not target.exists():target.symlink_to(original.resolve(),target_is_directory=True)
 p=dict(schema='p2-english-guidance-two-reaction-policies-v1',videos=list(videos.values()),pairs=pairs,
        strict_policy=strict_policy(),unfiltered_policy=policy(),code=code_identity(),source_workers=48,
        score_workers=16,score_device='cuda:0',rule_python=str(DP))
 atomic_json(WORK/'protocol.json',p,immutable=True)
def detect_worker(rank):
 p=read(WORK/'protocol.json');assert p['code']==code_identity();engine=Engine();results=[]
 for item in p['videos'][rank::48]:
  h=item['sha256'];folder=WORK/'rgb'/h
  engine.detect(item['video'],folder,h)
  q=process(item['video'],folder,WORK/'unfiltered'/h,h)
  results.append(dict(sha256=h,quality=q))
  atomic_json(WORK/'detect_workers'/f'{rank:02d}.json',dict(finished=False,results=results))
 atomic_json(WORK/'detect_workers'/f'{rank:02d}.json',dict(finished=True,results=results))
def detect_node(rank):
 children=[]
 env={**ENV,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TF_ENABLE_ONEDNN_OPTS':'0','TF_CPP_MIN_LOG_LEVEL':'2'}
 for local in range(24):
  i=rank*24+local
  with (WORK/'logs'/f'detect-{i:02d}.log').open('a') as log:children.append(subprocess.Popen([str(DP),str(__file__),'detect-worker',str(i)],env=env,stdout=log,stderr=subprocess.STDOUT))
  time.sleep(.2)
 codes=[p.wait() for p in children];atomic_json(WORK/f'detect-node-{rank}.json',dict(codes=codes,finished=True))
 if any(codes):raise RuntimeError(str(codes))
def prepare_scores():
 p=read(WORK/'protocol.json');qualities={};strict={};geometries={}
 for item in p['videos']:
  h=item['sha256'];markers={}
  for mode in ['rgb','unfiltered']:
   folder=WORK/mode/h;m=read(folder/'complete.json');assert artifact_valid(folder,m['contract']);markers[mode]=m
  assert sha(item['video'])==h
  strict[h]=markers['rgb']['valid'];geometries[h]=markers['rgb']['geometry'];qualities[h]=read(WORK/'unfiltered'/h/'quality.json')
 jobs={};scopes={};coverage={}
 for kind,conditions in p['pairs'].items():
  strict_records={};all_records={};coverage[kind]={}
  for condition,rows in conditions.items():
   sr={};ur={};counts=dict(planned_pairs=48,processed_pairs=48,excluded_pairs=0,strict_valid_pairs=0,frames_per_side=0)
   for side in ['gt','generated']:
    for field in ['fer_no_face_frames','mediapipe_no_face_frames','fer_multiple_face_frames']:counts[side+'_'+field]=0
   for row in rows:
    h={side:row['files'][row[side+'_video']] for side in ['gt','generated']};aligned(geometries[h['gt']],geometries[h['generated']])
    record=dict(row,geometry=geometries[h['gt']]);counts['frames_per_side']+=record['geometry']['frame_count']
    for side in h:
     for field in ['fer_no_face_frames','mediapipe_no_face_frames','fer_multiple_face_frames']:counts[side+'_'+field]+=qualities[h[side]][field]
    valid=all(strict[v] for v in h.values());counts['strict_valid_pairs']+=int(valid)
    for mode,target in [('rgb',sr),('unfiltered',ur)]:
     if mode=='rgb' and not valid:continue
     r=dict(record)
     for side in h:
      path=WORK/mode/h[side]/'scores.csv';r[side+'_csv']=str(path);r[side+'_csv_sha256']=sha(path)
     target[row['sample_id'],row['sample_seed']]=r
   assert len(ur)==48
   strict_records[condition]=sr;all_records[condition]=ur;coverage[kind][condition]=counts
   atomic_json(WORK/'pairs'/kind/condition/'records_unfiltered.json',list(ur.values()))
   atomic_json(WORK/'pairs'/kind/condition/'records_strict.json',list(sr.values()))
  ids=[row['sample_id'] for row in conditions['E0-fixed'] if row['sample_seed']==137];assert len(ids)==24
  scopes[kind]={}
  for mode,records in [('strict',strict_records),('unfiltered',all_records)]:
   scopes[kind][mode]={}
   for scope,keys in [('paired',list(conditions)),('fixed',[a+'-fixed' for a in ARMS]),('guided',[a+'-guided' for a in ARMS])]:
    selected=[sid for sid in ids if all((sid,s) in records[k] for k in keys for s in SEEDS)]
    if mode=='unfiltered':assert selected==ids
    scopes[kind][mode][scope]=dict(sample_ids=selected,samples_per_seed=len(selected),conditions={})
    for condition in keys:
     if not selected:continue
     indices={}
     for seed in SEEDS:
      rows=[records[condition][sid,seed] for sid in selected]
      # Include file hashes: strict and relaxed can share a job only when the complete records are identical.
      key=digest([kind,condition,seed,rows])[:24]
      jobs[key]=dict(key=key,dataset=kind,condition=condition,seed=seed,records=rows,protocol_sha256=digest(p))
      indices[str(seed)]=key
     scopes[kind][mode][scope]['conditions'][condition]=indices
 atomic_json(WORK/'scope.json',dict(scopes=scopes,coverage=coverage,video_quality=qualities,
             strict_valid_videos=sum(strict.values()),unfiltered_processed_videos=len(qualities)))
 atomic_json(WORK/'score_jobs.json',list(jobs.values()),immutable=True)
def score_worker(rank):
 from p2_glare_eval.scoring import score_seed
 p=read(WORK/'protocol.json');assert p['code']==code_identity()
 for job in read(WORK/'score_jobs.json')[rank::16]:
  dest=WORK/'scores'/(job['key']+'.json')
  if dest.exists():assert read(dest)['job']==job;continue
  for row in job['records']:
   for side in ['gt','generated']:
    assert sha(row[side+'_csv'])==row[side+'_csv_sha256']
  result=score_seed(job['records'],'cuda:0',32)
  atomic_json(dest,dict(job=job,result=result,device='cuda:0'))
def score_node(rank):
 children=[]
 for local in range(8):
  i=rank*8+local
  with (WORK/'logs'/f'score-{i:02d}.log').open('a') as log:children.append(subprocess.Popen([MP,str(__file__),'score-worker',str(i)],env={**ENV,'CUDA_VISIBLE_DEVICES':str(local)},stdout=log,stderr=subprocess.STDOUT))
 codes=[p.wait() for p in children];atomic_json(WORK/f'score-node-{rank}.json',dict(codes=codes,finished=True))
 if any(codes):raise RuntimeError(str(codes))
def launch_nodes(mode):
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for value in pool.map(lambda r:remote(r,[PY,__file__,'detach-'+mode,str(r)]),range(2)):print(value,flush=True)
def wait_nodes(mode):
 while True:
  nodes=[WORK/f'{mode}-node-{r}.json' for r in range(2)]
  for p in nodes:
   if p.exists() and (read(p).get('error') or any(read(p).get('codes',[]))):raise RuntimeError(p.read_text())
  if all(p.exists() for p in nodes):return
  if mode=='detect':
   n=sum(len(read(p)['results']) for p in (WORK/'detect_workers').glob('*.json'));state('detecting',completed=n,planned=816)
  else:state('scoring',completed=len(list((WORK/'scores').glob('*.json'))),planned=len(read(WORK/'score_jobs.json')))
  time.sleep(5)
def main():
 # Streaming workers own their cache directories until they all exit.
 state('waiting_stream_detection')
 while not all((WORK/f'stream-node-{r}.json').exists() for r in range(2)):
  time.sleep(5)
 prepare();state('detecting');launch_nodes('detect');wait_nodes('detect')
 state('preparing_scores');prepare_scores();state('waiting_av_gpu_release')
 while not (OUT/'av_complete.json').exists():
  if read(OUT/'status.json')['phase']=='failed':raise RuntimeError('AV pipeline failed')
  time.sleep(5)
 state('scoring');launch_nodes('score');wait_nodes('score')
 assert len(list((WORK/'scores').glob('*.json')))==len(read(WORK/'score_jobs.json'))
 state('complete');atomic_json(OUT/'reactions_complete.json',dict(complete=True))
if __name__=='__main__':
 cmd=sys.argv[1] if len(sys.argv)>1 else 'run'
 if cmd=='launch' or cmd.startswith('detach-'):
  target=['run'] if cmd=='launch' else [cmd.removeprefix('detach-')+'-node',sys.argv[2]]
  path=OUT/'reaction_eval.log' if cmd=='launch' else WORK/(target[0]+'-'+sys.argv[2]+'.log')
  with path.open('a') as log:p=subprocess.Popen([str(PY),str(__file__),*target],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True,env=ENV)
  print(p.pid)
 else:
  try:
   if cmd=='run':main()
   elif cmd=='detect-worker':detect_worker(int(sys.argv[2]))
   elif cmd=='detect-node':detect_node(int(sys.argv[2]))
   elif cmd=='score-worker':score_worker(int(sys.argv[2]))
   elif cmd=='score-node':score_node(int(sys.argv[2]))
   else:raise ValueError(cmd)
  except Exception as error:
   if cmd.endswith('-node'):atomic_json(WORK/(cmd+'-'+sys.argv[2]+'.json'),dict(error=str(error)))
   elif cmd=='run':state('failed',error=str(error))
   traceback.print_exc();raise
