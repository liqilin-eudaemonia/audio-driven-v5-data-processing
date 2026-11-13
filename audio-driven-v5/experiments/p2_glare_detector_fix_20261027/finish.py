"""Score corrected detections under three explicit common-support scopes, publish paired report."""
import sys,os,time,json,copy,subprocess,concurrent.futures,traceback
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_glare_eval.common import read,sha,digest,atomic_json,artifact_valid
from p2_glare_eval.media import aligned
from p2_glare_rgb.detector import code_identity
PY=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python';MP=read(ROOT/'configs/p2_omni_runtime.lock.json')['paths']['metrics_python'];ARMS=['E0','E1','E0F','E1F'];SEEDS=[137,251]
ENV={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUDA_VISIBLE_DEVICES':''}
def base(kind):return ROOT/f'experiments/p2_guided_{kind}24_fourway_eval_20261027'
def state(phase,**kw):atomic_json(OUT/'status.json',dict(phase=phase,time=time.time(),**kw))

def prepare_scores():
 p=read(OUT/'protocol.json');assert p['code']==code_identity();markers={};counts={'valid':0,'invalid':0};quality=[]
 expected_runtime=dict(read(ROOT/'.eval_models/detector_runtime.json'),fer_api='find_faces(RGB), detect_emotions(BGR, explicit_rectangles)',model_lifetime='persistent_stateless_FER_per_worker')
 for video in p['videos']:
  folder=OUT/'detections'/video['sha256'];m=read(folder/'complete.json');contract=m['contract']
  assert contract['video_sha256']==video['sha256'] and contract['policy']==p['policy'] and contract['code']==p['code'] and contract['runtime']==expected_runtime
  assert artifact_valid(folder,contract);assert sha(video['video'])==video['sha256'];markers[video['sha256']]=m;counts['valid' if m['valid'] else 'invalid']+=1
  q=read(folder/'quality.json');quality.append(dict(video=video['video'],sha256=video['sha256'],valid=m['valid'],frames=q['frames'],joint_valid=q['jointly_valid_frames'],mp_valid=q['mediapipe_valid_frames'],fer_valid=q['fer_valid_frames']))
 all_records={};scopes={};jobs={}
 for kind,pairs in p['pairs'].items():
  records={};reports={}
  for condition,rows in pairs.items():
   good=[];bad=[]
   for row in rows:
    flags={s:markers[row['files'][row[s+'_video']]] for s in ['gt','generated']}
    if not all(m['valid'] for m in flags.values()):
     bad.append(dict(sample_id=row['sample_id'],sample_seed=row['sample_seed'],invalid_sides=[s for s,m in flags.items() if not m['valid']]));continue
    aligned(flags['gt']['geometry'],flags['generated']['geometry']);record=dict(row,geometry=flags['gt']['geometry'])
    for side in ['gt','generated']:
     csv=OUT/'detections'/row['files'][row[side+'_video']]/'scores.csv';record[side+'_csv']=str(csv);record[side+'_csv_sha256']=sha(csv)
    good.append(record)
   records[condition]={(r['sample_id'],r['sample_seed']):r for r in good}
   report=dict(planned_pairs=48,valid_pairs=len(good),failed_pairs=len(bad),failures=bad,status='complete' if not bad else 'incomplete',detector_policy=p['policy'])
   atomic_json(OUT/'pairs'/kind/condition/'detected_pairs.json',good);atomic_json(OUT/'pairs'/kind/condition/'results.json',report);reports[condition]=report
  all_records[kind]=reports;scopes[kind]={};ids=[r['sample_id'] for r in pairs['E0-fixed'] if r['sample_seed']==137]
  for scope,conditions in [('paired',list(pairs)),('fixed',[a+'-fixed' for a in ARMS]),('guided',[a+'-guided' for a in ARMS])]:
   common=[sid for sid in ids if all((sid,seed) in records[c] for c in conditions for seed in SEEDS)];scopes[kind][scope]=dict(sample_ids=common,samples_per_seed=len(common),conditions={})
   for condition in conditions:
    if not common:continue
    key=digest([kind,condition,common,digest(p)])[:24];scopes[kind][scope]['conditions'][condition]=key
    jobs[key]=dict(key=key,dataset=kind,condition=condition,sample_ids=common,records=[records[condition][sid,seed] for seed in SEEDS for sid in common])
 atomic_json(OUT/'scope.json',dict(scopes=scopes,coverage=all_records,video_quality=quality,counts=counts));atomic_json(OUT/'score_jobs.json',list(jobs.values()));print('scope', {k:{s:v['samples_per_seed'] for s,v in scopes[k].items()} for k in scopes},flush=True)

def score(key):
 from p2_glare_eval.scoring import score_seed,aggregate_seeds
 job=next(j for j in read(OUT/'score_jobs.json') if j['key']==key);dest=OUT/'scores'/(key+'.json')
 if dest.exists():assert read(dest)['job']==job;return
 result={str(seed):score_seed([r for r in job['records'] if r['sample_seed']==seed],'cpu',32) for seed in SEEDS}
 atomic_json(dest,dict(job=job,per_seed=result,aggregation=aggregate_seeds(result)))

def publish():
 scope=read(OUT/'scope.json');cols=['FVD-32','UTMOSv2','LSE-D','LSE-C','R-F1','R-tIoU','R-ATD','R-FID'];rkeys=['R-F1','R-tIoU','R-ATD','R-FID']
 for kind in ['train','test']:
  r=read(base(kind)/'comparison.json');original=copy.deepcopy(r);r['detector_version']='rgb_v2';r['detector_policy']=read(OUT/'protocol.json')['policy'];s=scope['scopes'][kind];coverage=scope['coverage'][kind]
  fixedold=read(ROOT/f'experiments/p2_{kind}24_fourway_eval_20261027/table_values.json');fixedcomp=read(ROOT/f'experiments/p2_{kind}24_fourway_eval_20261027/comparison.json')['metrics']
  r['within_fixed']=dict(values=fixedold,coverage={m:fixedcomp[m]['samples_per_seed'] for m in ['fvd','utmos','sync']})
  for name in ['paired','fixed','guided']:
   item=s[name];n=item['samples_per_seed'];destination=r if name=='paired' else r['within_'+name];destination['coverage']['glare']=n
   details=dict(sample_ids=item['sample_ids'],samples_per_seed=n,planned_samples=24,conditions={})
   conditions=list(coverage) if name=='paired' else [a+'-'+name for a in ARMS]
   for condition in conditions:
    arm=condition.split('-')[0];key=condition if name=='paired' else arm
    if n:
     value=read(OUT/'scores'/(item['conditions'][condition]+'.json'));destination['values'][key].update(value['aggregation']['mean']);details['conditions'][condition]=value
    else:destination['values'][key].update({k:None for k in rkeys})
   destination.setdefault('details',{})['glare']=details
  r['glare_pair_coverage']=coverage;r['rgb_failure_analysis']=str(OUT/'scope.json')
  atomic_json(base(kind)/'comparison_rgb_v2.json',r)
  lines=['# 完整配对评测：RGB修正版','',f'数据集：{kind}。每条件24条×2seed=48输出，所有输出均执行评测。','MTCNN使用RGB检测人脸框；FER情绪分类使用原BGR图像。仍要求每帧MediaPipe和FER有效，且唯一人脸框匹配；没有补零、插值或降低覆盖率门槛。GLARE四个指标公式保持原样。','原BGR路径结果保存在comparison.json，新结果为comparison_rgb_v2.json。FVD/UTMOS/SyncNet与原配对评测一致，不重算或替换音频。']
  for title,item in [('八条件、双seed共同有效样本',r),('固定prompt内部四组、双seed共同有效样本',r['within_fixed']),('原文标注指导内部四组、双seed共同有效样本',r['within_guided'])]:
   lines+=['','## '+title,'',str(item['coverage']),'','| 条件 | '+' | '.join(cols)+' |','|---|'+'---:|'*len(cols)]
   for key,values in item['values'].items():lines.append('| '+key+' | '+' | '.join('N/A' if values.get(c) is None else f'{values[c]:.4f}' for c in cols)+' |')
  lines+=['','## GLARE完整配对覆盖','', '| 条件 | 成功/48 | 失败/48 |','|---|---:|---:|']
  for key,v in coverage.items():lines.append(f"| {key} | {v['valid_pairs']} | {v['failed_pairs']} |")
  lines+=['','不同支持集合的表不能直接跨表作差。R-ATD缺少匹配事件时保留N/A。指导文本含目标标注，不能当作仅凭speaker自主回应的证据。']
  (base(kind)/'report_rgb_v2.md').write_text('\n'.join(lines)+'\n')

def main():
 while True:
  workers=[read(OUT/'workers'/f'{i:03d}.json') if (OUT/'workers'/f'{i:03d}.json').exists() else {} for i in range(72)]
  state('detecting',completed=sum(len(w.get('results',[])) for w in workers),planned=816)
  if all(w.get('finished') for w in workers):break
  for rank in range(3):
   if (OUT/f'node-{rank}.json').exists() and any(read(OUT/f'node-{rank}.json')['codes']):raise RuntimeError('Detector worker exited with error')
  time.sleep(5)
 errors=[r for w in workers for r in w['results'] if r['status']=='error']
 if errors:raise RuntimeError('Detector execution errors: '+str(errors[:3]))
 state('preparing_scores');prepare_scores();state('scoring')
 def runjob(job):
  with (OUT/'logs'/('score-'+job['key']+'.log')).open('a') as log:subprocess.run([MP,str(__file__),'score',job['key']],stdout=log,stderr=subprocess.STDOUT,env=ENV,check=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(runjob,read(OUT/'score_jobs.json')))
 state('waiting_av_aggregation')
 parent=ROOT/'experiments/p2_guided_full_eval_20261027/status.json'
 while read(parent)['phase']!='complete':
  if read(parent)['phase']=='failed':raise RuntimeError('Original AV aggregation failed; resolve it before publishing')
  time.sleep(5)
 publish();state('building_html')
 with (OUT/'build_html.log').open('a') as log:subprocess.run([str(PY),str(ROOT/'scripts/build_p2_av_review.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
 state('complete',videos=816)
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='score':score(sys.argv[2])
 elif len(sys.argv)>1:
  with (OUT/'finish.log').open('a') as log:p=subprocess.Popen([str(PY),str(__file__)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
  print(p.pid)
 else:
  try:main()
  except Exception as error:state('failed',error=str(error));traceback.print_exc();raise
