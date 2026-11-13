"""Complete guided outputs, evaluate every planned pair, compare eight conditions on shared support."""
import sys,os,subprocess,time,concurrent.futures,uuid,fcntl,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(OUT))
from generate_cluster import remote,PY,ARMS
from p2_omni_eval.common import read,atomic_json,digest,job_key,artifact_valid,complete,subprocess_env,check_runtime,check_device
RUNTIME=ROOT/'configs/p2_omni_runtime.lock.json';R=read(RUNTIME);MP=R['paths']['metrics_python'];KINDS=['train','test']
def base(kind):return ROOT/f'experiments/p2_guided_en_{kind}24_fourway_eval_20261027'
def oldbase(kind):return ROOT/f'experiments/p2_{kind}24_fourway_eval_20261027'
def prepare():
 from p2_omni_eval.protocol import prepare as omni_prepare
 from p2_glare_eval.pipeline import prepare_pairs
 jobs=read(OUT/'jobs.json');assert all(artifact_valid(j['output'],j['identity']) for j in jobs)
 verified=[]
 for j in jobs:
  fixed=oldbase(j['kind'])/j['arm']/'generation/generated'/job_key(j['sample_id'],j['seed'])/'completed.json'
  before=read(fixed);after=read(Path(j['output'])/'completed.json')
  keys=['sample_id','group_id','sample_seed','mode','config_sha256','checkpoint_sha256','observation_sha256','text_lora_enabled','text_lora_targets','arm','train_seed','step','s0','geometry','status']
  assert all(before[k]==after[k] for k in keys),(j,'paired inference differs')
  assert after['target_annotation_access'] and not after['gt_media_access']
  verified.append(dict(dataset=j['kind'],arm=j['arm'],sample_id=j['sample_id'],seed=j['seed'],fixed_metadata=str(fixed),guided_output=j['output']))
 atomic_json(OUT/'generation_pairing_verified.json',dict(status='passed',pairs=len(verified),checked_fields=keys,rows=verified),immutable=True)
 check_runtime(R,full=True)
 for kind in KINDS:
  for arm in ARMS:
   dest=base(kind)/arm;selected=[j for j in jobs if j['kind']==kind and j['arm']==arm];assert len(selected)==48
   originals={(r['sample_id'],r['sample_seed']):r for r in read(oldbase(kind)/arm/'omni_pairs.json')['samples']}
   rows=[]
   for j in selected:
    r=dict(originals[j['sample_id'],j['seed']]);r['generated_video']=str(Path(j['output'])/'generated.mp4');r['generated_audio']=str(Path(j['output'])/'listener.wav');rows.append(r)
   atomic_json(dest/'omni_pairs.json',dict(schema='p2-omni-pairs-v1',samples=rows),immutable=True)
   omni_prepare(dest/'omni_pairs.json',dest/'omni',RUNTIME)
   grows=[{k:r[k] for k in ['sample_id','sample_seed','gt_video','generated_video']} for r in rows]
   atomic_json(dest/'glare_pairs.json',dict(schema='p2-glare-pairs-v1',samples=grows),immutable=True)
   prepare_pairs(dest/'glare_pairs.json',dest/'glare')
   # Reuse only verified successful independent GT detections, matched by video hash.
   cache=dest/'glare/detections';cache.mkdir(exist_ok=True)
   for marker in (oldbase(kind)/arm/'glare/detections').glob('*/complete.json'):
    destlink=cache/marker.parent.name
    if not destlink.exists():destlink.symlink_to(marker.parent.resolve(),target_is_directory=True)
   print('prepared',kind,arm,flush=True)
 atomic_json(OUT/'metrics_prepared.json',dict(outputs=384,finished=True,time=time.time()))

def metric_worker(rank):
 from p2_omni_eval.pipeline import compute_row,has_error,check_protocol
 from p2_omni_eval.backends import Backends
 pair=rank//2;kind=KINDS[pair//4];arm=ARMS[pair%4];shard=rank%2;work=base(kind)/arm/'omni';p=read(work/'protocol.json');check_protocol(p);backend=Backends(p,'cuda:0');results=[]
 for row in p['rows'][shard::2]:
  dest=work/'rows'/job_key(row['sample_id'],row['sample_seed']);dest.mkdir(parents=True,exist_ok=True);ident=dict(protocol=digest(p),row=row,device='cuda:0')
  with (dest/'row.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   if artifact_valid(dest,ident):record=read(dest/'result.json')
   else:
    attempt=dest/'attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True)
    record=compute_row(row,p,backend,attempt);atomic_json(attempt/'result.json',record);atomic_json(dest/'result.json',record)
    if not has_error(record):complete(dest,ident,[dest/'result.json',*attempt.glob('*.npy')])
   results.append(dict(sample_id=row['sample_id'],seed=row['sample_seed'],error=has_error(record)))
  atomic_json(work/f'shard-{shard}.json',dict(rank=rank,rows=results,finished=False,time=time.time()))
 check_protocol(p);atomic_json(work/f'shard-{shard}.json',dict(rank=rank,rows=results,finished=True,time=time.time()))

def node(rank):
 children=[];env=subprocess_env(R,'cuda:0')
 for local in range(8):
  check_device('cuda:'+str(local));r=rank*8+local
  with (OUT/'logs'/f'omni-{r:02d}.log').open('a') as log:children.append(('omni',subprocess.Popen([MP,str(__file__),'worker',str(r)],env={**env,'CUDA_VISIBLE_DEVICES':str(local)},stdout=log,stderr=subprocess.STDOUT)))
  time.sleep(.3)
 codes=[dict(type=k,code=p.wait()) for k,p in children];atomic_json(OUT/f'metric-node-{rank}.json',dict(finished=True,codes=codes,time=time.time()))
 assert all(r['code']==0 if r['type']=='omni' else r['code'] in (0,2) for r in codes)

def aggregate_all():
 from p2_omni_eval.pipeline import check_protocol,has_error
 from p2_omni_eval.scoring import aggregate
 for kind in KINDS:
  for arm in ARMS:
   work=base(kind)/arm/'omni';p=read(work/'protocol.json');check_protocol(p);rows=[]
   assert all(read(work/f'shard-{i}.json')['finished'] for i in range(2))
   for row in p['rows']:
    dest=work/'rows'/job_key(row['sample_id'],row['sample_seed']);record=read(dest/'result.json')
    if not has_error(record):assert artifact_valid(dest,dict(protocol=digest(p),row=row,device='cuda:0'))
    rows.append(record)
   result=aggregate(p,rows);result.update(protocol_sha256=digest(p),device='cuda:0',evaluation_role='target_annotation_guided_diagnostic')
   atomic_json(work/'results.json',result);print(kind,arm,'aggregate',result['status'],flush=True)

def compare(kind):
 import numpy as np
 from p2_omni_eval.scoring import aggregate
 from p2_omni_eval.compare import _supported
 from p2_glare_eval.scoring import score_seed,aggregate_seeds
 sources={a+'-'+v:(base(kind) if v=='guided' else oldbase(kind))/a for a in ARMS for v in ['fixed','guided']}
 rows={k:{(r['sample_id'],r['sample_seed']):r for r in read(path/'omni/results.json')['rows']} for k,path in sources.items()};seeds=[137,251];p=read(base(kind)/'E0/omni/protocol.json');values={k:{} for k in sources};coverage={};detail={};sids=[o['sample_id'] for o in read(base(kind)/'observations.json')]
 for metric in ['fvd','utmos','sync']:
  ids=[sid for sid in sids if all(_supported(rows[k][sid,s]['metrics'][metric]) for k in sources for s in seeds)];coverage[metric]=len(ids);detail[metric]=dict(sample_ids=ids,planned_samples=24,samples_per_seed=len(ids),conditions={})
  for key in sources:
   scores=aggregate(dict(p,metrics=[metric],sample_count=len(ids)),[rows[key][sid,s] for sid in ids for s in seeds]) if ids else None
   detail[metric]['conditions'][key]=scores['per_seed'] if scores else None
   def mean(field=None):
    if not scores:return None
    vals=[]
    for s in seeds:
     v=scores['per_seed'][str(s)][metric];v=v['full_value'] if metric=='fvd' else v['generated']['full_value'];vals.append(v.get(field) if v is not None and field else v)
    return float(np.mean(vals)) if all(v is not None for v in vals) else None
   if metric=='fvd':values[key]['FVD-32']=mean()
   elif metric=='utmos':values[key]['UTMOSv2']=mean()
   else:values[key].update({'LSE-D':mean('lse_d'),'LSE-C':mean('lse_c')})
 detected={k:{(r['sample_id'],r['sample_seed']):r for r in read(path/'glare/detected_pairs.json')} for k,path in sources.items()}
 ids=[sid for sid in sids if all((sid,s) in detected[k] for k in sources for s in seeds)];coverage['glare']=len(ids);detail['glare']=dict(sample_ids=ids,planned_samples=24,samples_per_seed=len(ids),conditions={})
 for key in sources:
  target=base(kind)/'matched_scores'/(key+'.json')
  if ids:
   # Match IDs on all 8 conditions and both seeds; keep failed full-set pairs explicit.
   if target.exists():res=read(target);assert res['sample_ids']==ids
   else:
    scores={str(s):score_seed([detected[key][sid,s] for sid in ids],'cpu',32) for s in seeds}
    res=dict(sample_ids=ids,per_seed=scores,aggregation=aggregate_seeds(scores));atomic_json(target,res)
   detail['glare']['conditions'][key]=res;values[key].update(res['aggregation']['mean'])
  else:values[key].update({k:None for k in ['R-F1','R-tIoU','R-ATD','R-FID']})
 # A strict eight-condition intersection can be empty. Also score the guided
 # four-arm common set, explicitly separately, never substitute its scores into
 # the eight-condition comparison or quietly loosen the detector.
 guided_keys=[a+'-guided' for a in ARMS];within_values={a:{} for a in ARMS};within_coverage={};within_details={}
 for metric in ['fvd','utmos','sync']:
  gids=[sid for sid in sids if all(_supported(rows[k][sid,s]['metrics'][metric]) for k in guided_keys for s in seeds)]
  within_coverage[metric]=len(gids);within_details[metric]=dict(sample_ids=gids,conditions={})
  for arm in ARMS:
   key=arm+'-guided'
   scores=(detail[metric]['conditions'][key] if gids==detail[metric]['sample_ids'] else aggregate(dict(p,metrics=[metric],sample_count=len(gids)),[rows[key][sid,s] for sid in gids for s in seeds])['per_seed']) if gids else None
   within_details[metric]['conditions'][arm]=scores
   def val(field=None):
    if not scores:return None
    vs=[]
    for s in seeds:
     v=scores[str(s)][metric];v=v['full_value'] if metric=='fvd' else v['generated']['full_value'];vs.append(v.get(field) if v is not None and field else v)
    return float(np.mean(vs)) if all(v is not None for v in vs) else None
   if metric=='fvd':within_values[arm]['FVD-32']=val()
   elif metric=='utmos':within_values[arm]['UTMOSv2']=val()
   else:within_values[arm].update({'LSE-D':val('lse_d'),'LSE-C':val('lse_c')})
 gids=[sid for sid in sids if all((sid,s) in detected[k] for k in guided_keys for s in seeds)]
 within_coverage['glare']=len(gids);within_details['glare']=dict(sample_ids=gids,conditions={})
 for arm in ARMS:
  key=arm+'-guided';target=base(kind)/'guided_common_scores'/(arm+'.json')
  if gids:
   if gids==ids:res=detail['glare']['conditions'][key]
   elif target.exists():res=read(target);assert res['sample_ids']==gids
   else:
    scores={str(s):score_seed([detected[key][sid,s] for sid in gids],'cpu',32) for s in seeds};res=dict(sample_ids=gids,per_seed=scores,aggregation=aggregate_seeds(scores));atomic_json(target,res)
   within_values[arm].update(res['aggregation']['mean']);within_details['glare']['conditions'][arm]=res
  else:within_values[arm].update({k:None for k in ['R-F1','R-tIoU','R-ATD','R-FID']})
 within=dict(values=within_values,coverage=within_coverage,details=within_details,scope='Four guided arms and both seeds; not a fixed-vs-guided matched scope')
 result=dict(schema='p2-guided-eight-condition-comparison-v1',dataset=kind,planned_outputs_per_condition=48,seeds=seeds,coverage=coverage,values=values,details=detail,within_guided=within,scope='All eight fixed/guided conditions and both seeds intersected separately per metric; exact failed/inapplicable pairs retained in per-condition results',target_annotation_access=True)
 atomic_json(base(kind)/'comparison.json',result);atomic_json(base(kind)/'table_values.json',values)
 cols=['FVD-32','UTMOSv2','LSE-D','LSE-C','R-F1','R-tIoU','R-ATD','R-FID'];lines=['# 完整原文标注指导与固定prompt对照','',f'数据集：{kind}；每个条件24条 × seed137/251，共48个输出。四种权重 × 两种prompt。','仅改变Gemma正向文本；负向文本、权重、输入音视频、采样参数保持原设置。','',f'共同支持：{coverage}。每项取8个条件、2个seed均有效的共同样本。完整48输出全部执行评测，检测失败不视为负例，缺失分数不补0。','', '| 条件 | '+' | '.join(cols)+' |','|---|'+'---:|'*len(cols)]
 for k,v in values.items():lines.append('| '+k+' | '+' | '.join('N/A' if v.get(c) is None else f'{v[c]:.4f}' for c in cols)+' |')
 lines+=['','## 指导版内部：四组权重、双seed共同支持','',f'覆盖：{within_coverage}；这一子集不能与上表或原固定prompt的不同支持集合直接作差。','', '| 条件 | '+' | '.join(cols)+' |','|---|'+'---:|'*len(cols)]
 for k,v in within_values.items():lines.append('| '+k+'-guided | '+' | '.join('N/A' if v.get(c) is None else f'{v[c]:.4f}' for c in cols)+' |')
 lines+=['','指导版使用了目标标注，不是仅凭speaker自主回应的结论。UTMOS/SyncNet不能替代词语正确性和回应适当性人工判断。','原固定prompt完整评测保持不变；本表为严格配对比较而重算共同支持范围。']
 (base(kind)/'report.md').write_text('\n'.join(lines)+'\n');print(kind,coverage,flush=True)

if __name__=='__main__':
 cmd=sys.argv[1]
 if cmd=='prepare':prepare()
 elif cmd=='worker':metric_worker(int(sys.argv[2]))
 elif cmd=='node':node(int(sys.argv[2]))
 elif cmd=='detach':
  rank=int(sys.argv[2])
  with (OUT/f'metric-node-{rank}.log').open('a') as log:child=subprocess.Popen([str(PY),str(__file__),'node',str(rank)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
  print(json.dumps(dict(rank=rank,pid=child.pid)))
 elif cmd=='launch':
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
   for r in pool.map(lambda r:remote(r,[PY,__file__,'detach',r]),range(2)):print(r)
 elif cmd=='aggregate':aggregate_all()
 elif cmd=='compare':compare(sys.argv[2])
