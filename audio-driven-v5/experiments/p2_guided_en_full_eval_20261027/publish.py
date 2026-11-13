"""Publish English guidance with strict and full-sample GLARE reports."""
import copy,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_glare_eval.common import read,atomic_json,sha
from p2_glare_eval.scoring import aggregate_seeds
from p2_omni_eval.scoring import aggregate
from p2_omni_eval.compare import _supported
ARMS=['E0','E1','E0F','E1F'];SEEDS=[137,251]
COLS=['FVD-32','UTMOSv2','LSE-D','LSE-C','R-F1','R-tIoU','R-ATD','R-FID']
def mean(scores,metric,field=None):
 if not scores:return None
 values=[]
 for seed in SEEDS:
  v=scores[str(seed)][metric]
  v=v['full_value'] if metric=='fvd' else v['generated']['full_value']
  values.append(v.get(field) if v is not None and field else v)
 return sum(values)/len(values) if all(v is not None for v in values) else None
def av(kind):
 base=ROOT/f'experiments/p2_guided_en_{kind}24_fourway_eval_20261027'
 sources={a+'-'+v:(base if v=='guided' else ROOT/f'experiments/p2_{kind}24_fourway_eval_20261027')/a for a in ARMS for v in ['fixed','guided']}
 rows={k:{(r['sample_id'],r['sample_seed']):r for r in read(path/'omni/results.json')['rows']} for k,path in sources.items()}
 sids=[r['sample_id'] for r in read(base/'observations.json')];p=read(base/'E0/omni/protocol.json');sections={}
 for scope,keys in [('paired',list(sources)),('fixed',[a+'-fixed' for a in ARMS]),('guided',[a+'-guided' for a in ARMS])]:
  result=dict(values={k if scope=='paired' else k.split('-')[0]:{} for k in keys},coverage={},details={})
  for metric in ['fvd','utmos','sync']:
   ids=[sid for sid in sids if all(_supported(rows[k][sid,s]['metrics'][metric]) for k in keys for s in SEEDS)]
   result['coverage'][metric]=len(ids);result['details'][metric]=dict(sample_ids=ids,samples_per_seed=len(ids),planned_samples=24,conditions={})
   for key in keys:
    selected=[rows[key][sid,s] for sid in ids for s in SEEDS]
    scores=aggregate(dict(p,metrics=[metric],sample_count=len(ids)),selected)['per_seed'] if ids else None
    label=key if scope=='paired' else key.split('-')[0]
    result['details'][metric]['conditions'][label]=scores
    if metric=='fvd':result['values'][label]['FVD-32']=mean(scores,metric)
    elif metric=='utmos':result['values'][label]['UTMOSv2']=mean(scores,metric)
    else:result['values'][label].update({'LSE-D':mean(scores,metric,'lse_d'),'LSE-C':mean(scores,metric,'lse_c')})
  sections[scope]=result
 result=sections['paired'];result.update(within_fixed=sections['fixed'],within_guided=sections['guided'],
     schema='p2-english-guided-eight-condition-comparison-v1',dataset=kind,seeds=SEEDS,
     planned_outputs_per_condition=48,target_annotation_access=True,guidance_language='English',
     guidance_generation_manifest=str(OUT/'jobs.json'),guidance_generation_manifest_sha256=sha(OUT/'jobs.json'))
 return result
def main():
 work=OUT/'reactions';scope=read(work/'scope.json')
 for kind in ['train','test']:
  base=ROOT/f'experiments/p2_guided_en_{kind}24_fourway_eval_20261027'
  cached=base/'av_comparison.json'
  if cached.exists():original=read(cached)
  else:original=av(kind);atomic_json(cached,original)
  for mode in ['strict','unfiltered']:
   result=copy.deepcopy(original);ungated=mode=='unfiltered'
   version='unfiltered_v1' if ungated else 'rgb_v2'
   result.update(detector_version='rgb_no_quality_gate_v1' if ungated else 'rgb_v2',quality_gates_applied=not ungated,
                 detector_policy=read(work/'protocol.json')['unfiltered_policy' if ungated else 'strict_policy'],
                 glare_pair_coverage=scope['coverage'][kind],reaction_scope=str(work/'scope.json'))
   for section_name,target in [('paired',result),('fixed',result['within_fixed']),('guided',result['within_guided'])]:
    item=scope['scopes'][kind][mode][section_name];n=item['samples_per_seed'];target['coverage']['glare']=n
    details=dict(sample_ids=item['sample_ids'],samples_per_seed=n,planned_samples=24,conditions={})
    keys=list(scope['coverage'][kind]) if section_name=='paired' else [a+'-'+section_name for a in ARMS]
    for condition in keys:
     label=condition if section_name=='paired' else condition.split('-')[0]
     if n:
      scores={s:read(work/'scores'/(key+'.json'))['result'] for s,key in item['conditions'][condition].items()}
      details['conditions'][condition]=dict(per_seed=scores,aggregation=aggregate_seeds(scores))
      target['values'][label].update(details['conditions'][condition]['aggregation']['mean'])
     else:target['values'][label].update({k:None for k in ['R-F1','R-tIoU','R-ATD','R-FID']})
    target['details']['glare']=details
   result['scope']='Full planned 24 samples per seed for GLARE; AV scopes are metric-specific' if ungated else 'Strict full-frame detection, shared support across conditions and seeds'
   atomic_json(base/f'comparison_{version}.json',result)
   title='去额外门槛版' if ungated else '严格逐帧版'
   lines=[f'# 全英文标注指导评测：{title}','',f'数据集：{kind}；每条件24条×2seed=48输出，四组权重×固定/全英文标注指导。',
     '指导版已全部重新编码Gemma、重新生成并评测。原始英文描述和时间不变；10个事件的中文发声词按用户授权翻译，原文及对应记录保留。',
     '两版使用同一套RGB检测原始结果、相同的事件规则和GLARE公式，FVD/UTMOS/SyncNet分数不随GLARE口径切换而变化。',
     ('去掉整段全帧通过、唯一FER人脸和框IoU门槛；多脸选FER返回的第一张脸。缺失信号沿用现有规则的零值、detected mask与姿态插值。这是去门槛诊断，不是官方标准或人工真值。' if ungated else 'GT与生成视频每帧MediaPipe、FER单人脸及IoU≥0.1均通过，再取同样本、双seed共同集合。'),
     '参考数据筛选脚本也包含完整检测门槛；不能将去门槛版称为参考工作的官方评测流程。']
   for title,item in [('八条件共同比较',result),('固定prompt内部四组',result['within_fixed']),('英文指导内部四组',result['within_guided'])]:
    lines+=['','## '+title,'',str(item['coverage']),'','| 条件 | '+' | '.join(COLS)+' |','|---|'+'---:|'*len(COLS)]
    for key,v in item['values'].items():lines.append('| '+key+' | '+' | '.join('N/A' if v.get(c) is None else f'{v[c]:.4f}' for c in COLS)+' |')
   lines+=['','## 全部48配对的检测诊断','','| 条件 | 已处理 | 严格通过 | GT FER漏检帧 | 生成FER漏检帧 | GT MP漏检帧 | 生成MP漏检帧 | 每侧总帧 |','|---|---:|---:|---:|---:|---:|---:|---:|']
   fields=['processed_pairs','strict_valid_pairs','gt_fer_no_face_frames','generated_fer_no_face_frames','gt_mediapipe_no_face_frames','generated_mediapipe_no_face_frames','frames_per_side']
   for key,q in scope['coverage'][kind].items():lines.append('| '+key+' | '+' | '.join(str(q[f]) for f in fields)+' |')
   lines+=['','R-ATD无匹配事件等情况保留N/A。R-FID按每seed的反应帧池计算，不能平均逐视频分数。不同支持集合的表不可跨表直接作差。指导访问目标标注，不代表仅凭speaker自主回应能力。']
   (base/f'report_{version}.md').write_text('\n'.join(lines)+'\n')
   print(kind,mode,result['coverage'],flush=True)
 atomic_json(OUT/'publication_ready.json',dict(complete=True,english_only=True,outputs=384,glare_protocols=['strict','unfiltered']))
if __name__=='__main__':main()
