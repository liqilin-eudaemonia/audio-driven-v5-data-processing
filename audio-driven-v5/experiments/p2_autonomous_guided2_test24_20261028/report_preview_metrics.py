"""Reaggregate exactly matched IDs and both seeds; do not compare full24 with3."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];WORK=HERE/'preview_metrics';sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,sha,digest,atomic_json
from p2_omni_eval.common import artifact_valid,job_key
from p2_omni_eval.pipeline import check_protocol,has_error
from p2_omni_eval.scoring import aggregate
from p2_omni_eval.compare import _supported
from p2_glare_eval.scoring import aggregate_seeds
ARMS=['oracle_av','auto_av','oracle_av_event','auto_av_event'];NAMES=['AV · 真实标注指导','AV · Qwen自主规划','AV＋Event · 真实标注指导','AV＋Event · Qwen自主规划'];COLS=['FVD-32','UTMOSv2','LSE-D','LSE-C','R-F1','R-tIoU','R-ATD','R-FID'];SEEDS=[137,251]

def mean(scores,metric,field=None):
 if scores is None:return None
 values=[]
 for seed in SEEDS:
  value=scores[str(seed)][metric];value=value['full_value'] if metric=='fvd' else value['generated']['full_value']
  values.append(value[field] if value is not None and field else value)
 return sum(values)/len(values) if all(v is not None for v in values) else None

def main():
 scope=read(WORK/'scope.json');ids=scope['sample_ids'];rows={};protocols={};source_receipts={};native={}
 assert read(WORK/'queue/AVAILABLE_COMPLETE.json')['status']=='complete'
 for arm in ARMS:
  if arm.startswith('auto'):
   folder=WORK/arm/'omni';p=read(folder/'protocol.json');check_protocol(p);selected=[]
   for planned in p['rows']:
    dest=folder/'rows'/job_key(planned['sample_id'],planned['sample_seed']);expected=dict(protocol=digest(p),row=planned,device='cuda:0');assert artifact_valid(dest,expected);selected.append(read(dest/'result.json'))
   assert not has_error(selected)
   result=aggregate(p,selected);result.update(protocol_sha256=digest(p),device='cuda:0');atomic_json(folder/'results.json',result)
   source_receipts[arm]=dict(results_sha256=sha(folder/'results.json'),protocol_sha256=digest(p))
  else:
   ref=scope['oracle_sources'][arm];folder=Path(ref['path']);assert sha(folder/'results.json')==ref['results_sha256'] and sha(folder/'protocol.json')==ref['protocol_sha256']
   p=read(folder/'protocol.json');check_protocol(p);result=read(folder/'results.json');assert result['protocol_sha256']==digest(p)
   selected=[r for r in result['rows'] if r['sample_id'] in ids]
   for row in selected:
    planned=next(r for r in p['rows'] if (r['sample_id'],r['sample_seed'])==(row['sample_id'],row['sample_seed']));dest=folder/'rows'/job_key(row['sample_id'],row['sample_seed']);assert artifact_valid(dest,dict(protocol=digest(p),row=planned,device=result['device']))
   source_receipts[arm]=ref
  assert {(r['sample_id'],r['sample_seed']) for r in selected}=={(s,k) for s in ids for k in SEEDS}
  protocols[arm]=p;rows[arm]={(r['sample_id'],r['sample_seed']):r for r in selected}
  for key in ['metrics','options','seeds','runtime','code']:assert p[key]==protocols[ARMS[0]][key]
  native[arm]=aggregate(dict(p,sample_count=3),selected)
  atomic_json(WORK/arm/'subset_results.json',native[arm])
 values={a:{} for a in ARMS};details={};coverage={}
 for metric in ['fvd','utmos','sync']:
  common=[s for s in ids if all(_supported(rows[a][s,k]['metrics'][metric]) for a in ARMS for k in SEEDS)];coverage[metric]=len(common);details[metric]=dict(sample_ids=common,conditions={})
  for arm in ARMS:
   p=dict(protocols[arm],metrics=[metric],sample_count=len(common));scores=aggregate(p,[rows[arm][s,k] for s in common for k in SEEDS])['per_seed'] if common else None;details[metric]['conditions'][arm]=scores
   if metric=='fvd':values[arm]['FVD-32']=mean(scores,metric)
   elif metric=='utmos':values[arm]['UTMOSv2']=mean(scores,metric)
   else:values[arm].update({'LSE-D':mean(scores,metric,'lse_d'),'LSE-C':mean(scores,metric,'lse_c')})
 reaction=read(WORK/'reaction_scope.json');modes={}
 for mode in ['unfiltered','strict']:
  scores={};table=[];members=reaction[mode+'_ids']
  for arm,name in zip(ARMS,NAMES):
   per={str(k):read(WORK/'reaction_scores'/f'{arm}.{mode}.{k}.json')['result'] for k in SEEDS}
   agg=aggregate_seeds(per) if members else None;scores[arm]=dict(per_seed=per,aggregation=agg)
   rowvalues=dict(values[arm]);rowvalues.update(agg['mean'] if agg else {k:None for k in COLS[4:]});table.append(dict(id=arm,name=name,outputs=6,values=rowvalues))
  modes[mode]=dict(rows=table,coverage=dict(coverage,glare=len(members)),glare_sample_ids=members,glare_scores=scores)
 audio={}
 for arm in ARMS:
  audio[arm]=[]
  for sid in ids:
   for seed in SEEDS:
    r=rows[arm][sid,seed];audio[arm].append(dict(sample_id=sid,seed=seed,audio=r.get('audio'),utmos=r['metrics']['utmos'],sync=r['metrics']['sync']))
 result=dict(status='complete',sample_ids=ids,seeds=SEEDS,outputs_per_condition=6,new_autonomous_outputs=12,comparison_conditions=4,details=details,modes=modes,audio_diagnostics=audio,source_receipts=source_receipts,scope=scope)
 atomic_json(WORK/'comparison.json',result)
 lines=['# 自主规划3条样本：与真实标注指导同样本配对','', '两种指导×AV/AV＋Event，共4条件，每条件3条×2seed＝6个输出。新增自主规划12个视频；真实标注指导复用原相同样本的特征与检测结果重新汇总。','', '仅原test24中3条规划通过的样本，不代表完整测试集或自主规划成功率。模型均为guided、冻结Qwen的最终step1249；同GT、首帧、speaker、seed和采样设置。计分使用独立listener.wav，不使用叠加speaker的音轨。','', '每项AV指标在4条件×2seed之间取共同支持；GLARE严格版取相同共同有效ID，去门槛版使用全部3条。每seed先计算后等权平均，不能把6输出当作6个独立源样本。']
 for mode in ['unfiltered','strict']:
  item=modes[mode];lines+=['','## '+('去额外门槛版' if mode=='unfiltered' else '严格逐帧版'),'', '共同支持（每seed）：'+str(item['coverage']),'','| 条件 | '+' | '.join(COLS)+' |','|---|'+'---:|'*len(COLS)]
  for row in item['rows']:lines.append('| '+row['name']+' | '+' | '.join('N/A' if row['values'].get(k) is None else f'{row["values"][k]:.4f}' for k in COLS)+' |')
 lines+=['','## 指标覆盖与音轨诊断','','N/A表示不适用或没有共同有效样本，不表示0分；逐条原因保存在comparison.json的audio_diagnostics。FVD和R-FID在3条样本上方差很大，只用于诊断。R-F1等取决于自动反应检测，不能直接视为人工判断的回应合理性。','','| 条件 | seed | 样本 | GT RMS | 生成RMS | 生成UTMOS状态 | 生成Sync状态 |','|---|---:|---|---:|---:|---|---|']
 for arm,name in zip(ARMS,NAMES):
  for d in audio[arm]:lines.append('| '+name+' | '+str(d['seed'])+' | '+d['sample_id']+' | '+str(d['audio']['gt'].get('rms'))+' | '+str(d['audio']['generated'].get('rms'))+' | '+d['utmos']['generated']['status']+' | '+d['sync']['generated']['status']+' |')
 (WORK/'report.md').write_text('\n'.join(lines)+'\n')
 print('COVERAGE',modes['unfiltered']['coverage']);print('STRICT',modes['strict']['coverage'])
 for row in modes['unfiltered']['rows']:print(row['name'],row['values'])
if __name__=='__main__':main()
