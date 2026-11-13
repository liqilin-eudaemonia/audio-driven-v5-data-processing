"""Equal-weight means of two complete256-seed evaluations, never pooled512 FID/FVD."""
import json,math,statistics,time
from pathlib import Path
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
SOURCES={137:ROOT/'experiments/p2_joint_qwen_ltx_test256_16gpu_20261113',238:ROOT/'experiments/p2_joint_qwen_ltx_test256_seed238_16gpu_20261113'}
COLUMNS=[('psnr','PSNR ↑',True),('ssim','SSIM ↑',True),('lpips','LPIPS ↓',False),('fid','FID ↓',False),('fvd','FVD ↓',False),('emo_acc','EmoAcc ↑',True),('delta_resp','ΔResp ↓',False),('audio.acoustic.missed_response_rate','漏回应率 ↓',False),('audio.acoustic.false_sound_clip_rate','静默误声率 ↓',False)]
EXTRA=['emo_div','utmos','audio.acoustic.strict.f1','audio.acoustic.tolerant.f1','audio.acoustic.extra_activity_rate','audio.acoustic.duration_mae_s','audio.speech.missed_response_rate','audio.speech.false_sound_clip_rate','audio.speech.strict.f1','audio.speech.tolerant.f1']
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def value(scores,name):
 if not name.startswith('audio.'):return scores['metrics'].get(name)
 parts=name.split('.')[1:];v=scores['audio']['activity']
 for k in parts:
  if not isinstance(v,dict):return None
  v=v.get(k)
 return v

def statistics2(values):
 valid=[v for v in values.values() if isinstance(v,(int,float)) and math.isfinite(v)]
 complete=len(valid)==len(values)==2
 return dict(by_seed=values,available_seeds=len(valid),mean=statistics.mean(valid) if complete else None,sample_std=statistics.stdev(valid) if complete else None,absolute_seed_difference=abs(valid[1]-valid[0]) if complete else None,minimum=min(valid) if complete else None,maximum=max(valid) if complete else None)

def aggregate():
 outputs={};plans={}
 for seed,w in SOURCES.items():
  assert read(w/'COMPLETE.json')['success'];plans[seed]=read(w/'PLAN.json');assert plans[seed]['seed']==seed
  outputs[seed]=read(w/'RESULTS.json')['results'];assert len(outputs[seed])==4
 assert set(outputs[137])==set(outputs[238]);assert read(SOURCES[137]/'shared_items.json')==read(SOURCES[238]/'shared_items.json')
 rows={}
 for key in outputs[137]:
  for seed in SOURCES:
   r=outputs[seed][key]['scores'];assert all(r['coverage'][m]['pairs']==256 for m in ['psnr','ssim','lpips','fid','fvd']) and r['audio']['valid_pairs']==256 and r['audio']['pairs_with_errors']==0
  old=outputs[137][key];rows[key]=dict(qwen=old['qwen'],ltx=old['ltx'],metrics={name:statistics2({str(seed):value(outputs[seed][key]['scores'],name) for seed in SOURCES}) for name in [c[0] for c in COLUMNS]+EXTRA},coverage={str(seed):outputs[seed][key]['scores']['coverage'] for seed in SOURCES},audio_counts={str(seed):{kind:{f:outputs[seed][key]['scores']['audio']['activity'][kind].get(f) for f in ['pairs','active_reference_pairs','silent_reference_pairs','missed_entire_response','false_activity_on_silent_clip']} for kind in ['acoustic','speech']} for seed in SOURCES})
 rankings={}
 for name,label,higher in COLUMNS:
  valid=[k for k in rows if rows[k]['metrics'][name]['mean'] is not None];rankings[name]=sorted(valid,key=lambda k:rows[k]['metrics'][name]['mean'],reverse=higher)
 result=dict(at=time.time(),seeds=[137,238],samples_per_seed=256,combinations=4,total_generated_evaluations=2048,aggregation='equal weight mean of each seed full256 metric; sample std over2 seeds; no pooled512 FID/FVD',no_arbitrary_cross_metric_composite=True,results=rows,rankings_by_metric=rankings)
 save(WORK/'RESULTS.json',result)
 lines=['# 四组256条：双seed平均结果','', 'Seed137与238，各256条。每个单元格为均值 ± 两个seed的样本标准差；第一名加粗，第二名下划线。','', '| Qwen | LTX | '+' | '.join(c[1] for c in COLUMNS)+' |','|---|---|'+'---:|'*len(COLUMNS)]
 for key,r in rows.items():
  cells=[]
  for name,_,_ in COLUMNS:
   s=r['metrics'][name];v=s['mean']
   if v is None:cells.append('N/A');continue
   text=f"{v:.4f}";rank=rankings[name].index(key)
   if rank==0:text='**'+text+'**'
   elif rank==1:text='<u>'+text+'</u>'
   cells.append(text+f" ± {s['sample_std']:.4f}")
  lines.append(f"| {r['qwen'] or '原生'} | {r['ltx']} | "+' | '.join(cells)+' |')
 lines+=['','各项均值用于选择候选；不将不同单位的指标直接相加。若存在指标取舍，保留取舍说明，不自动伪造统一赢家。两个seed的标准差只描述本次波动，不是统计显著性证明。','', 'FID／FVD先分别对每个seed的全部256条合并特征计算，再取两个结果的算术平均。未把两轮视作512条独立来源。N/A不补零；只有两个seed都有数值才给出均值。表情／ΔResp的有效覆盖见JSON。','', '[Seed137结果](../p2_joint_qwen_ltx_test256_16gpu_20261113/RESULTS.md) · [Seed238结果](../p2_joint_qwen_ltx_test256_seed238_16gpu_20261113/RESULTS.md)']
 (WORK/'RESULTS.md').write_text('\n'.join(lines)+'\n');save(WORK/'COMPLETE.json',dict(success=True,at=time.time(),seeds=[137,238],combinations=4,samples=256));return result
if __name__=='__main__':aggregate()
