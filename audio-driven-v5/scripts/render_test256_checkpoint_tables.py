"""Read completed, matched test256 scores; keep seed scopes separate."""
import json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EX=ROOT/'experiments'
SEEDS={137:EX/'p2_joint_qwen_ltx_test256_16gpu_20261113',238:EX/'p2_joint_qwen_ltx_test256_seed238_16gpu_20261113'}
MEAN=EX/'p2_joint_qwen_ltx_test256_two_seeds_20261113'
GLARE=EX/'glare_test256_3nodes_20261113'
ORDER=['qwen7500_ltx6800','qwen8256_ltx6800','qwen2752_ltx6000','qwen0_ltx6800']
# The raw baseline key is resolved by qwen identity below (historical name may differ).
LABELS={'qwen7500_ltx6800':'Qwen7500＋LTX6800','qwen8256_ltx6800':'Qwen8256＋LTX6800','qwen2752_ltx6000':'Qwen2752＋LTX6000','qwen0_ltx6800':'原生Qwen＋LTX6800'}
TITLES={137:'Seamless：固定256条 · seed137（含GLARE）',238:'Seamless：固定256条 · seed238',0:'Seamless：固定256条 · 双seed均值（137／238）'}
ANCHORS={TITLES[137]:'seamless256-seed137',TITLES[238]:'seamless256-seed238',TITLES[0]:'seamless256-mean'}
NOTES={TITLES[137]:'同一固定test256、seed137。Qwen7500＋LTX6800相对GLARE10618：FID降低6.07%，ΔResp降低23.35%；GLARE的PSNR、SSIM、LPIPS、FVD、Emo-Acc更好。图像／分布均覆盖256条；表情、反应有效覆盖不同，未重算共同检测交集。音频两项为声学活动口径（65条参考有声、191条参考静默）；GLARE不预测listener音频。该集合用于权重筛选，不是独立验证集。',TITLES[238]:'同一256条，seed238。GLARE尚无本轮seed238结果，因此不加入本表、不参与排名。采样与评分公式保持不变。',TITLES[0]:'仅四组Qwen/LTX组合：各seed完整256条指标等权平均，显示均值±两seed样本标准差；FID/FVD先各自全256条汇总，再求均值，不拼成512条，也不平均分片分数。GLARE仅seed137，不纳入双seed排名。Emo-Div显示生成／对应GT均值，按二者距离排名。两seed不足以证明统计显著性。'}
SPECS=[('PSNR ↑','psnr','up'),('SSIM ↑','ssim','up'),('LPIPS ↓','lpips','down'),('FID ↓','fid','down'),('FVD ↓','fvd','down'),('Emo-Acc ↑','emo_acc','up'),('Emo-Div<br>生成／GT ≈GT','emo_div_generated','gt'),('ΔResp ↓','delta_resp','down'),('漏回应率 ↓','missed_response_rate','down'),('静默误声率 ↓','false_sound_clip_rate','down')]

def read(p):return json.loads(p.read_text())
def glare_runs():
 runs=[(s,GLARE/f'step{s}',f'GLARE step{s}') for s in [10618,19050]]
 extra=EX/'glare3935_test256_after_ablation_20261113'
 if (extra/'COMPLETE.json').exists() and (extra/'step3935/COMPLETE.json').exists():
  assert read(extra/'COMPLETE.json')['success']
  runs.append((3935,extra/'step3935','GLARE step3935（早期batch8）'))
 extra2000=EX/'glare2000_test256_after_ablation_20261113'
 if (extra2000/'COMPLETE.json').exists() and (extra2000/'step2000/COMPLETE.json').exists():
  assert read(extra2000/'COMPLETE.json')['success']
  runs.append((2000,extra2000/'step2000','GLARE step2000（早期batch8）'))
 return runs
def sources():return [*(p/'RESULTS.json' for p in SEEDS.values()),MEAN/'RESULTS.json',*(p/'RESULTS.json' for _,p,_ in glare_runs())]
def value(r,key):
 if key in ('missed_response_rate','false_sound_clip_rate'):
  return r.get('audio',{}).get('activity',{}).get('acoustic',{}).get(key)
 return r['metrics'].get(key)
def coverage(r):
 c=r['coverage'];return f"{c['psnr']['pairs']}／{c['emotion']['video_identities']}／{c['delta_resp']['valid_phases']}"
def load():
 raw={}
 for seed,folder in SEEDS.items():
  assert read(folder/'COMPLETE.json')['success']
  data=read(folder/'RESULTS.json');assert data['completed_combinations']==4
  rows={}
  for item in data['results'].values():
   key=f"qwen{item['qwen']}_ltx{item['ltx']}";r=item['scores'];assert r['seed']==seed and r['requested_pairs']==256;rows[key]=r
  assert set(rows)==set(ORDER);raw[seed]=rows
 protocol=read(SEEDS[137]/'combinations/qwen7500_ltx6800/protocol.json')
 for step,folder,_ in glare_runs():
  assert read(folder/'COMPLETE.json')['success']
  p=read(folder/'protocol.json');assert p['seed']==137 and p['references_sha256']==protocol['datasets']['seamless']['references_sha256']
  assert p['shared_items_sha256']==protocol['shared_items_sha256']
 return raw

def build_sections(cells,markdown):
 raw=load();sections=[];headers=['方法／权重',*[s[0] for s in SPECS],'覆盖<br>图像／表情／ΔResp']
 if any(step==3935 for step,_,_ in glare_runs()) and '3935来自' not in NOTES[TITLES[137]]:
  NOTES[TITLES[137]]+=' GLARE3935来自早期有效batch8的独立训练线，不是batch256训练线的中间权重。'
 for seed in [137,238]:
  results=[raw[seed][key] for key in ORDER];labels=[LABELS[key] for key in ORDER]
  if seed==137:
   results += [read(folder/'RESULTS.json') for _,folder,_ in glare_runs()];labels += [label for _,_,label in glare_runs()]
  columns=[]
  for _,key,direction in SPECS:
   vals=[value(r,key) for r in results]
   scores=[math.inf if v is None else -v if direction=='up' else abs(v-r['metrics']['emo_div_gt']) if direction=='gt' else v for v,r in zip(vals,results)]
   def fmt(i,v):
    if key in ['emo_acc','missed_response_rate','false_sound_clip_rate']:return f'{v*100:.4f}%'
    if direction=='gt':return f"{v:.6f}／{results[i]['metrics']['emo_div_gt']:.6f}"
    return f'{v:.6f}'
   col=cells(vals,scores,fmt)
   if key in ['missed_response_rate','false_sound_clip_rate']:col=['N/A' if v is None else text for v,text in zip(vals,col)]
   columns.append(col)
  rows=[[label,*[c[i] for c in columns],coverage(results[i])] for i,label in enumerate(labels)]
  sections.append('## '+TITLES[seed]+'\n\n'+NOTES[TITLES[seed]]+'\n\n'+markdown(headers,rows)+'\n')
 summary=read(MEAN/'RESULTS.json');assert summary['seeds']==[137,238]
 columns=[]
 for _,key,direction in SPECS:
  vals=[];stds=[];scores=[];gts=[]
  for combo in ORDER:
   vv=[value(raw[s][combo],key) for s in [137,238]];assert all(v is not None for v in vv)
   avg=statistics.mean(vv);std=statistics.stdev(vv);vals.append(avg);stds.append(std)
   gt=statistics.mean(raw[s][combo]['metrics']['emo_div_gt'] for s in [137,238]);gts.append(gt)
   scores.append(-avg if direction=='up' else abs(avg-gt) if direction=='gt' else avg)
   if key!='emo_div_generated':
    skey='audio.acoustic.'+key if key in ['missed_response_rate','false_sound_clip_rate'] else key
    sr=next(x for x in summary['results'].values() if f"qwen{x['qwen']}_ltx{x['ltx']}"==combo)['metrics'][skey]
    assert math.isclose(avg,sr['mean'],abs_tol=1e-10) and math.isclose(std,sr['sample_std'],abs_tol=1e-10)
  def fmt(i,v):
   if key in ['emo_acc','missed_response_rate','false_sound_clip_rate']:return f'{v*100:.4f}% ± {stds[i]*100:.4f}pp'
   if direction=='gt':return f'{v:.6f}／{gts[i]:.6f}'
   return f'{v:.6f} ± {stds[i]:.6f}'
  columns.append(cells(vals,scores,fmt))
 rows=[[LABELS[key],*[c[i] for c in columns],'<br>'.join(f'{s}: '+coverage(raw[s][key]) for s in [137,238])] for i,key in enumerate(ORDER)]
 sections.append('## '+TITLES[0]+'\n\n'+NOTES[TITLES[0]]+'\n\n'+markdown(headers,rows)+'\n')
 return '\n'.join(sections)+'\n'
