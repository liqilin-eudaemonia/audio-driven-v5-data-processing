"""Publish the completed, corrected original-test24 batch alongside the old review."""
import copy, concurrent.futures, importlib.util, json, math, shutil, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,atomic_json
from p2_omni_eval.scoring import aggregate
from p2_omni_eval.compare import _supported
import build_p2_av_review as media
OLD=media.OUT;OUT=OLD/'reintegrated_20261028';OUT.mkdir(exist_ok=True)
media.OUT=OUT;media.MEDIA=OUT/'media';media.MEDIA.mkdir(exist_ok=True)
ARMS=['legacy_E0','legacy_E1','legacy_E0F','legacy_E1F','guided_av_step1249','guided_av_event_step1249']
NAMES=['AV','AV＋Event','AV＋Feature','AV＋Event＋Feature','AV','AV＋Event']
SEEDS=[137,251];COLS=['FVD-32','UTMOSv2','LSE-D','LSE-C','R-F1','R-tIoU','R-ATD','R-FID']
def mean(scores,metric,field=None):
 if not scores:return None
 vals=[]
 for seed in SEEDS:
  v=scores[str(seed)][metric];v=v['full_value'] if metric=='fvd' else v['generated']['full_value'];vals.append(v.get(field) if field and v is not None else v)
 return sum(vals)/len(vals) if all(v is not None for v in vals) else None

def main():
 assert read(HERE/'cluster24_test24/AVAILABLE_COMPLETE.json')['outputs']==288
 jobs=[j for j in read(HERE/'jobs.test24.json') if j['arm'] in ARMS]
 old=next(d for d in read(OLD/'data.json')['datasets'] if d['id']=='test')
 obs={o['sample_id']:o for o in read(HERE/'test24/observations.json')};ids=[s['id'] for s in old['samples']]
 assert set(ids)==set(obs) and len(ids)==24 and len(jobs)==288
 expected={(a,s,seed) for a in ARMS for s in ids for seed in SEEDS}
 assert {(j['arm'],j['sample_id'],j['seed']) for j in jobs}==expected
 results={};glare={};rows={};work={};hashes={}
 for arm in ARMS:
  work[arm]=HERE/'metrics'/arm/'test24';done=read(work[arm]/'COMPLETE.json');assert done['outputs']==48 and done['omni_status']=='complete'
  for name,h in done['results'].items():assert sha(work[arm]/name)==h
  results[arm]=read(work[arm]/'omni/results.json');glare[arm]=read(work[arm]/'glare_results.json')['results']
  rows[arm]={(r['sample_id'],r['sample_seed']):r for r in results[arm]['rows']}
  assert set(rows[arm])=={(s,seed) for s in ids for seed in SEEDS}
  hashes[arm]=done['results']
 # Aggregate the same IDs across all six conditions and both seeds for each AV metric.
 protocol=read(work[ARMS[0]]/'omni/protocol.json');values={a:{} for a in ARMS};coverage={};details={}
 for metric in ['fvd','utmos','sync']:
  common=[s for s in ids if all(_supported(rows[a][s,seed]['metrics'][metric]) for a in ARMS for seed in SEEDS)]
  coverage[metric]=len(common);details[metric]={'sample_ids':common,'conditions':{}}
  for a in ARMS:
   scores=aggregate(dict(protocol,metrics=[metric],sample_count=len(common)),[rows[a][s,seed] for s in common for seed in SEEDS])['per_seed'] if common else None
   details[metric]['conditions'][a]=scores
   if metric=='fvd':values[a]['FVD-32']=mean(scores,metric)
   elif metric=='utmos':values[a]['UTMOSv2']=mean(scores,metric)
   else:values[a].update({'LSE-D':mean(scores,metric,'lse_d'),'LSE-C':mean(scores,metric,'lse_c')})
  print('common metric',metric,len(common),flush=True)
 protocols={}
 for mode in ['unfiltered','strict']:
  table=[]
  for a,name in zip(ARMS,NAMES):
   g=glare[a][mode];v=dict(values[a]);v.update(g['aggregation']['mean'] if g['aggregation'] else {c:None for c in COLS[4:]})
   if mode=='unfiltered':assert set(g['sample_ids'])==set(ids)
   table.append(dict(id=a,name=name,variant='固定prompt训练 · step1268' if a.startswith('legacy') else '标注指导训练 · step1249',outputs=48,values=v,glare_samples=g['samples_per_seed'],scope='AV指标为六条件双seed共同支持；GLARE去门槛版全24条，严格版为各条件自身双seed交集'))
  protocols[mode]={'metrics':table,'coverage':coverage,'report':'report.md'}
 samples=[];tasks=[];by={(j['arm'],j['sample_id'],j['seed']):j for j in jobs}
 for s0 in old['samples']:
  s=copy.deepcopy(s0);sid=s['id'];s['outputs']=[]
  for key in ['image','speaker','gt_audio','gt_video','metric_gt_audio']:
   if s.get(key):s[key]='../'+s[key]
  assert s['sources']['speaker']==obs[sid]['speaker_audio']
  for seed in SEEDS:
   prompts=[]
   for arm,name in zip(ARMS,NAMES):
    j=by[arm,sid,seed];out=Path(j['output']);assert (out/'complete.json').exists()
    wave,_=sf.read(out/'listener.wav');rms=float(np.sqrt(np.mean(wave*wave)));mr=rows[arm][sid,seed]['metrics']
    prompt=(out/'prompt.txt').read_text();prompts.append(prompt)
    assert not any('\u3400'<=c<='\u9fff' for c in prompt)
    entry=dict(arm=arm,name=name,variant='fixed' if arm.startswith('legacy') else 'guided',seed=seed,video=media.link(out/'generated.mp4'),audio=media.link(out/'listener.wav'),
      rms=rms,metrics=dict(utmos=mr['utmos']['generated'].get('value'),lse_d=mr['sync']['generated'].get('lse_d'),lse_c=mr['sync']['generated'].get('lse_c'),rms_dbfs=20*math.log10(max(rms,1e-15))),
      metric_status='evaluated',prompt=prompt,guidance_language='English',checkpoint=j['checkpoint'])
    s['outputs'].append(entry);tasks.append((entry,out/'generated.mp4',out/'listener.wav',s['duration'],obs[sid]['speaker_audio']))
   assert len(set(prompts))==1,'Gemma guidance differs across conditions'
  samples.append(s)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures={pool.submit(media.mux,v,a,d,sp):obj for obj,v,a,d,sp in tasks}
  for n,f in enumerate(concurrent.futures.as_completed(futures),1):
   futures[f]['mixed_video']=f.result()
   if n%48==0:print('media',n,'/288',flush=True)
 dataset=dict(id='test',title='原独立test24 · 重新整合标注',samples=samples,glare_protocols=protocols,coverage=coverage,metrics=protocols['unfiltered']['metrics'],report='report.md')
 atomic_json(OUT/'data.json',dict(schema='p2-reintegrated-review-v1',datasets=[dataset]))
 atomic_json(OUT/'comparison.json',dict(coverage=coverage,details=details,glare_protocols=protocols,source_hashes=hashes))
 lines=['# 重新整合标注 · 原独立test24评测','', '6组×原test24×seed137/251＝288输出。旧4组固定prompt训练权重step1268，仅替换推理Gemma指导文本；新2组标注指导训练权重step1249，同一指导进入Qwen hidden与Gemma。所有组实际Gemma文本逐条相同。', '', 'GT、speaker、首帧、采样设置与原test24相同。旧/新权重训练数据及条件链路不同，跨批差异不能单独归因于文本训练。目标标注指导属于oracle诊断，不代表自主规划能力。', '', '各AV指标使用6条件和两个seed共同支持。GLARE去门槛版使用全部24条；严格版展示各条件自身有效子集，子集不同不能直接排名。静音/不适用保留N/A，不填0。', '', 'AV共同支持：'+str(coverage), '', '新版AV＋Feature最终step1249尚未评测，不使用step200/587代替。']
 for mode,item in protocols.items():
  lines+=['','## '+mode,'','| Pipeline | GLARE支持/24 | '+' | '.join(COLS)+' |','|---|---:|'+'---:|'*len(COLS)]
  for r in item['metrics']:lines.append('| '+r['variant']+' / '+r['name']+' | '+str(r['glare_samples'])+' | '+' | '.join('N/A' if r['values'].get(k) is None else f'{r["values"][k]:.4f}' for k in COLS)+' |')
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
 shutil.copy2(OLD/'style.css',OUT/'style.css')
 html=(OLD/'index.html').read_text()
 html=html.replace('<h1>听见输入，对照回应</h1>','<h1>重新整合标注 · 原独立test24</h1><p><a href="../">返回历史评测</a> · 本批6组288输出已完成。旧4组仅更换Gemma指导；新2组同时使用计划Qwen hidden与Gemma。新版AV＋Feature等待最终权重评测。</p>')
 html=html.replace('<button data-dataset="train" aria-pressed="true">训练集24条</button>','').replace('固定prompt＋英文标注指导','全部6组训练权重').replace('仅固定prompt','固定prompt训练（4组）').replace('仅英文标注指导','标注指导训练（2组）')
 html=html.replace('固定／指导八条件同样本配对','六条件同样本配对').replace('各prompt内部四组权重比较','统一比较口径')
 html=html.replace('表格为固定评测范围内的两个seed均值','表格为两个seed等权均值；AV使用六条件共同支持，严格GLARE支持数另列').replace('固定与指导条件采用相同样本、seed和采样参数','所有条件采用相同test24、seed和采样参数')
 (OUT/'index.html').write_text(html)
 js=(OLD/'app.js').read_text().replace("dataset:'train'","dataset:'test'").replace("dataset('train')","dataset('test')")
 # Reuse tested audio toggle, ASR, scrolling, seed controls; relabel groups by TRAINING condition.
 js=js.replace("'英文标注指导 · 已完成结果'","'标注指导训练 · step1249 · 已完成结果'").replace("'标注指导'","'标注指导训练'").replace("'固定prompt'","'固定prompt训练'")
 js+='\n'+(HERE/'review_metrics.js').read_text()
 (OUT/'app.js').write_text(js)
 summary=dict(complete=True,at=time.time(),outputs=288,samples=24,conditions=6,seeds=SEEDS,coverage=coverage,url='http://127.0.0.1:8876/p2_av_review/reintegrated_20261028/',source_hashes=hashes)
 atomic_json(OUT/'publication.json',summary);atomic_json(HERE/'HTML_PUBLISHED.json',summary)
 index=OLD/'index.html';original=index.read_text();backup=OLD/'index.before_reintegrated_20261028.html'
 if not backup.exists():backup.write_text(original)
 banner='<div class="review-note" id="reintegrated-batch"><strong>新增：重新整合标注后的同一test24</strong> · 6组288输出及完整指标已完成。<a href="reintegrated_20261028/">打开本批对照结果 →</a></div>'
 if 'id="reintegrated-batch"' not in original:
  temp=index.with_suffix('.tmp');temp.write_text(original.replace('<nav class="toolbar"',banner+'\n<nav class="toolbar"',1));temp.replace(index)
 print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
