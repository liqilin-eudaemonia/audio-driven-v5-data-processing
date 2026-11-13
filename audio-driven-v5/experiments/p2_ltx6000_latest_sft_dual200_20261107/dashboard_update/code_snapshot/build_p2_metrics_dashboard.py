"""Publish result-derived system numbers and source-checked ablation deltas."""
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
RN=ROOT/'experiments/p2_responsenet_random200_twelveway_20261105'
SEAM=ROOT/'experiments/p2_seamless200_tenway_24gpu_20261105'
WHISPER=ROOT/'experiments/p2_seamless200_whisper_20261105'
JOINT=ROOT/'experiments/p2_combined_whisper_eval_20261106'
LATEST=ROOT/'experiments/p2_ltx6000_latest_sft_dual200_20261107'
PAGE=ROOT/'data_previews/listen_benchmark_review/p2_responsenet_random200_20261105'
LABELS={
 'av_only_6000':'AV · 6000','av_only_8927':'AV · 8927',
 'av_event_6000':'AV＋Event · 6000','av_event_8927':'AV＋Event · 8927',
 'base_feature_6000':'AV＋Feature · 6000','base_feature_8927':'AV＋Feature · 8927',
 'base_event_feature_7000':'AV＋Event＋Feature · 7000',
 'sft_event_6000':'SFT H＋Event · 6000','sft_event_8927':'SFT H＋Event · 8927',
 'sft_on_base_event_6000':'SFT推理Qwen＋原始H训练Event · 6000',
 'whisper_av_6000':'AV＋Whisper直连 · 6000','whisper_av_8000':'AV＋Whisper直连 · 8000',
 'base_event_feature_whisper_6000':'AV＋Event＋Feature＋Whisper · 6000',
 'base_event_feature_whisper_8000':'AV＋Event＋Feature＋Whisper · 8000',
 'ltx_hidden_event_feature_whisper_6000_sft2752':'主干Event＋Feature＋Whisper · 6000＋最新SFT2752'}
ORDER=['av_only_6000','av_event_6000','base_feature_6000','base_event_feature_7000',
 'sft_on_base_event_6000','sft_event_6000','whisper_av_6000',
 'av_only_8927','av_event_8927','base_feature_8927','sft_event_8927','whisper_av_8000',
 'base_event_feature_whisper_6000','base_event_feature_whisper_8000',
 'ltx_hidden_event_feature_whisper_6000_sft2752']
METRICS=[
 dict(key='psnr',label='PSNR',direction=1,digits=4,unit='dB'),
 dict(key='ssim',label='SSIM',direction=1,digits=5,unit=''),
 dict(key='fid',label='FID',direction=-1,digits=4,unit=''),
 dict(key='fvd',label='FVD',direction=-1,digits=4,unit=''),
 dict(key='lpips',label='LPIPS',direction=-1,digits=5,unit=''),
 dict(key='emo_acc',label='Emo-Acc',direction=1,digits=3,unit='%',percent=True),
 dict(key='emo_div_gap',label='Emo-Div距GT',direction=-1,digits=4,unit=''),
 dict(key='delta_resp',label='ΔResp',direction=-1,digits=4,unit=''),
 dict(key='strict_audio_f1',label='声学活动F1',direction=1,digits=3,unit='%',percent=True),
 dict(key='missed_response_rate',label='有声GT整段漏声率',direction=-1,digits=3,unit='%',percent=True),
 dict(key='false_sound_clip_rate',label='静音GT误发声率',direction=-1,digits=3,unit='%',percent=True),
 dict(key='utmos',label='UTMOS',direction=1,digits=4,unit=''),
 dict(key='emo_div_generated',label='Emo-Div生成类别数',direction=0,digits=4,unit=''),
 dict(key='emo_div_gt',label='Emo-Div有效GT类别数',direction=0,digits=4,unit='')]
GROUPS=[
 dict(id='event',title='1 · 只加Event loss',note='同一步数、原生Qwen、原始H，比较AV与AV＋Event。',pairs=[
  ('event6000','6000步','av_only_6000','av_event_6000'),('event8927','8927步','av_only_8927','av_event_8927')]),
 dict(id='feature',title='2 · 只加Feature loss',note='同一步数、原生Qwen、原始H。Feature权重为视觉0.025＋音频0.025。',pairs=[
  ('feature6000','6000步','av_only_6000','base_feature_6000'),('feature8927','8927步','av_only_8927','base_feature_8927')]),
 dict(id='combined',title='3 · Event＋Feature能否继续增益',note='组合7000累计约190000次样本展示，接近其他6000步的192000；步数、batch与顺序不完全相同，只能按近似预算比较。',pairs=[
  ('combined_av','组合相对AV','av_only_6000','base_event_feature_7000'),
  ('combined_event','组合相对Event','av_event_6000','base_event_feature_7000'),
  ('combined_feature','组合相对Feature','base_feature_6000','base_event_feature_7000')]),
 dict(id='qwen',title='4 · 仅更新推理Qwen',note='使用同一个原始H训练的Event6000 LTX权重；推理Qwen换为规划SFT LoRA。规划文本和推理H同时变化，不是仅改文本。',pairs=[
  ('qwen_sft','原生→SFT推理Qwen','av_event_6000','sft_on_base_event_6000')]),
 dict(id='hidden',title='5 · 训练H也改用SFT Qwen提取',note='两组推理共享同一SFT规划文本、Gemma与H缓存；比较LTX训练H的来源。训练时长分组启用时刻也不同，并非严格单因素。',pairs=[
  ('hidden_sft','原始H训练→SFT H训练','sft_on_base_event_6000','sft_event_6000')]),
 dict(id='whisper',title='6 · 额外speaker audio cross-attention',note='保留Qwen H和Gemma文本，增加Whisper speaker直连分支。主要对照为同6000步；8000对6000只表示训练进展，不是与8927等预算对照。',pairs=[
  ('whisper6000','同6000步：增加Whisper','av_only_6000','whisper_av_6000'),
  ('whisper_progress','Whisper继续训练6000→8000','whisper_av_6000','whisper_av_8000')]),
 dict(id='joint_whisper',title='7 · Event＋Feature＋Whisper联合消融',
  note='主对照用联合6000步分别对比AV、Event、Feature、原组合监督与仅AV＋Whisper；另列同8000步的Whisper基线和联合6000→8000进展。联合8000还与历史AV／Event／Feature的8927终点对照，但只作预算不等的参考。联合batch31，6000步186000次展示，比历史batch32的192000少3.125%；原组合7000约190000；联合8000为248000次，历史8927为285664次。训练分组也有差异，不是严格单因素实验。',pairs=[
   ('joint_av','联合6000 vs 仅AV6000','av_only_6000','base_event_feature_whisper_6000'),
   ('joint_event','联合6000 vs AV＋Event6000','av_event_6000','base_event_feature_whisper_6000'),
   ('joint_feature','联合6000 vs AV＋Feature6000','base_feature_6000','base_event_feature_whisper_6000'),
   ('joint_no_whisper','原Event＋Feature上增加Whisper','base_event_feature_7000','base_event_feature_whisper_6000'),
   ('joint_whisper6000','Whisper6000上增加Event＋Feature','whisper_av_6000','base_event_feature_whisper_6000'),
   ('joint_whisper8000','Whisper8000上增加Event＋Feature','whisper_av_8000','base_event_feature_whisper_8000'),
   ('joint8000_av','联合8000 vs AV8927（预算不等）','av_only_8927','base_event_feature_whisper_8000'),
   ('joint8000_event','联合8000 vs Event8927（预算不等）','av_event_8927','base_event_feature_whisper_8000'),
   ('joint8000_feature','联合8000 vs Feature8927（预算不等）','base_feature_8927','base_event_feature_whisper_8000'),
   ('joint_progress','联合继续训练6000→8000','base_event_feature_whisper_6000','base_event_feature_whisper_8000'),
   ('latest_joint6000','最新主干＋SFT2752 vs 历史联合6000（多项变化）','base_event_feature_whisper_6000','ltx_hidden_event_feature_whisper_6000_sft2752'),
   ('latest_av6000','最新主干＋SFT2752 vs 仅AV6000（多项变化）','av_only_6000','ltx_hidden_event_feature_whisper_6000_sft2752'),
   ('latest_event6000','最新主干＋SFT2752 vs AV＋Event6000（多项变化）','av_event_6000','ltx_hidden_event_feature_whisper_6000_sft2752')])]
GROUPS[-1]['note']+=' 新增最新主干Event＋Feature＋Whisper6000＋SFT2752，与历史联合6000、AV6000及Event6000对照。新组同时改变训练数据、Event读出位置及规划器，属于pipeline比较，不能归因于某一个组件或SFT。'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dataset(key,runs,title,description):
 results={};arms={};sources={};audio_files={}
 for run in runs:
  p=read(run/'protocol.json');r=read(run/'RESULTS.json')
  anchor=read(runs[0]/'protocol.json')
  assert p['seed']==anchor['seed'] and p['generation']==anchor['generation']
  assert p['datasets'][key]['references_sha256']==anchor['datasets'][key]['references_sha256']
  assert read(run/'COMPLETE.json')['status'] in ['complete','partial']
  sources[str(run/'RESULTS.json')]=sha(run/'RESULTS.json')
  for arm in p['arms']:
   a=arm['name'];results[a]=r[a+'/'+key];arms[a]=arm;audio_files[a]=run/'metrics'/a/key/'audio_pairs.jsonl'
 p=read(runs[0]/'protocol.json');refs=Path(p['datasets'][key]['prepared'])/'references.jsonl'
 assert sha(refs)==p['datasets'][key]['references_sha256']
 row_refs=[json.loads(x) for x in refs.read_text().splitlines()];ids={x['id'] for x in row_refs};assert len(ids)==len(row_refs)==200
 rows=[]
 for a in ORDER:
  s=results[a];audio=[json.loads(x) for x in audio_files[a].read_text().splitlines()]
  assert len(audio)==200 and {x['id'] for x in audio}==ids
  ac=s['audio']['activity']['acoustic'];m=s['metrics']
  row=dict(arm=a,label=LABELS[a],step=arms[a]['step'],**m,
   emo_div_gap=abs(m['emo_div_generated']-m['emo_div_gt']),
   strict_audio_f1=ac['strict']['f1'],missed_response_rate=ac['missed_response_rate'],
   false_sound_clip_rate=ac['false_sound_clip_rate'],
   gt_active_clips=ac['active_reference_pairs'],gt_silent_clips=ac['silent_reference_pairs'],
   missed_clips=ac['missed_entire_response'],false_sound_clips=ac['false_activity_on_silent_clip'],
   generated_active_clips=sum(bool(x['audio']['generated']['acoustic']) for x in audio),
   coverage=s['coverage'],status=s['status'])
  rows.append(row)
 pairs=[];lookup={r['arm']:r for r in rows}
 for group in GROUPS:
  for pid,label,left,right in group['pairs']:
   a,b=lookup[left],lookup[right];metrics=[]
   for metric in METRICS:
    av,bv=a.get(metric['key']),b.get(metric['key']);delta=None if av is None or bv is None else bv-av
    metrics.append(dict(key=metric['key'],baseline=av,comparison=bv,delta=delta,
       display_delta=None if delta is None else delta*(100 if metric.get('percent') else 1),
       delta_unit='百分点' if metric.get('percent') else metric['unit'],
       better=None if delta is None or abs(delta)<1e-12 or metric['direction']==0 else delta*metric['direction']>0))
   pairs.append(dict(id=pid,group=group['id'],label=label,baseline=left,comparison=right,metrics=metrics))
 if key=='seamless':
  analysis=[
   '最新主干Event＋Feature＋Whisper6000＋SFT2752相对历史联合6000：FID 16.5329→14.9525、FVD 179.2737→172.5138、Emo-Acc 38.353%→43.768%改善；PSNR 16.3009→16.1165、SSIM 0.67624→0.67302、LPIPS 0.34090→0.34924退步，ΔResp基本相当，未体现全面收益。',
   '最新组音频仍有明显漏回应：134条声学有活动GT中99条没有重叠的生成活动，漏回应率73.88%。图像及分布指标200/200，表情199/200、运动198/200；这是检测支持不足，不是生成失败。',
   '最新组与历史组同时存在训练数据、Event读出位置及规划器差异；下面新增差值仅描述整条pipeline的变化，不能视为最新SFT的单变量收益。以下其余分析指历史联合6000／8000：',
   '联合6000相对仅AV6000：PSNR 15.8926→16.3009、SSIM 0.66620→0.67624、LPIPS 0.36186→0.34090；FID略变差，FVD仅小幅下降，ΔResp基本持平，不能说所有指标都有收益。',
   '联合6000相对仅AV＋Whisper6000：PSNR、SSIM、LPIPS、FID、FVD和ΔResp均未进一步改善；加入Event＋Feature没有带来一致叠加收益。',
   '联合6000→8000：FID 16.5329→16.0049、FVD 179.2737→174.6751、ΔResp 0.54748→0.49722改善；像素／感知相似度与表情一致率下降，Emo-Div距GT增大，存在取舍。',
   '有声GT整段漏声率6000约89.55%、8000约93.28%，音频回应漏检仍明显。情绪／运动各199/200，图像与分布指标均200/200。']
 else:
  analysis=[
   '最新主干Event＋Feature＋Whisper6000＋SFT2752相对历史联合6000：PSNR 17.5129→16.7695、SSIM 0.66710→0.65380、LPIPS 0.30379→0.33664、FID 12.6472→13.2560、FVD 150.7708→184.5401、Emo-Acc 61.489%→59.050%、ΔResp 0.22306→0.25639均退步；不能据此认定最新组更好。',
   '最新组13条声学有活动GT中11条没有重叠的生成活动；187条无活动GT中33条误发声。所有200条图像、表情和运动评分完成。最新与历史同时改变数据、Event读出位置及规划器，不是单变量消融。以下其余分析指历史联合6000／8000：',
   '联合6000相对仅AV6000：FVD 161.1684→150.7708、ΔResp 0.23126→0.22306改善；PSNR、SSIM、LPIPS、FID与表情一致率退化，不能归纳成全面收益。',
   '联合6000相对AV＋Event6000：FVD小幅改善，但像素、感知、FID、表情和ΔResp未改善；相对原Event＋Feature7000，七项主要指标里除表情一致率外均更差。',
   '联合6000相对仅AV＋Whisper6000：ΔResp与Emo-Div距GT改善，PSNR、SSIM、LPIPS、FID、FVD与表情一致率退化；联合8000也未稳定优于对应Whisper8000。',
   '联合6000→8000：PSNR、SSIM、LPIPS、FID、FVD、表情一致率、Emo-Div距GT和ΔResp均退化，当前这批结果更支持优先保留6000权重。音频有活动GT仅13条，比例变化需结合实际次数看。']
 return dict(key=key,title=title,description=description,rows=rows,comparisons=pairs,sources=sources,analysis=analysis,
  samples=200,duration_s=sum(r['duration_s'] for r in row_refs),seed=p['seed'],generation=p['generation'],
  references_sha256=sha(refs),fps=row_refs[0]['media']['listener']['video']['fps'] if key=='responsenet' else row_refs[0]['fps'],
  fps_values=sorted({r['media']['listener']['video']['fps'] if key=='responsenet' else r['fps'] for r in row_refs}),
  width=512,height=512)


def attach_seamless():
 """Replace prose-only ablation cards with the same source-checked tables."""
 begin='<details id="ablations">';end='  <details><summary>配对比较的95%区间</summary>'
 replacement='''<details id="ablations" open><summary>七组消融：原始数值与增减表（含联合Whisper）</summary>
   <p>以下仅为Seamless固定200条。变化量＝右侧方案B−基线A，Emo-Acc和音频比例用百分点；绿色改善、橙色退化。<a href="../p2_responsenet_random200_20261105/?dataset=responsenet">查看最新ResponseNet随机200条</a></p>
   <div id="ablationTables">正在加载已核验的消融数值…</div>
  </details>
'''
 paths=[ROOT/'scripts/p2_seamless200_review.html',ROOT/'data_previews/listen_benchmark_review/p2_seamless200_20261105/index.html',ROOT/'experiments/p2_seamless200_analysis_20261105/index.html']
 for path in paths:
  text=path.read_text();match=re.search(r'<details id="ablations"(?: open)?>',text)
  if match is None:raise ValueError('Missing ablation section: '+str(path))
  start=match.start();stop=text.index(end,start);text=text[:start]+replacement+text[stop:]
  if 'ablation_tables.js' not in text:
   text=text.replace("<script>\n'use strict';",'<script src="../p2_responsenet_random200_20261105/ablation_tables.js"></script>\n<script>\n\'use strict\';',1)
  # Keep media init and audio entirely intact. Tables fetch a separate frozen report.
  hook="window.P2Ablations.mount(document.getElementById('ablationTables'),'../p2_responsenet_random200_20261105/dashboard.json','seamless');"
  if hook not in text:text=text.replace('init().catch(fail);',hook+'\ninit().catch(fail);')
  if '最新ResponseNet随机200 · 12组数值' not in text:
   text=text.replace('<h1>','<p><a href="../p2_responsenet_random200_20261105/?dataset=responsenet">最新ResponseNet随机200 · 12组数值与消融增减表 →</a></p>\n <h1>',1)
  path.write_text(text)
 # Historical selected examples are preserved and explicitly separated.
 for path in [ROOT/'data_previews/listen_benchmark_review/p2_responsenet200_demo_20261103/index.html',ROOT/'experiments/p2_responsenet200_demo_20261103/index.html']:
  text=path.read_text();nav='<nav id="random200-metrics-nav" style="padding:14px 24px;background:#e8f0fa;color:#173653"><a href="../p2_responsenet_random200_20261105/">最新：ResponseNet分层随机200条 · 12组指标与消融增减表 →</a><br>下方保留历史精选样本，和最新随机200条不是同一清单。</nav>'
  if 'random200-metrics-nav' not in text:path.write_text(text.replace('<body>','<body>\n'+nav,1))


def main():
 assert PAGE.is_symlink() and PAGE.resolve()==RN.resolve()
 doc=dict(schema='p2-two-datasets-fifteenway-ablations-v3',metrics=METRICS,groups=[dict(id=g['id'],title=g['title'],note=g['note']) for g in GROUPS],
  datasets={
   'responsenet':dataset('responsenet',[RN,JOINT,LATEST],'ResponseNet · 分层随机200',
    '从完整586条按GT音频状态、时长、ASR来源分层随机；seed20261105。不参考模型得分，与历史精选200条只重合72条。'),
   'seamless':dataset('seamless',[SEAM,WHISPER,JOINT,LATEST],'Seamless · 固定200',
    '按音频、人脸、来源与行为线索抽样的固定诊断子集；不是全量Seamless的自然分布估计。')})
 (PAGE/'dashboard.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 for src,dst in [('p2_metrics_dashboard.html','index.html'),('p2_ablation_tables.js','ablation_tables.js')]:
  shutil.copy2(ROOT/'scripts'/src,PAGE/dst)
 for key,ds in doc['datasets'].items():
  with (PAGE/(key+'_ablation_deltas.csv')).open('w',encoding='utf-8-sig',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=['dataset','group','comparison','baseline','trial','metric','baseline_value','trial_value','delta_B_minus_A','display_delta','delta_unit','better']);writer.writeheader()
   for pair in ds['comparisons']:
    for metric in pair['metrics']:writer.writerow(dict(dataset=key,group=pair['group'],comparison=pair['label'],baseline=LABELS[pair['baseline']],trial=LABELS[pair['comparison']],metric=metric['key'],baseline_value=metric['baseline'],trial_value=metric['comparison'],delta_B_minus_A=metric['delta'],display_delta=metric['display_delta'],delta_unit=metric['delta_unit'],better=metric['better']))
  with (PAGE/(key+'_all_summary.csv')).open('w',encoding='utf-8-sig',newline='') as f:
   fields=list(dict.fromkeys(['label','step',*[m['key'] for m in METRICS],'emo_div_generated','emo_div_gt']));writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(ds['rows'])
  # Preserve the prior download URL used by the existing video review page.
  shutil.copy2(PAGE/(key+'_all_summary.csv'),PAGE/(key+'_all14_summary.csv'))
 attach_seamless()
 report=dict(success=True,datasets=2,systems_each=len(ORDER),comparisons_each=sum(len(g['pairs']) for g in GROUPS),metrics_each=len(METRICS),
  numeric_pairs=sum(len(ds['comparisons'])*len(METRICS) for ds in doc['datasets'].values()),sources={k:v['sources'] for k,v in doc['datasets'].items()},
  page='http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/')
 (RN/'DASHBOARD_BUILD.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
