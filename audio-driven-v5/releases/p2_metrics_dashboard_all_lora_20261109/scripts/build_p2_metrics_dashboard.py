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
CURRENT_AV=ROOT/'experiments/p2_current500h_av6000_sft2752_dual200_20261107'
CURRENT_FEATURE=ROOT/'experiments/p2_current500h_feature6000_sft2752_dual200_20261108'
CURRENT_WHISPER=ROOT/'experiments/p2_current500h_whisper6000_sft2752_dual200_20261108'
ALL_LORA6000=ROOT/'experiments/p2_ltx_all_lora6000_seamless200_sft2752_eval32_20261109'
ALL_LORA6428=ROOT/'experiments/p2_ltx_all_lora6428_seamless200_sft2752_eval28_20261109'
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
 'ltx_hidden_event_feature_whisper_6000_sft2752':'主干Event＋Feature＋Whisper · 6000＋最新SFT2752',
 'current500h_av_only_6000_sft2752':'当前500h AV-only · 6000＋最新SFT2752',
 'current500h_av_feature_6000_sft2752':'当前500h AV＋Feature · 6000＋最新SFT2752',
 'current500h_av_whisper_6000_sft2752':'当前500h AV＋Whisper · 6000＋最新SFT2752',
 'ltx_all_lora_event_feature_whisper_6000_sft2752':'全attention／FFN LoRA＋Event＋Feature＋Whisper · 6000＋最新SFT2752',
 'ltx_all_lora_event_feature_whisper_6428_sft2752':'全attention／FFN LoRA＋Event＋Feature＋Whisper · 6428＋最新SFT2752'}
ORDER=['av_only_6000','av_event_6000','base_feature_6000','base_event_feature_7000',
 'sft_on_base_event_6000','sft_event_6000','whisper_av_6000',
 'av_only_8927','av_event_8927','base_feature_8927','sft_event_8927','whisper_av_8000',
 'base_event_feature_whisper_6000','base_event_feature_whisper_8000',
 'current500h_av_only_6000_sft2752','current500h_av_feature_6000_sft2752',
 'current500h_av_whisper_6000_sft2752','ltx_hidden_event_feature_whisper_6000_sft2752',
 'ltx_all_lora_event_feature_whisper_6000_sft2752','ltx_all_lora_event_feature_whisper_6428_sft2752']
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
   ('latest_event6000','最新主干＋SFT2752 vs AV＋Event6000（多项变化）','av_event_6000','ltx_hidden_event_feature_whisper_6000_sft2752'),
   ('current_av_joint6000','相同SFT2752条件：当前仅AV→联合6000','current500h_av_only_6000_sft2752','ltx_hidden_event_feature_whisper_6000_sft2752')])]
GROUPS[-1]['note']+=' 新增最新主干Event＋Feature＋Whisper6000＋SFT2752，与历史联合6000、AV6000及Event6000对照。新组同时改变训练数据、Event读出位置及规划器，属于pipeline比较，不能归因于某一个组件或SFT。'
GROUPS[-1]['note']+=' 最新补充当前500h AV-only6000＋SFT2752基线，直接与主干联合6000共享同一400条规划文本、H、Gemma和采样参数；两者同时相差Event、Feature与Whisper三个组件，只能评价组合pipeline的整体收益。该基线与历史AV6000独立列出，不覆盖历史数据。'


# Current500h experiments share the exact SFT2752 conditions. Retain historical
# arms and comparisons independently; their training data/planners differ.
GROUPS[1]['pairs'].append(('current_feature6000','当前500h／SFT2752：仅AV→加Feature','current500h_av_only_6000_sft2752','current500h_av_feature_6000_sft2752'))
GROUPS[1]['note']+=' 新增当前500h／SFT2752同条件对照：仅AV→AV＋Feature6000；与上方历史原生Qwen实验分开解读。'
GROUPS[5]['pairs'].extend([
 ('current_whisper6000','当前500h／SFT2752：仅AV→加Whisper','current500h_av_only_6000_sft2752','current500h_av_whisper_6000_sft2752'),
 ('current_feature_whisper','当前500h／SFT2752：Feature方案→Whisper方案','current500h_av_feature_6000_sft2752','current500h_av_whisper_6000_sft2752')])
GROUPS[5]['note']+=' 新增当前500h／SFT2752同条件对照。Feature→Whisper一行同时移除Feature监督和增加Whisper条件，用于方案选择，不能归因为单组件变化。'
GROUPS[-1]['pairs'].extend([
 ('current_feature_joint6000','相同SFT2752条件：当前Feature→联合6000','current500h_av_feature_6000_sft2752','ltx_hidden_event_feature_whisper_6000_sft2752'),
 ('current_whisper_joint6000','相同SFT2752条件：当前Whisper→联合6000','current500h_av_whisper_6000_sft2752','ltx_hidden_event_feature_whisper_6000_sft2752')])
GROUPS[-1]['note']+=' 本轮补入当前Feature与Whisper单组件6000步；同一双200、同SFT2752文本／H／Gemma／seed／采样。Feature→联合同时增加Event与Whisper，Whisper→联合同时增加Event与Feature；联合训练日程及Event读出按原协议解读。'

GROUPS.append(dict(id='all_lora',title='8 · 扩大LTX attention／FFN LoRA范围',
 note='本组仅完成Seamless固定200；ResponseNet未评测。全LoRA指48层原生attention及FFN共1632处rank8 LoRA，双路speaker Whisper仍全参数训练。主对照为同6000步的原范围完整pipeline→扩大LoRA范围；两者均为AV＋Event＋Feature＋Whisper、相同SFT2752／seed137／采样，但训练时长分组窗口512→496、训练日程不同，不能严格单变量归因。6000→6428只比较继续训练；其余基线对照同时改变多个组件，历史Event还改变规划器。',pairs=[
  ('all_lora_scope6000','同6000步：原完整pipeline→全LoRA','ltx_hidden_event_feature_whisper_6000_sft2752','ltx_all_lora_event_feature_whisper_6000_sft2752'),
  ('all_lora_scope6428','原完整6000→全LoRA6428（步数不同）','ltx_hidden_event_feature_whisper_6000_sft2752','ltx_all_lora_event_feature_whisper_6428_sft2752'),
  ('all_lora_progress','全LoRA继续训练6000→6428','ltx_all_lora_event_feature_whisper_6000_sft2752','ltx_all_lora_event_feature_whisper_6428_sft2752'),
  ('all_lora_av','当前仅AV6000→全LoRA完整6000（多项变化）','current500h_av_only_6000_sft2752','ltx_all_lora_event_feature_whisper_6000_sft2752'),
  ('all_lora_event','历史AV＋Event6000→全LoRA完整6000（多项变化）','av_event_6000','ltx_all_lora_event_feature_whisper_6000_sft2752'),
  ('all_lora_feature','当前Feature6000→全LoRA完整6000（多项变化）','current500h_av_feature_6000_sft2752','ltx_all_lora_event_feature_whisper_6000_sft2752'),
  ('all_lora_whisper','当前Whisper6000→全LoRA完整6000（多项变化）','current500h_av_whisper_6000_sft2752','ltx_all_lora_event_feature_whisper_6000_sft2752')]))


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dataset(key,runs,title,description):
 results={};arms={};sources={};audio_files={}
 for run in runs:
  p=read(run/'protocol.json');r=read(run/'RESULTS.json')
  anchor=read(runs[0]/'protocol.json')
  assert p['seed']==anchor['seed'] and p['generation']==anchor['generation']
  assert p['datasets'][key]['references_sha256']==anchor['datasets'][key]['references_sha256']
  if (run/'COMPLETE.json').exists():
   assert read(run/'COMPLETE.json')['status'] in ['complete','partial']
  else:
   assert read(run/'FINAL_AUDIT.json')['success'] and read(run/'HANDOFF_COMPLETE.json')['success']
   assert p['total_outputs']==200 and list(p['datasets'])==[key]
  sources[str(run/'RESULTS.json')]=sha(run/'RESULTS.json')
  for arm in p['arms']:
   a=arm['name'];results[a]=r[a+'/'+key];arms[a]=arm;audio_files[a]=run/'metrics'/a/key/'audio_pairs.jsonl'
 p=read(runs[0]/'protocol.json');refs=Path(p['datasets'][key]['prepared'])/'references.jsonl'
 assert sha(refs)==p['datasets'][key]['references_sha256']
 row_refs=[json.loads(x) for x in refs.read_text().splitlines()];ids={x['id'] for x in row_refs};assert len(ids)==len(row_refs)==200
 rows=[]
 for a in ORDER:
  if a not in results:
   assert key=='responsenet' and a.startswith('ltx_all_lora_')
   continue
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
   if left not in lookup or right not in lookup:continue
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
   '新增当前500h AV-only6000＋SFT2752基线：相同SFT2752条件下，联合6000相对当前仅AV，PSNR 15.4124→16.1165、SSIM 0.65972→0.67302、LPIPS 0.38670→0.34924、FID 15.5568→14.9525、FVD 221.7308→172.5138、Emo-Acc 42.600%→43.768%、ΔResp 0.74382→0.54294改善；表中Emo-Div距GT也改善。FVD下降约22.2%，ΔResp下降约27.0%。',
   '当前仅AV和最新联合6000的声学整段漏回应均为99/134（73.88%）；声音回应仍是短板。这个对照共享同一400条SFT2752条件，但同时增加Event、Feature和Whisper，不能分别归因于某一组件。下列分析另与历史实验比较：',
   '最新主干Event＋Feature＋Whisper6000＋SFT2752相对历史联合6000：FID 16.5329→14.9525、FVD 179.2737→172.5138、Emo-Acc 38.353%→43.768%改善；PSNR 16.3009→16.1165、SSIM 0.67624→0.67302、LPIPS 0.34090→0.34924退步，ΔResp基本相当，未体现全面收益。',
   '最新组音频仍有明显漏回应：134条声学有活动GT中99条没有重叠的生成活动，漏回应率73.88%。图像及分布指标200/200，表情199/200、运动198/200；这是检测支持不足，不是生成失败。',
   '最新组与历史组同时存在训练数据、Event读出位置及规划器差异；下面新增差值仅描述整条pipeline的变化，不能视为最新SFT的单变量收益。以下其余分析指历史联合6000／8000：',
   '联合6000相对仅AV6000：PSNR 15.8926→16.3009、SSIM 0.66620→0.67624、LPIPS 0.36186→0.34090；FID略变差，FVD仅小幅下降，ΔResp基本持平，不能说所有指标都有收益。',
   '联合6000相对仅AV＋Whisper6000：PSNR、SSIM、LPIPS、FID、FVD和ΔResp均未进一步改善；加入Event＋Feature没有带来一致叠加收益。',
   '联合6000→8000：FID 16.5329→16.0049、FVD 179.2737→174.6751、ΔResp 0.54748→0.49722改善；像素／感知相似度与表情一致率下降，Emo-Div距GT增大，存在取舍。',
   '有声GT整段漏声率6000约89.55%、8000约93.28%，音频回应漏检仍明显。情绪／运动各199/200，图像与分布指标均200/200。']
 else:
  analysis=[
   '新增当前500h AV-only6000＋SFT2752基线：相同SFT2752条件下，联合6000相对当前仅AV，FVD 221.4636→184.5401（下降16.7%）、ΔResp 0.37779→0.25639（下降32.1%）、FID 14.1459→13.2560、Emo-Acc 56.494%→59.050%改善；LPIPS和Emo-Div距GT也改善。PSNR 16.8743→16.7695、SSIM 0.65693→0.65380略退步。',
   '当前仅AV与最新联合6000的声学整段漏回应均为11/13；静音GT误发声分别为31/187和33/187，联合没有改善全部音频指标。两者共享同一400条SFT2752条件；Event、Feature和Whisper同时变化，不能分开归因。下列分析另与历史实验比较：',
   '最新主干Event＋Feature＋Whisper6000＋SFT2752相对历史联合6000：PSNR 17.5129→16.7695、SSIM 0.66710→0.65380、LPIPS 0.30379→0.33664、FID 12.6472→13.2560、FVD 150.7708→184.5401、Emo-Acc 61.489%→59.050%、ΔResp 0.22306→0.25639均退步；不能据此认定最新组更好。',
   '最新组13条声学有活动GT中11条没有重叠的生成活动；187条无活动GT中33条误发声。所有200条图像、表情和运动评分完成。最新与历史同时改变数据、Event读出位置及规划器，不是单变量消融。以下其余分析指历史联合6000／8000：',
   '联合6000相对仅AV6000：FVD 161.1684→150.7708、ΔResp 0.23126→0.22306改善；PSNR、SSIM、LPIPS、FID与表情一致率退化，不能归纳成全面收益。',
   '联合6000相对AV＋Event6000：FVD小幅改善，但像素、感知、FID、表情和ΔResp未改善；相对原Event＋Feature7000，七项主要指标里除表情一致率外均更差。',
   '联合6000相对仅AV＋Whisper6000：ΔResp与Emo-Div距GT改善，PSNR、SSIM、LPIPS、FID、FVD与表情一致率退化；联合8000也未稳定优于对应Whisper8000。',
   '联合6000→8000：PSNR、SSIM、LPIPS、FID、FVD、表情一致率、Emo-Div距GT和ΔResp均退化，当前这批结果更支持优先保留6000权重。音频有活动GT仅13条，比例变化需结合实际次数看。']
 # Describe the newly completed arms from exact source values.
 av=lookup['current500h_av_only_6000_sft2752'];feature=lookup['current500h_av_feature_6000_sft2752'];whisper=lookup['current500h_av_whisper_6000_sft2752']
 def transition(a,b):
  return '、'.join(f"{m['label']} {a[m['key']]*(100 if m.get('percent') else 1):.{m['digits']}f}→{b[m['key']]*(100 if m.get('percent') else 1):.{m['digits']}f}{m['unit']}" for m in METRICS[:8])
 fresh=[
  '本轮新增当前500h／SFT2752的AV＋Feature6000与AV＋Whisper6000，均完整生成和评分200条，与当前AV-only6000共享同一批规划文本、H、Gemma与采样条件。历史同名实验保留，不能混作同一次训练。',
  '只加Feature相对当前仅AV：'+transition(av,feature)+'。',
  '只加Whisper相对当前仅AV：'+transition(av,whisper)+'。',
  f"音频整段漏回应：当前AV为{av['missed_clips']}/{av['gt_active_clips']}，Feature为{feature['missed_clips']}/{feature['gt_active_clips']}，Whisper为{whisper['missed_clips']}/{whisper['gt_active_clips']}。音频仍需结合有效GT数量与活动F1解读。新增差值表还列出Feature→Whisper方案选择，以及两组分别→联合6000的比较。"]
 analysis=fresh+analysis
 if key=='seamless':
  base=lookup['ltx_hidden_event_feature_whisper_6000_sft2752'];full=lookup['ltx_all_lora_event_feature_whisper_6000_sft2752'];latest=lookup['ltx_all_lora_event_feature_whisper_6428_sft2752']
  analysis=[
   '新增全attention／FFN LoRA完整组6000和6428，两组均完成同Seamless200生成、解码、视频及音频评分；SFT2752规划文本／H／Gemma／speaker Whisper／seed137与原完整6000相同。5000已按用户要求停止，未作为完整评测加入。',
   '同6000步扩大LoRA范围：'+transition(base,full)+f"。FVD下降{(1-full['fvd']/base['fvd'])*100:.2f}%，FID下降{(1-full['fid']/base['fid'])*100:.2f}%，LPIPS下降{(1-full['lpips']/base['lpips'])*100:.2f}%；视觉质量和分布指标改善，表情一致率略降。",
   '全LoRA6000→6428：'+transition(full,latest)+'。6428像素／感知相似度略好，FID、FVD和ΔResp回退，不能认定继续训练全面改善；目前更倾向将6000作为综合候选，仍需结合视频复核。',
   f"全LoRA音频整段漏回应：6000为{full['missed_clips']}/{full['gt_active_clips']}，6428为{latest['missed_clips']}/{latest['gt_active_clips']}，原完整6000为{base['missed_clips']}/{base['gt_active_clips']}；按当前声学活动阈值衡量，发声回应改善有限，不能认为音频问题已解决。",
   '这两组只在Seamless评测；不要外推为ResponseNet收益。原范围与全LoRA训练时长分组窗口分别512和496，训练日程不同；仅一个生成seed，无显著性结论。']+analysis
 else:
  analysis=['全attention／FFN LoRA6000与6428目前仅完成Seamless固定200，尚未进行本ResponseNet子集评测，因此本页不填入其分数。']+analysis
 return dict(key=key,title=title,description=description,rows=rows,comparisons=pairs,sources=sources,analysis=analysis,
  samples=200,duration_s=sum(r['duration_s'] for r in row_refs),seed=p['seed'],generation=p['generation'],
  references_sha256=sha(refs),fps=row_refs[0]['media']['listener']['video']['fps'] if key=='responsenet' else row_refs[0]['fps'],
  fps_values=sorted({r['media']['listener']['video']['fps'] if key=='responsenet' else r['fps'] for r in row_refs}),
  width=512,height=512)


def attach_seamless():
 """Replace prose-only ablation cards with the same source-checked tables."""
 begin='<details id="ablations">';end='  <details><summary>配对比较的95%区间</summary>'
 replacement='''<details id="ablations" open><summary>八组消融：原始数值与增减表（含全LoRA）</summary>
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
  # The media selector remains the historical fourteen systems; the summary
  # and ablation tables show every completed Seamless run from one source.
  text=text.replace("D=await r.json();window.reviewData=D;buildOverview();", "D=await r.json();const summaryResponse=await fetch('../p2_responsenet_random200_20261105/dashboard.json',{cache:'no-cache'});if(!summaryResponse.ok)throw Error('总体指标HTTP '+summaryResponse.status);const summaryDoc=await summaryResponse.json();D.summary=summaryDoc.datasets.seamless.rows;window.reviewData=D;buildOverview();")
  text=text.replace("'展开200条总体指标与解释（'+D.models.length+'组）'", "'展开200条总体指标与解释（'+D.summary.length+'组）'")
  if '最新ResponseNet随机200 · 12组数值' not in text:
   text=text.replace('<h1>','<p><a href="../p2_responsenet_random200_20261105/?dataset=responsenet">最新ResponseNet随机200 · 12组数值与消融增减表 →</a></p>\n <h1>',1)
  path.write_text(text)
 # Historical selected examples are preserved and explicitly separated.
 for path in [ROOT/'data_previews/listen_benchmark_review/p2_responsenet200_demo_20261103/index.html',ROOT/'experiments/p2_responsenet200_demo_20261103/index.html']:
  text=path.read_text();nav='<nav id="random200-metrics-nav" style="padding:14px 24px;background:#e8f0fa;color:#173653"><a href="../p2_responsenet_random200_20261105/">最新：ResponseNet分层随机200条 · 12组指标与消融增减表 →</a><br>下方保留历史精选样本，和最新随机200条不是同一清单。</nav>'
  if 'random200-metrics-nav' not in text:path.write_text(text.replace('<body>','<body>\n'+nav,1))


def main():
 assert PAGE.is_symlink() and PAGE.resolve()==RN.resolve()
 for run in [CURRENT_AV,CURRENT_FEATURE,CURRENT_WHISPER]:
  audit=read(run/'FINAL_AUDIT.json');assert audit['success'] and audit['outputs']==400
  assert read(run/'protocol.json')['planner_adapter_identity']==read(CURRENT_AV/'protocol.json')['planner_adapter_identity']
 for run in [ALL_LORA6000,ALL_LORA6428]:
  audit=read(run/'FINAL_AUDIT.json');p=read(run/'protocol.json')
  assert audit['success'] and audit['outputs']==200 and audit['step']==p['arms'][0]['step']
  assert read(run/'HANDOFF_COMPLETE.json')['success']
  assert p['planner_adapter_identity']==read(LATEST/'protocol.json')['planner_adapter_identity']
 doc=dict(schema='p2-two-datasets-completed-ablations-v6',metrics=METRICS,groups=[dict(id=g['id'],title=g['title'],note=g['note']) for g in GROUPS],
  datasets={
   'responsenet':dataset('responsenet',[RN,JOINT,LATEST,CURRENT_AV,CURRENT_FEATURE,CURRENT_WHISPER],'ResponseNet · 分层随机200',
    '从完整586条按GT音频状态、时长、ASR来源分层随机；seed20261105。不参考模型得分，与历史精选200条只重合72条。'),
   'seamless':dataset('seamless',[SEAM,WHISPER,JOINT,LATEST,CURRENT_AV,CURRENT_FEATURE,CURRENT_WHISPER,ALL_LORA6000,ALL_LORA6428],'Seamless · 固定200',
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
 report=dict(success=True,datasets=2,systems_by_dataset={k:len(v['rows']) for k,v in doc['datasets'].items()},
  comparisons_by_dataset={k:len(v['comparisons']) for k,v in doc['datasets'].items()},metrics_each=len(METRICS),
  numeric_pairs=sum(len(ds['comparisons'])*len(METRICS) for ds in doc['datasets'].values()),sources={k:v['sources'] for k,v in doc['datasets'].items()},
  page='http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/')
 (RN/'DASHBOARD_BUILD.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
