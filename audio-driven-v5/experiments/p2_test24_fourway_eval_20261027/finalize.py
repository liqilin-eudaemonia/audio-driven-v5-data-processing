import sys,os,time,subprocess,json,statistics
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,atomic_json
PY=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python';arms=['E0','E1','E0F','E1F']
while not (OUT/'glare_common_scope.json').exists():time.sleep(10)
if read(OUT/'glare_common_scope.json')['sample_ids']:
 children=[]
 for arm in arms:
  with (OUT/'logs'/f'score-{arm}.log').open('a') as log:children.append(subprocess.Popen([str(PY),str(OUT/'score_glare_common.py'),arm,'cpu'],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}))
 for p in children:assert p.wait()==0,'GLARE scoring failed; inspect logs'
while not (OUT/'scoring_finished.json').exists():time.sleep(10)
comp=read(OUT/'comparison.json')['metrics'];values={a:{} for a in arms}
for a in arms:
 for m in ['fvd','utmos','sync']:
  if not comp[m]['samples_per_seed']:continue
  rows=[comp[m]['values'][a][str(s)][m] for s in [137,251]]
  if m=='fvd':values[a]['FVD-32']=statistics.mean(r['supported_subset_value'] for r in rows)
  elif m=='utmos':values[a]['UTMOSv2']=statistics.mean(r['generated']['supported_subset_value'] for r in rows)
  else:
   for k,n in [('lse_d','LSE-D'),('lse_c','LSE-C')]:values[a][n]=statistics.mean(r['generated']['supported_subset_value'][k] for r in rows)
 if comp.get('glare',{}).get('models'):values[a].update(comp['glare']['models'][a]['mean'])
atomic_json(OUT/'table_values.json',values)
finished={}
for a in arms:
 shards=[read(OUT/a/'generation'/f'generation_shard_{i:03d}.json') for i in range(24)];rows=[r for x in shards for r in x['results']];assert len(rows)==48 and all(r['status']=='success' for r in rows)
 omni=read(OUT/a/'omni/results.json');finished[a]=dict(generated=48,omni_status=omni['status'],glare=read(OUT/a/'glare/results.json')['status'])
names={'E0':'AV','E1':'AV＋Event','E0F':'AV＋Feature','E1F':'AV＋Event＋Feature'}
def fmt(v):return 'N/A' if v is None else f'{v:.4f}'
scope=read(OUT/'glare_common_scope.json');sampling=read(OUT/'sampling.json')
lines=['# 独立 test24：四组最终权重评测','',f'24条、{sampling["seconds"]:.3f}秒、{sampling["groups"]}个连通组。固定生成seed=137/251，共192条。四组均使用第1268步最终权重；训练seed20261023、global batch24、同一训练集。Event系数0.05；Feature为视频/音频各0.1。','',
'本次已核验与四组实际旧训练集的样本、会话、参与者及连通组无交集。该集合按七类覆盖及大人脸挑选，属于事件丰富的独立测试子集，不代表完整test自然分布。原有七类标签仍为弱标签，没有升级成人工真值。','',
'生成仅读取speaker音频和listener首帧。参考音频按数据集既定recipe执行官方词/句/VAD union及多汉字ASR静音，独立物化float WAV并保留逐文件哈希；原数据未改。HTML页面按用户要求暂未修改。','',
'## 音视频质量','', '| 配置 | FVD-32 ↓ | UTMOSv2 ↑ | LSE-D ↓ | LSE-C ↑ |','|---|---:|---:|---:|---:|']
for a in arms:lines.append('| '+names[a]+' | '+' | '.join(fmt(values[a].get(k)) for k in ['FVD-32','UTMOSv2','LSE-D','LSE-C'])+' |')
lines+=['','每项使用四组、两个seed共同有效的片段，各seed计算后等权平均。FVD是每seed共同样本池的分布距离，采用连续32帧窗口；非逐条FVD平均。支持范围：'+ '；'.join(f'{m} {comp[m]["samples_per_seed"]}/24条，{comp[m]["common_groups"]}组' for m in ['fvd','utmos','sync'])+'。','',
'## GLARE公开反应指标','',f'四组两个seed共同成功检测{scope["samples_per_seed"]}/24条。每帧要求MediaPipe与FER均有效；失败保留在完整集合分母，完整集合不能算分时记N/A。下表为事后共同成功子集。','',
'| 配置 | R-F1 ↑ | R-tIoU ↑ | R-ATD ↓ | R-FID ↓ |','|---|---:|---:|---:|---:|']
for a in arms:lines.append('| '+names[a]+' | '+' | '.join(fmt(values[a].get(k)) for k in ['R-F1','R-tIoU','R-ATD','R-FID'])+' |')
lines+=['','完整集合状态与检测覆盖：`glare_common_scope.json`；每seed分数、匹配数及检测失败原因保留在各组目录。','',
'## 对照实验计划的解释边界','',
'这次补上了独立test对照，但仍仅一个训练seed、24条事件丰富片段。公开GLARE使用六类视觉反应、tIoU≥0.5贪心匹配，缺席类F1计0；不同于实验计划预定K_primary、tIoU≥0.3最大总IoU匹配协议，不能套用计划主指标0.03门槛。自动检测不是人工真值。','',
'FVD、UTMOS、SyncNet不能独立证明更会回应speaker。七类音频回应时机、人工确认的无事件区间误触发、speaker替换对照及人工盲评仍待完成。配对组级bootstrap区间见comparison.json，只反映本子集重采样，不包含训练seed及检测器误差。','',
'## 复核入口','',
'- [机器可读比较](comparison.json)、[指标表](table_values.json)、[隔离检查](isolation_audit.json)。',
'- [样本范围](sampling.json)、[四组权重](experiments.json)、[音频参考来源](audio_reference_provenance.json)。',
'- E0/E1/E0F/E1F目录分别保存生成视频、listener独立音轨、每条指标及检测结果。']
(OUT/'report.md').write_text('\n'.join(lines)+'\n');(OUT/'README.md').write_text('# 独立test24四组评测\n\n[结果与解释](report.md)；[机器可读比较](comparison.json)。\n\n24条×4组×2生成seed，192输出。HTML暂未修改。\n')
atomic_json(OUT/'COMPLETION.json',dict(status='completed',arms=finished,glare_common_samples=scope['samples_per_seed'],html_changed=False,time=time.time()))
print(json.dumps(values,indent=2),flush=True)
