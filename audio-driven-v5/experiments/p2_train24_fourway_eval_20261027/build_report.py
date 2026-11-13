import sys,json,math,statistics,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text());comp=read(OUT/'comparison.json')['metrics'];arms=['E0','E1','E0F','E1F'];names={'E0':'AV','E1':'AV＋Event','E0F':'AV＋Feature','E1F':'AV＋Event＋Feature'}
values={a:{} for a in arms}
for a in arms:
 for metric in ['fvd','utmos','sync']:
  rows=[comp[metric]['values'][a][str(s)][metric] for s in [137,251]]
  if metric=='fvd':values[a]['FVD-32']=statistics.mean(r['supported_subset_value'] for r in rows)
  elif metric=='utmos':values[a]['UTMOSv2']=statistics.mean(r['generated']['supported_subset_value'] for r in rows)
  else:
   for key,label in [('lse_d','LSE-D'),('lse_c','LSE-C')]:values[a][label]=statistics.mean(r['generated']['supported_subset_value'][key] for r in rows)
 if 'glare' in comp:values[a].update(comp['glare']['models'][a]['mean'])
def f(x):return 'N/A' if x is None else f'{x:.4f}'
lines=['# 训练集随机24条：四组最终权重快速对照','',
'本报告为 **训练集诊断**，不是 dev/test 泛化评测。随机种子20261027，从共同训练集30,415段无放回抽取24段、152.261秒、10个group_id；没有为凑类别挑样本。每组生成seed=137、251，共192条全部成功。四组最终checkpoint均为第1,268步，训练seed=20261023、global batch=24。','',
'四组数据/目标、基座资产、数值配置、LoRA、共享参数初始化一致，见 [配对核验](pairing_audit.json)。Feature指0.1×视频Feature＋0.1×音频Feature；Event系数0.05。E0的事件头仅作detach(U)诊断；四组LoRA均由AV梯度更新。使用训练时原音轨，没有替换为后来全量静音版本。','',
'## 音视频指标：相同支持样本','',
'每项先取四组、两个seed共同有效的片段，再各seed单独计算并等权平均；FVD按每seed的完整共同样本池计算，绝非逐视频FVD平均。UTMOS使用独立listener音轨；SyncNet为参考实现有效轨迹等权。','',
'| 配置 | FVD-32 ↓ | UTMOSv2 ↑ | LSE-D ↓ | LSE-C ↑ |','|---|---:|---:|---:|---:|']
for a in arms:lines.append('| '+names[a]+' | '+' | '.join(f(values[a][k]) for k in ['FVD-32','UTMOSv2','LSE-D','LSE-C'])+' |')
lines+=['', '共同支持：'+ '；'.join(f"{m}：{comp[m]['samples_per_seed']}/24段、{comp[m]['common_groups']}个group" for m in ['fvd','utmos','sync'])+'。',
'FVD-32采用每条固定32帧连续窗口，不代表整个片段的全部动态；24个样本的分布距离仅适于本次诊断。UTMOS/SyncNet不能替代音频回应时机、语义适当性或人工自然度判断。','',
'## GLARE公开四指标：仅共同检测成功的5条','',
'严格检测要求每帧MediaPipe与FER共同有效。GT有12/24条未满足；四组有效配对分别为16/48、16/48、17/48、16/48。完整24条集合的四项分数均为N/A。下表只覆盖四组两个seed共同成功的5条（10个配对输出/组），为事后共同支持子集，不能冒充完整集合成绩。','',
'| 配置 | R-F1 ↑ | R-tIoU ↑ | R-ATD ↓（×100） | R-FID ↓ | 匹配事件总数（两seed） |','|---|---:|---:|---:|---:|---:|']
for a in arms:
 g=read(OUT/a/'glare_common.json');n=sum(s['matched_events'] for s in g['per_seed'].values());lines.append('| '+names[a]+' | '+' | '.join(f(values[a][k]) for k in ['R-F1','R-tIoU','R-ATD','R-FID'])+f' | {n} |')
lines+=['','前三组有一个seed没有匹配事件，R-ATD按既定协议传播为N/A；不能只平均另一个seed。R-tIoU同样只有1–2个匹配事件支撑，不能凭E1＋Feature的较高值宣布时机改善。',
'公开R-F1使用六类、tIoU≥0.5贪心匹配，缺席类计0；与原计划§8.2的预定K_primary、tIoU≥0.3最大总IoU匹配及多seed计数聚合不同，**不能将本表R-F1当成原计划主指标，也不能套用其0.03提升门槛**。','',
'共同子集的GT自动检测有smiling 4、surprised 1、nodding 1、head_shaking 14，laughing/frowning为0。原24条弱标签中摇头为0。两套标注明显不一致，需人工核验；当前不能把任何一套自动结果视为人工真值。','',
'## 配对差值和组级不确定性','',
'差值均为右组减左组；下行指标负值较好。2000次按抽样group_id成组重采样，保留组内片段、模型和两个生成seed。区间仅反映本小训练子集的重采样不确定性，不含训练seed变化、检测误差或泛化不确定性。','',
'| 对照 | ΔFVD-32 | ΔUTMOS [95%区间] | ΔLSE-D [95%区间] | ΔLSE-C [95%区间] | ΔR-F1 [95%区间] |','|---|---:|---|---|---|---|']
for left,right in [('E0','E1'),('E0F','E1F'),('E0','E0F'),('E1','E1F')]:
 key=left+'→'+right;cols=[]
 for metric,field in [('utmos','utmos'),('sync','lse_d'),('sync','lse_c')]:
  d=comp[metric]['paired_group_bootstrap'][key][field];cols.append(f"{d['delta']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
 d=comp['glare']['paired_group_bootstrap'][key];cols.append(f"{d['delta']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
 lines.append('| '+names[left]+' → '+names[right]+f" | {values[right]['FVD-32']-values[left]['FVD-32']:+.4f} | "+' | '.join(cols)+' |')
lines+=['','## 对照原实验计划','',
'| 原计划要验证的内容 | 本次证据 | 当前能否得出结论 |','|---|---|---|',
'| 共同初始化、数据、预算的E0/E1成对实验 | 四组最终权重及192条同输入/seed输出已核验 | 工程对照成立 |',
'| Event使实际生成的反应和时机改善 | 有GLARE独立检测，但共同集仅5条、每组仅1–2次匹配 | 不足以支持或否定H1 |',
'| 七类与音频回应的事件时机 | GLARE只覆盖视觉六类；未取得独立音频事件真值 | 不能声称音频回应改善 |',
'| 无事件区间误触发护栏 | 没有人工确认的负区间 | 未验证；不能用更少/更多动作代替 |',
'| 自然度、角色、身份、音频质量护栏 | FVD/UTMOS/SyncNet及输出可检查；盲看页已生成 | 自动质量诊断可比较，人工自然度护栏尚未验证 |',
'| speaker替换后的响应适当性（§8.4） | 此轮按用户要求跑正常输入24条快速验证，未执行audio_swap/盲评 | 未验证是否真正更理解speaker |',
'| 独立dev/test与第二训练seed复验 | 本轮均为训练样本、一个训练seed | 无泛化或稳定收益结论 |','',
'## 产物与复核','',
'- [完整机器可读比较](comparison.json)、[抽样清单](sampling.json)、[权重与配置](experiments.json)、[历史源码恢复](source_restoration.json)。',
'- [四组输出盲看页](review/index.html)，每个样本/seed重新随机打乱A–D；[映射单独保存](blind_mapping.json)。尚无人类盲评分数。',
'- [固定三个样本中间帧对照](midframe_contact.png)，仅作静帧展示，不代替视频自然度评价。',
'- 四组目录下保留原生成、独立音轨、检测覆盖/失败、每样本AV结果、两个seed分数和原始特征。失败未从完整集合分母删除。',
'- 37项评测入口测试通过；训练诊断必须显式开启且输入必须与该训练集原观察完全一致；正式dev/test隔离检查未放宽。',
'- 较早两组源码从固定备份按原SHA恢复为独立副本，原权重未改。评测资源出现其他vLLM任务占卡后，将8个未启动分片转到1号机空卡；原模型、精度、CUDA设备类型和指标参数一致。']
(OUT/'report.md').write_text('\n'.join(lines)+'\n');(OUT/'table_values.json').write_text(json.dumps(values,indent=2)+'\n')
try:
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,3,figsize=(13,7));colors=['#4575b4','#74add1','#fdae61','#d73027']
 for ax,key,down in zip(axes.flat,['FVD-32','UTMOSv2','LSE-D','LSE-C','R-F1','R-FID'],[True,False,True,False,False,True]):
  vals=[values[a][key] for a in arms];ax.bar(arms,vals,color=colors);ax.set_title(key+(' (lower is better)' if down else ' (higher is better)'));ax.set_ylim(bottom=0)
  for i,v in enumerate(vals):ax.annotate(f'{v:.3f}',(i,v),ha='center',va='bottom',fontsize=9)
  ax.margins(y=.18)
 fig.suptitle('Train24 diagnostic | 2 generation seeds | GLARE uses 5 common clips only');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
except ImportError:pass
print(json.dumps(values,indent=2))
