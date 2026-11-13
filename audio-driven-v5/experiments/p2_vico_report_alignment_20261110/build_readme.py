"""Build comparison tables from frozen audit outputs without changing scores."""
from pathlib import Path
import csv, hashlib, json
P=Path(__file__).resolve().parent
ROOT=P.parents[1]
read=lambda name:json.loads((P/name).read_text())
# Existing measured outputs must remain intact.
sealed=read('COMPLETE.json')
for name,expected in sealed['files'].items():
    assert hashlib.sha256((P/name).read_bytes()).hexdigest()==expected,name
results=read('RESULTS.json')['methods'];lf=read('LISTENFORMER_COMPARISON.json');dt=read('DUALTALK_PROXY.json')
keys=['FD','P_FD','MSE','SID','Var','rPCC'];groups=['exp','pose']
labels={'FD':'FD ↓','P_FD':'P-FD ↓','MSE':'MSE ↓','SID':'SID','Var':'Var','rPCC':'rPCC ↓'}
# Transcribed/verified in the previous audit, with local PDF and screenshot retained.
dy_report={'FD':{'exp':.074,'pose':3.192},'MSE':{'exp':.018,'pose':1.636},'SID':{'exp':4.586,'pose':3.070},'Var':{'exp':.275,'pose':.596}}
l2l_report={}
with (ROOT/'experiments/p2_vico_literature_protocol_20261108/literature_rows.csv').open() as f:
    for row in csv.DictReader(f):
        if row['method']=='L2L' and row['result_source']=='DIM Table1' and row['reported_value']:
            l2l_report.setdefault(row['metric'],{})[row['group']]=float(row['reported_value'])
dt_report={}
for row in read('PAPER_ROWS.json'):
    if row['method']=='DualTalk (paper Table4)':dt_report.setdefault(row['metric'],{})[row['group']]=row['reported']
def fmt(x):return '—' if x is None else f'{x:.6f}'
def pair(m,k):return ' / '.join(fmt(m.get(k,{}).get(g)) for g in groups)
lines=['# ViCo report 与当前 eval config：数值对照及使用说明','',
'**结论：部分指标接近，但尚未证实整套 report 基本对齐。** ListenFormer 本地重训的 PSNR 与论文两位小数相同，FID 高25.79%；其他方法存在数据集、权重或系数空间差异，不能用裸数值判断复现成功、失败或方法优劣。','',
'本目录记录2026-11-10完成的核查。范围来自飞书评测文档；Avatar Forcing 按要求排除。所有本地输出均为 ViCo 官方 test43。本轮复用既有预测，重新计算指定指标，未重新训练或生成视频。','',
'## 1. 如何读这些表','',
'- `Report` 是论文报告值；`本地`是实际生成结果的评分，不是我们的模型结果。',
'- `Exp / Pose` 固定为表情／姿态顺序；`—`表示所引用表没有报告该项，不能当成0。',
'- ListenFormer允许作描述性差值比较，但其本地重训权重和训练划分与作者不同，不是严格复现。',
'- L2L、DualTalk与DyStream的报告和本地结果存在表示或数据集差异，因此只并列展示，不计算相对差、不据此排名。',
'- FD在第2节是BFM系数L1距离；后续FD是分布Fréchet距离，二者不是同一指标。SID/Var只列统计量，不能普遍理解为越大越好。','',
'## 2. ListenFormer：本地完整重训与论文','',
'本地完成200轮／173800步；351条训练、46条验证，验证集选择epoch1／869步。作者官方预训练分享已取消，本地找到的是本地重训best.pt和last.pt。','',
'### 2.1 画质与身份（论文Table2）','',
'| 指标 | Report | 本地test43 | 本地−Report | 相对差 |','|---|---:|---:|---:|---:|']
allrows=[]
for x in lf['rows'][:5]:
    name={'psnr':'PSNR ↑','ssim':'SSIM ↑','fid':'图像FID ↓','csim':'CSIM ↑','cpbd':'CPBD ↑'}[x['metric']]
    lines.append(f"| {name} | {x['reported']:.2f} | {x['local']:.6f} | {x['difference']:+.6f} | {x['relative_difference_percent']:+.2f}% |")
lines+=['','PSNR/SSIM本轮使用原官方脚本重新解码43条／16,288帧计算，与归档值完全相同。FID/CSIM/CPBD引用已封存实测；本轮核验结果文件及43条视频哈希，未重跑这三项。评分256×256/30fps；PSNR/SSIM/CSIM/CPBD按帧加权，图像FID按全集计算。','',
'### 2.2 原生动作（论文Table1，test Average）','',
'| BFM L1 ×100 ↓ | Report | 本地逐片段等权 | 本地−Report | 相对差 |','|---|---:|---:|---:|---:|']
for x in lf['rows'][5:]:
    lines.append(f"| {x['metric'].replace('BFM_L1x100_','')} | {x['reported']:.2f} | {x['local']:.6f} | {x['difference']:+.6f} | {x['relative_difference_percent']:+.2f}% |")
lines+=['','本轮从43份原生预测和GT重算；与归档一致。论文精确汇总流程尚未确认，保留该差异。低距离不代表反馈自然；该本地模型的视频DECA表情时序方差约为GT的3.2%，已有动作不足现象。','',
'来源：[论文表格截图](sources/listenformer_table2.png)、[原始差值CSV](listenformer_vs_report.csv)、[官方PSNR/SSIM复算回执](LISTENFORMER_BASE_RECHECK.json)、[权重检索记录](LISTENFORMER_WEIGHT_SEARCH.json)。','',
'## 3. DyStream：报告不是ViCo，禁止直接比较','',
'Report来自DyStream论文Table2的RealTalk 38条、MediaPipe；本地是ViCo 43条、视频反提取EMOCA。两侧数据集和特征表示均不同。','',
'| 指标 | Report Exp | Report Pose | 本地 Exp | 本地 Pose |','|---|---:|---:|---:|---:|']
def append_table(report,local):
    for k in keys:
        lines.append('| '+labels[k]+' | '+' | '.join(fmt(report.get(k,{}).get(g)) for g in groups)+' | '+' | '.join(fmt(local[k][g]) for g in groups)+' |')
append_table(dy_report,results['dystream_common']['metrics'])
lines+=['','本地主表使用固定共同有效帧16,238帧、seed137单次输出、listener自身音轨全零。与旧封存12项指标最大差2.24e-9。[论文Table2截图](sources/dystream_table2.png) · [原论文§4.4](https://arxiv.org/pdf/2512.24408)。','',
'## 4. L2L：ViCo重训Report与四套公开人物权重','',
'Report来自DIM Table1：作者在ViCo上用EMOCA重新训练L2L。本地四套官方人物权重输出DECA动作，训练数据也不同；没有公开写实renderer用于共同RGB反提取。以下仅并列数值，不代表本地优于Report。','',
'每个单元格均为 **Exp / Pose**；本地每套43条、16,245评分帧。','',
'| 指标 | Report / ViCo重训EMOCA | Conan / DECA | Fallon / DECA | Stephen / DECA | Trevor / DECA |','|---|---:|---:|---:|---:|---:|']
for k in keys:
    lines.append('| '+labels[k]+' | '+pair(l2l_report,k)+' | '+' | '.join(pair(results['l2l_'+n+'_native']['metrics'],k) for n in ['conan','fallon','stephen','trevor'])+' |')
lines+=['','四套模型全部列出，未根据GT选择人物。使用当前指标数学定义和时间偏移，但保留DECA原生表示；完整数据见[重算CSV](rescored_metrics.csv)。原文依据：本地`writing/17_DIM_ECCV2024.pdf`第9–11页，明确说明EMOCA重训。','',
'## 5. DualTalk：官方公开权重的迁移诊断','',
'Report来自论文ViCo Table4。本地官方权重与当前HF LFS哈希一致，核心源码与当前官方发布逐字节一致；已有43条self-zero预测，本轮重新评分。','',
'**本地是DECA迁移代理：neck与global head的几何对应关系尚未确认，公开权重是否就是ViCo Table4所用权重也未确认。不能将这一行视为严格同协议复现。**','',
'| 指标 | Report Exp | Report Pose | 本地代理 Exp | 本地代理 Pose |','|---|---:|---:|---:|---:|']
append_table(dt_report,dt['metrics'])
lines+=['','[论文Table4](https://arxiv.org/html/2505.18096v1#S5.T4) · [本地完整结果](DUALTALK_PROXY.json) · [官方发布一致性](DUALTALK_RELEASE_PARITY.json)。论文正文提及SID，但Table4只给FD/P-FD/MSE，因此其余Report栏留空。','',
'## 6. 当前eval config与GT锚点','',
'主配置：官方test43；EMOCA_v2_lr_mse_20，原始pose6+exp50；listener[1:T]对应speaker[0:T−1]；FD/P-FD/MSE逐片段等权；SID在GT上拟合KMeans40/20；Var采用所有评分帧按系数求方差再平均、ddof=0；rPCC采用拉平相关系数之差的绝对值。L2L/DualTalk只复用数学定义，不改名为EMOCA结果。','',
'| GT统计 | DIM Report | 当前共同帧集 |','|---|---:|---:|']
for k,g,v in [('Var','exp',.93),('Var','pose',.01),('SID','exp',5.03),('SID','pose',4.07)]:
    actual=results['dystream_common']['metrics'][k+'_GT'][g];lines.append(f'| {k} {g} | {v:.2f} | {actual:.6f} |')
lines+=['','共同16,238帧改为DyStream自身有效16,245帧时，FD Exp仅31.715231→31.705459，但SID Exp为4.396098→4.205501。由于GT聚类随帧集重新拟合，比较时必须固定帧掩码和聚类输入。GT统计接近、函数实现一致都不足以证明完整论文协议一致。','',
'## 7. 本轮做了什么、没完成什么','',
'| 项目 | 本轮执行 |','|---|---|',
'| DyStream | 129份GT/speaker/pred特征哈希核验，12项指标重算，DIM原函数16个原始统计一致性核验 |',
'| L2L | 172条公开人物预测核验，四套模型按当前公式重算，保持DECA单独标注 |',
'| DualTalk | 官方checkpoint及当前代码核验，43条预测重评分，保留迁移代理限制 |',
'| ListenFormer | 本地重训产物和43条视频核验；原生BFM L1及官方PSNR/SSIM重新计算 |',
'| 新训练／新生成 | 无 |',
'| Avatar Forcing | 排除 |','',
'没有取得ListenFormer作者可用预训练副本；DIM/RLHG等仍缺本轮可用核心生成权重。资源检查详情及其他方法状态见[完整调查报告](REPORT.md)。没有设置一个事后挑选的“基本对齐”百分比阈值，也没有用test挑权重／人物或调指标尺度。','',
'## 8. 文件导航','',
'| 文件 | 内容 |','|---|---|',
'| [README.md](README.md) | 本说明和全部报告／实测数值对照 |',
'| [report_vs_local_all.csv](report_vs_local_all.csv) | 所有方法逐指标并列表；附可比性和数据来源 |',
'| [REPORT.md](REPORT.md) | 完整核查过程、结论与资源缺项 |',
'| [RESULTS.json](RESULTS.json) / [rescored_metrics.csv](rescored_metrics.csv) | DyStream、L2L详细结果 |',
'| [LISTENFORMER_COMPARISON.json](LISTENFORMER_COMPARISON.json) | ListenFormer逐项差值及逐片段BFM L1 |',
'| [DUALTALK_PROXY.json](DUALTALK_PROXY.json) | DualTalk原生迁移代理的完整指标 |',
'| [AUDIT.json](AUDIT.json) / [ADDITIONAL_AUDIT.json](ADDITIONAL_AUDIT.json) | 输入哈希、版本和数值验证 |',
'| [LISTENFORMER_BASE_RECHECK.json](LISTENFORMER_BASE_RECHECK.json) | 本轮官方PSNR/SSIM复算回执 |',
'| [COMPLETE.json](COMPLETE.json) | 原实验产物哈希清单，不因新增README改写 |',
'| [README_DELIVERY.json](README_DELIVERY.json) | 本次新增文档、CSV及生成脚本的哈希 |','',
'## 9. 复跑','',
'在项目根目录执行，沿用原环境，CPU运行；命令会重写本实验目录的结果文件，原归档预测只读。重跑后若浮点结果或文件内容变化，需重新检查产物哈希，不能继续将旧COMPLETE.json当成新结果封存证明。','',
'```bash','cd /private/liqi_workspace/project_code/audio-driven-v5','',
'env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \\\n  compare/.envs/vico_more/bin/python experiments/p2_vico_report_alignment_20261110/rescore.py','',
'env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \\\n  compare/.envs/vico_more/bin/python experiments/p2_vico_report_alignment_20261110/rescore_additional.py','',
'env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \\\n  compare/.envs/vico_eval/bin/python evaluation/vico_challenge_baseline/evaluations/compute_base_metrics.py \\\n  --gt_video_folder /public/0_DATA/2_Avatar/ViCo/dataset/by_split/test/videos \\\n  --pd_video_folder experiments/p2_vico_additional_baselines_20261107/listenformer/scoring/listenformer \\\n  --anno_file /public/0_DATA/2_Avatar/ViCo/dataset/by_split/test/test.csv --task listener','```','']
# Long-form tables: differences only when intentionally reported for ListenFormer.
for x in lf['rows']:
    allrows.append(dict(method='ListenFormer local retraining',metric=x['metric'],group='',reported=x['reported'],local=x['local'],difference=x['difference'],relative_difference_percent=x['relative_difference_percent'],comparability='descriptive_only_different_training_checkpoint',reported_dataset='ViCo',local_dataset='ViCo test43',reported_space='RGB or native BFM L1x100',local_space='RGB or native BFM L1x100',source=x['source'],notes=x['evidence']))
def rows(method,rep,local,status,dataset,reportspace,localspace,source,notes):
    for k in keys:
        for g in groups:
            allrows.append(dict(method=method,metric=k,group=g,reported=rep.get(k,{}).get(g),local=local[k][g],difference=None,relative_difference_percent=None,comparability=status,reported_dataset=dataset,local_dataset='ViCo test43',reported_space=reportspace,local_space=localspace,source=source,notes=notes))
rows('DyStream',dy_report,results['dystream_common']['metrics'],'not_comparable_dataset_and_space','RealTalk38','MediaPipe','EMOCA','DyStream Table2 / section4.4','43 clips; 16238 shared frames; no direct paper replication')
for n in ['conan','fallon','stephen','trevor']:
    rows('L2L '+n,l2l_report,results['l2l_'+n+'_native']['metrics'],'not_comparable_checkpoint_and_space','ViCo','EMOCA','DECA','DIM Table1','Report is ViCo retrained L2L, not original public person-specific checkpoint')
rows('DualTalk transfer proxy',dt_report,dt['metrics'],'not_comparable_unconfirmed_checkpoint_and_geometry','ViCo','ViCo Table4 representation not confirmed from release','DECA transfer proxy','DualTalk Table4','neck/global head mapping and released checkpoint relation to Table4 unconfirmed')
(P/'README.md').write_text('\n'.join(lines))
with (P/'report_vs_local_all.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
manifest={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['README.md','report_vs_local_all.csv','build_readme.py']}
(P/'README_DELIVERY.json').write_text(json.dumps({'source_seal_verified':True,'existing_experiment_files_modified':False,'csv_rows':len(allrows),'files':manifest},ensure_ascii=False,indent=2)+'\n')
print('Created README.md and report_vs_local_all.csv:',len(allrows),'metric rows; original seal verified')
