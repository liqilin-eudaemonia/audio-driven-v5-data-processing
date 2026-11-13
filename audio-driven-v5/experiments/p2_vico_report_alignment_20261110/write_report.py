"""Write a traceable comparison; never call cross-space differences reproductions."""
from pathlib import Path
import json,csv,hashlib
P=Path(__file__).resolve().parent
read=lambda n:json.loads((P/n).read_text())
r=read('RESULTS.json');lf=read('LISTENFORMER_COMPARISON.json');dt=read('DUALTALK_PROXY.json')
lines=['# ViCo：公开可用方法与 report 的对齐核查（2026-11-10）','',
'结论：目前不能确认整体基本对齐。ListenFormer 的本地完整重训版本只有 PSNR 在论文两位小数下匹配；FID 高约25.8%，BFM头角L1高约40.1%。DyStream、L2L、DualTalk 的当前公开权重输出可以评分，但原表的数据集、权重版本或动作表示不同，因此不能把这些数值差异直接解释成 eval config 错误。','',
'本轮复用已生成预测，重新运行 CPU 指标计算；没有重新生成视频、训练模型或评测 Avatar Forcing。首次定位到旧归档中的 DualTalk 官方权重及完整43条结果后，停止重复下载，并验证该权重与当前官方 Hugging Face LFS SHA256 完全一致。','',
'输入依据：`feishu_docs/评测v2/评测v2.md`、`feishu_docs/Writing/Writing.md`；当前主配置为官方test43、EMOCA_v2_lr_mse_20、pose6+exp50、listener[1:T] / speaker[0:T-1]、DIM发布指标函数、逐系数跨帧方差均值（ddof=0）。此任务不改配置，也不根据论文数值调尺度或选种子。','',
'## 1. ListenFormer：能比较的本地重训结果','',
'来源：本地论文PDF第6页Table2；逐项目视核对见 [表格截图](sources/listenformer_table2.png)。本地模型完成200轮／173800步，使用351train+46validation，验证集选择epoch1／869步；不是作者预训练模型。原版官方权重曾在 [issue #1](https://github.com/liushenme/ListenFormer/issues/1) 发布，但当前百度分享列表接口返回 `errno=-21`、分享已取消。当前HF搜索无结果，本地已找到的best.pt/last.pt均有本地训练回执。','',
'| 指标 | report | 本地test43 | 本地−report | 相对差 |','|---|---:|---:|---:|---:|']
for x in lf['rows'][:5]:lines.append(f"| {x['metric']} | {x['reported']:.2f} | {x['local']:.6f} | {x['difference']:+.6f} | {x['relative_difference_percent']:+.2f}% |")
lines+=['','PSNR/SSIM本轮重新解码43条视频并运行原官方CPU脚本；FID/CSIM/CPBD来自原封存实测，本轮验证43条视频及结果文件哈希，未再次跑GPU画质评分。帧数16,288，256×256/30fps，帧加权汇总，FID按整个集合计算。','',
'Table1 的原生BFM动作距离另外比较；不是EMOCA分布FD：','',
'| BFM L1 ×100 | report test Average | 本地逐片段等权 | 相对差 |','|---|---:|---:|---:|']
for x in lf['rows'][5:]:lines.append(f"| {x['metric'].replace('BFM_L1x100_','')} | {x['reported']:.2f} | {x['local']:.6f} | {x['relative_difference_percent']:+.2f}% |")
lines+=['','原生BFM L1已从43份预测及GT重新计算，与封存值一致。论文跨态度／跨帧的精确汇总流程未充分确认，故这部分保留协议差异。训练划分、权重选择与renderer也可能影响结果；现有实验不能把偏差归因于某一个环节。','',
'## 2. 当前指标函数重算：DyStream','',
'43条官方公开权重预测，seed137，一次采样；listener自身音轨置零，speaker音轨进入other分支。采用封存共同帧集16,238帧。','',
'| 指标 | Exp | Pose |','|---|---:|---:|---:|']
for k in ['FD','P_FD','MSE','SID','Var','rPCC']:
 m=r['methods']['dystream_common']['metrics'][k];lines.append(f"| {k} | {m['exp']:.6f} | {m['pose']:.6f} |")
lines+=['','与此前封存12个指标的最大绝对差为2.24e-9。独立调用DIM原版打印函数，16个原始统计完全一致；其中Var另按已选逐系数定义输出，两种Var都保留在JSON。','',
'DyStream论文第6页Table2／§4.4的听者评测实际是RealTalk 38条、MediaPipe系数，FD Exp=0.074、Pose=3.192。它没有可与上述ViCo/EMOCA一行直接对齐的原生ViCo报告。[原论文](https://arxiv.org/pdf/2512.24408) · [本地核对截图](sources/dystream_table2.png)。GLARE表中DyStream的数字也来自RealTalk／Seamless，不能将方法名ViCo误读为数据集列。','',
'## 3. L2L／DualTalk：按同一数学定义的原生动作诊断','',
'这些行不是EMOCA共同表，不能做同空间排名；仅回答公开输出套用目前指标公式得到什么。L2L四组均保留，没有按GT挑选人物。全部43条、16,245个listener[1:T]评分帧。','',
'| 原生动作输出 | FD Exp | FD Pose | MSE Exp | MSE Pose |','|---|---:|---:|---:|---:|']
for name in ['conan','fallon','stephen','trevor']:
 m=r['methods']['l2l_'+name+'_native']['metrics'];lines.append(f"| L2L {name} / DECA | {m['FD']['exp']:.6f} | {m['FD']['pose']:.6f} | {m['MSE']['exp']:.6f} | {m['MSE']['pose']:.6f} |")
m=dt['metrics'];lines.append(f"| DualTalk / DECA迁移代理 | {m['FD']['exp']:.6f} | {m['FD']['pose']:.6f} | {m['MSE']['exp']:.6f} | {m['MSE']['pose']:.6f} |")
lines+=['','- DIM Table1 的 L2L report：FD 33.93/0.06、P-FD 35.88/0.06、MSE 0.93/0.01。论文明确将L2L在ViCo上以EMOCA重新训练；公开Conan/Fallon/Stephen/Trevor权重是不同训练数据与DECA输出。缺少公开写实renderer，不能将这些DECA系数当成EMOCA。','- [DualTalk Table4](https://arxiv.org/html/2505.18096v1#S5.T4) report：FD 22.27/0.05、P-FD 23.81/0.05、MSE 0.58/0.01。现有公开完整权重SHA256已对齐HF发布，核心源码与当前官方版本逐字节一致，旧归档的官方demo前向一致性验收为0。43条是self-zero迁移结果；官方公开模型是否正是Table4的ViCo模型尚无证据，原实现neck与DECA global head几何尚未确认，因此仅能视为迁移代理，不能宣称复现失败或按该差距评价作者。','',
'## 4. eval config自身仍需固定的边界','',
'GT锚点（与DyStream共同帧集相同）：','',
'| 统计 | DIM report GT | 当前GT |','|---|---:|---:|',
'| Var Exp | 0.93 | 0.904523 |','| Var Pose | 0.01 | 0.012038 |','| SID Exp | 5.03 | 5.041578 |','| SID Pose | 4.07 | 3.883174 |','',
'Var Exp相差约−2.74%；Pose的report只保留两位小数，不能据其舍入值判断细微相对误差。GT统计接近不足以证明完整作者协议已复现。','',
'只将DyStream的有效帧集从共同16,238帧改为自身/GT/speaker共同16,245帧：FD Exp 31.715231→31.705459，变化很小；SID Exp 4.396098→4.205501，SID Pose 3.335322→3.417671。SID重新在GT上拟合KMeans，即使random_state固定，删掉7帧也会改变聚类解。横向比较必须固定UUID、帧掩码、聚类输入；不能每个方法各拟合各自的GT码本后忽略这一变化。','',
'当前是单次预测；DIM代码中的多候选GT择优、原生系数与视频反提取的差别仍然存在，不能用公式实现一致代替完整protocol等价。','',
'## 5. 可用性核查与当前缺项','',
'| 方法 | 本轮核查结果 |','|---|---|',
'| DyStream | 官方权重可用，已有43条RGB；本轮重新评分EMOCA，但原报告听者任务不是ViCo |',
'| Learning2Listen | 四套官方人物权重／172条动作可用；没有ViCo重训EMOCA权重与写实renderer |',
'| DualTalk | 官方权重与43条动作可用；ViCo表格专用模型对应关系和neck/global语义未确认 |',
'| ListenFormer | 已有本地完整重训模型；作者2025年确实发过模型链接，当前分享已取消。不能写成“从未发布权重” |',
'| DIM／RLHG | 当前官方树／Releases／相关issue未取得核心生成权重，辅助LPIPS/PIRender不算生成器 |',
'| ReactFace | 当前仓库无核心checkpoint；预训练模型请求issue尚无回复 |',
'| DiTaiListener | 项目页Code/Model按钮href为空，本轮未取得可运行模型 |',
'| DiffListener（CVPR2025） | 项目页Code链接仍为YOUR REPO HERE；与另一篇同名ICASSP工作区分 |',
'| INFP（补查） | 项目页Code/Data无下载链接，本轮未取得官方生成模型 |','',
'检查证据保存在`sources/`，包括完整GitHub trees、Releases、open/closed issues与关键评论、HF清单、百度错误回执、项目页及论文文本。未联系作者或发送外部消息。','',
'## 6. 文件与复跑','',
'- [所有重算指标CSV](rescored_metrics.csv)、[详细JSON](RESULTS.json)：DyStream与四套L2L。',
'- [ListenFormer与report逐项差值](listenformer_vs_report.csv)、[详细比较](LISTENFORMER_COMPARISON.json)。',
'- [DualTalk迁移代理](DUALTALK_PROXY.json)、[当前官方发布与归档代码一致性](DUALTALK_RELEASE_PARITY.json)。',
'- [数值审计](AUDIT.json)、[补充审计](ADDITIONAL_AUDIT.json)、[DIM原函数一致性](AUTHOR_FUNCTION_PARITY.json)。','',
'```bash','env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \\\n  compare/.envs/vico_more/bin/python experiments/p2_vico_report_alignment_20261110/rescore.py','env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \\\n  compare/.envs/vico_more/bin/python experiments/p2_vico_report_alignment_20261110/rescore_additional.py','```','',
'判断：当前可报告“数值链路复核通过、若干结果接近，但完整report对齐尚未证实”。最具体的差异是ListenFormer本地重训的FID和头角距离；对公开权重迁移方法，先消除权重/系数空间差异才有条件将误差归因于评测配置。']
(P/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('REPORT.md written')
