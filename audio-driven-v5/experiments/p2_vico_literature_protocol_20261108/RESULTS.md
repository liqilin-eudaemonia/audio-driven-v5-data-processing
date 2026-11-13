# ViCo：EMOCA与DIM发布指标的对齐实测

**评价组成已对齐，发表时的完整执行协议仍有未公开项。** 本地结果不能标为DIM原表的严格复现，也不据此自动计算跨来源提升率。

[逐项协议对照](ALIGNMENT.md) · [完整JSON](RESULTS.json) · [全部指标CSV](paper_tables/all_metrics.csv) · [每条FD/P-FD/MSE](paper_tables/ours_per_clip.csv)

| 来源 | 方法 | FD Exp ↓ | FD Pose ↓ | MSE Exp ↓ | MSE Pose ↓ | SID Exp | SID Pose | Var Exp | Var Pose | rPCC Exp ↓ | rPCC Pose ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 引用：DIM Table1 | ELP reimplementation | 47.17 | 0.08 | 0.98 | 0.02 | 1.76 | 1.66 | 1.49 | 0.02 | 0.09 | 0.01 |
| 引用：DIM Table1 | RLHG | 39.02 | 0.07 | 0.86 | 0.01 | 3.62 | 3.17 | 1.52 | 0.02 | 0.08 | 0.02 |
| 引用：DIM Table1 | L2L | 33.93 | 0.06 | 0.93 | 0.01 | 2.77 | 2.66 | 0.83 | 0.02 | 0.06 | 0.08 |
| 引用：DIM Table1 | L2L + CANDOR | 31.03 | 0.06 | 0.87 | 0.01 | 3.53 | 3.05 | 0.83 | 0.02 | 0.06 | 0.09 |
| 引用：DIM Table1 | DIM + CANDOR | 23.88 | 0.06 | 0.70 | 0.01 | 3.71 | 2.35 | 1.53 | 0.02 | 0.06 | 0.03 |
| 引用：INFP Table3 | INFP original | 18.63 | 0.07 | 0.51 | 0.01 | 4.78 | 3.92 | 2.83 | 0.18 | N/A | N/A |
| 引用：AvatarForcing Table4 | INFP* / AvatarForcing reimplementation | 17.52 | 0.07 | N/A | N/A | 2.19 | 3.20 | 2.10 | 0.03 | 0.01 | 0.07 |
| 引用：AvatarForcing Table4 | Avatar Forcing | 16.64 | 0.05 | N/A | N/A | 3.12 | 3.00 | 2.80 | 0.03 | 0.01 | 0.01 |
| 引用：DIM Table1 | GT / DIM | N/A | N/A | N/A | N/A | 5.03 | 4.07 | 0.93 | 0.01 | N/A | N/A |
| 本地：RGB→EMOCA | joint | 25.89253 | 0.08714 | 0.65253 | 0.01601 | 4.55095 | 3.52459 | 1.42274 | 0.02207 | 0.01039 | 0.03493 |
| 本地：RGB→EMOCA | joint_sft2752 | 24.33909 | 0.06182 | 0.71181 | 0.01243 | 4.64889 | 3.66934 | 1.47158 | 0.01867 | 0.00600 | 0.03062 |
| 本地：GT提取锚点 | GT / our EMOCA | N/A | N/A | N/A | N/A | 5.04158 | 3.88317 | 1.45611 | 0.02327 | N/A | N/A |

## 人脸检出与实际评分覆盖

全部43条pair均保留；去除首帧后候选16245帧，共同评分16238帧，共同掩码排除7帧。

| 流 | 原始帧数 | 未检出人脸帧数 | 涉及pair数 |
|---|---:|---:|---:|
| speaker | 16288 | 0 | 0 |
| gt | 16288 | 0 | 0 |
| joint | 16288 | 7 | 1 |
| joint_sft2752 | 16288 | 0 | 0 |

缺失帧不赋予零误差、不插值；这些帧没有进入动作距离，失败数单独报告。原生系数基线不存在同样的RGB检测步骤，比较时需要保留该差异。
[覆盖CSV](paper_tables/coverage.csv) · [逐帧有效范围JSON](RESULTS.json)

[首个未检出案例的原始帧截图](diagnostics/missing_face_first_case.png) · [对应视频与帧号](diagnostics/MISSING_FACE_FIRST_CASE.json)


## 本次执行及差异

- 使用DIM预处理明确指定的EMOCA_v2_lr_mse_20；pose6+exp50，原始系数，无统计重标定。
- 固定43个listener UUID，不含OOD；两组预测共用同一有效帧掩码。listener去首帧，与speaker前T−1帧对应，沿发布tester。
- FD/P-FD/MSE逐片段平均；SID/Var/rPCC按照固定版本DIM代码执行。P-FD见完整CSV。
- 原生动作系数与RGB反提取有差异；我们为后者，不伪称原生输出。
- 我们固定单次样本；DIM公开tester的十次GT择优是否用于发表主表未确认，未对我们进行GT择优。
- 发布加载器的重复audio-key覆盖、长度筛选及常量speaker问题尚不能代表论文实际运行。本次保留真实speaker和完整UUID，不把猜测写成已确认。
- GT自比较、GT统计锚点和逐指标作者代码对照均单独保存；与发表GT不一致时如实报告，不反推缩放系数。

[公开加载器范围敏感性诊断](LOADER_SCOPE_DIAGNOSTIC.json)：另算audio-key覆盖和视频5–1024帧限制的投影；没有作者HuBERT长度，不能称其为作者的实际test。本表仍保留预先固定的完整43条。

[GT统计对照CSV](paper_tables/gt_anchor.csv)：同时保留DIM发表的GT数值和本次GT提取统计，不通过缩放使其相等。

## 原始出处

[DIM Table1及补充A](../../writing/17_DIM_ECCV2024.pdf)；[INFP Table3、§5.2.2](../../writing/35_INFP_CVPR2025.pdf)；[Avatar Forcing Table4](../../writing/43_Avatar_Forcing_CVPR2026.pdf)。
RLHG/L2L/DIM旧行源于DIM；INFP原版与Avatar Forcing重实现的INFP*分行，未报告的格子为N/A。

[正式发表版本逐行／逐格核验](LITERATURE_CELL_PROVENANCE.json) · [结果来源BibTeX](paper_tables/result_sources.bib)。
DIM的正式ECCV版本Table1在PDF第10页，本地预印本在第11页；引用行数值一致。INFP Table3在第8页，Avatar Forcing Table4在第6页。引用结果来源时，还应同时引用对应方法原文。

## 对齐程度的实测结论

GT表情Var：本次1.45611，论文0.93；GT姿态Var：本次0.02327，论文0.01。
GT统计并未完全对齐；文献行可以按来源引用，我们的行应标为“按DIM公开EMOCA及指标设置实测，存在已披露差异”。当前证据不足以称为完全相同发表协议下的严格排名，不能只凭数值计算跨来源提升率。
这个差异来自GT侧，无法通过更换我们的生成权重来解决；也不通过调尺度、改裁剪或按分数选择子集强行匹配。作者实际有效清单、GT特征或完整预处理配置是进一步定位所需的证据。
