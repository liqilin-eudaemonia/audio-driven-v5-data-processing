# 参考方法实测结果汇总：ViCo、Seamless、ResponseNet

整理时间：2026-11-13 21:24（北京时间）。范围为本轮5个参考方法、8个配置：Avatar Forcing、DyStream、ListenFormer、Learning2Listen四套人物权重、DualTalk。L2L四套分别报告。

**已有完整结果：ViCo正式test43、历史固定Seamless200、历史固定ResponseNet200。完整Seamless6624和ResponseNet586尚未完成；当前两项全量任务均已暂停，自动恢复关闭。** 本文使用已封存的完整范围成绩；全量部分只列产物和评分覆盖，不把阶段结果当成全量成绩。

[范围与模型](#1-范围与模型) · [ViCo43](#2-vico正式test43) · [Seamless200](#3-seamless历史固定200) · [ResponseNet200](#4-responsenet历史固定200) · [全量暂停进度](#5-全量测试暂停进度) · [来源与精确值](#6-来源精确值与论文对照)

## 1. 范围与模型

| 方法／配置 | 使用权重 | 视频与动作结果说明 |
|---|---|---|
| Avatar Forcing | 官方公开权重 | 本文主结果为speaker音频＋listener首帧；listener音频置零、speaker视觉置零latent，标记audio-only适配 |
| DyStream | 官方公开权重 | 沿用已验收的listener输入适配 |
| ListenFormer | 本地ViCo完整200轮重训best.pt | 不是作者公开checkpoint；视频经PIRenderer生成 |
| L2L Conan／Fallon／Stephen／Trevor | 四套官方人物权重及配套码本、统计 | 各自预测动作，视频使用公开GPAvatar；不是L2L原写实渲染器，也不是论文中在ViCo重训的L2L |
| DualTalk | 官方公开权重 | 原生动作和DECA代理诊断；RGB渲染／全局头部映射尚未验证，视频指标为N/A |

- ViCo图像表使用既有256×256／30fps评分口径；图像FID与动作FD是不同指标。
- Seamless固定200：512×512、25fps、face-padding=0。ResponseNet固定200：512×512、保留GT帧率、face-padding=64；GT按既定PIL bicubic方式缩放。
- 双200视频FID／FVD按各自完整200范围的有效特征汇总，不平均分片FID／FVD。Emo-Acc使用0–1比例。
- `partial`表示该范围的评测已经结束，但部分样本／指标不可测；保留有效覆盖和N/A，不补零。它与“6624／586全量尚未完成”不同。
- 本轮方法均未生成listener音轨，音频生成指标为N/A。表中数值保留6位小数，精确值及覆盖以原始JSON／CSV为准。
- EMOCA、DECA、BFM、SPECTRE表示与参数口径分别保留，不把同名FD跨表排名。下列Var沿用各来源的既定定义；统一EMOCA主表采用逐系数跨帧population方差再平均。

## 2. ViCo正式test43

8个参考配置各43条推理及对应评分已完成；其中7个RGB配置、DualTalk仅原生动作。

### 2.1 图像／身份五项

共43条、16,288个评分帧。来源：[image_metrics.csv](../experiments/p2_vico_avatarforcing_added_20261111/image_metrics.csv)。

| 方法 | PSNR ↑ | SSIM ↑ | FID ↓ | CSIM ↑ | CPBD ↑ |
|---|---:|---:|---:|---:|---:|
| Avatar Forcing（audio-only） | 16.183142 | 0.572531 | 39.140994 | 0.466332 | 0.190193 |
| DyStream | 18.379981 | 0.618699 | 35.067230 | 0.467174 | 0.226187 |
| ListenFormer（本地重训） | 18.887633 | 0.641760 | 30.842915 | 0.602755 | 0.193178 |
| L2L Conan＋GPAvatar | 14.215021 | 0.514727 | 87.993300 | 0.440848 | 0.215046 |
| L2L Fallon＋GPAvatar | 13.946953 | 0.496049 | 91.571952 | 0.422281 | 0.221127 |
| L2L Stephen＋GPAvatar | 14.156595 | 0.505480 | 94.723012 | 0.432411 | 0.223262 |
| L2L Trevor＋GPAvatar | 13.788192 | 0.493226 | 94.580409 | 0.405309 | 0.230406 |
| DualTalk（原生动作） | N/A | N/A | N/A | N/A | N/A |

### 2.2 统一RGB→EMOCA动作指标

7个参考RGB配置，各43条；共同有效帧16,238。listener 1:T对齐speaker 0:T−1，采用相同共同有效帧交集。下表不包含自有joint模型，完整九行源表仍保留。

| 方法 | FD Exp | FD Pose | P-FD Exp | P-FD Pose | MSE Exp | MSE Pose |
|---|---:|---:|---:|---:|---:|---:|
| Avatar Forcing（audio-only） | 37.633272 | 0.068397 | 39.178973 | 0.071590 | 0.897029 | 0.014181 |
| DyStream | 31.715231 | 0.067969 | 33.394710 | 0.069342 | 0.807346 | 0.013097 |
| ListenFormer（本地重训） | 32.334237 | 0.098622 | 32.986166 | 0.099563 | 0.695527 | 0.016980 |
| L2L Conan＋GPAvatar | 62.679655 | 0.107203 | 64.681366 | 0.108949 | 1.521195 | 0.021275 |
| L2L Fallon＋GPAvatar | 69.255731 | 0.135133 | 71.364966 | 0.137539 | 1.643143 | 0.026208 |
| L2L Stephen＋GPAvatar | 74.354409 | 0.124185 | 76.200755 | 0.126149 | 1.711027 | 0.024168 |
| L2L Trevor＋GPAvatar | 61.132577 | 0.173117 | 62.897938 | 0.174421 | 1.421860 | 0.030332 |

| 方法 | SID Exp | SID Pose | Var Exp | Var Pose | rPCC Exp | rPCC Pose |
|---|---:|---:|---:|---:|---:|---:|
| Avatar Forcing（audio-only） | 4.303114 | 3.678300 | 0.796301 | 0.010247 | 0.042412 | 0.038954 |
| DyStream | 4.396098 | 3.335322 | 0.614315 | 0.009441 | 0.030032 | 0.004435 |
| ListenFormer（本地重训） | 4.365777 | 3.441167 | 0.879423 | 0.015105 | 0.004554 | 0.036108 |
| L2L Conan＋GPAvatar | 3.695481 | 2.041414 | 0.639775 | 0.002957 | 0.118562 | 0.086077 |
| L2L Fallon＋GPAvatar | 3.547120 | 2.223506 | 0.634490 | 0.003837 | 0.073322 | 0.247026 |
| L2L Stephen＋GPAvatar | 3.275903 | 1.873696 | 0.607269 | 0.003264 | 0.117501 | 0.187773 |
| L2L Trevor＋GPAvatar | 3.331143 | 0.425406 | 0.611248 | 0.001790 | 0.042864 | 0.526125 |

来源：[motion_metrics.csv](../experiments/p2_vico_avatarforcing_added_20261111/motion_metrics.csv)；逐条共同覆盖与数学实现对照：[MOTION_RESULTS.json](../experiments/p2_vico_avatarforcing_added_20261111/MOTION_RESULTS.json)。DualTalk未加入该统一RGB→EMOCA表。

### 2.3 原生DECA／DualTalk代理动作诊断

以下为已有43条原生／代理动作结果，与上一节RGB反提取EMOCA不同。DualTalk的neck／global-head代理映射仍有限制，不能据此宣称复现其论文表。

| 方法 | FD Exp | FD Pose | P-FD Exp | P-FD Pose | MSE Exp | MSE Pose |
|---|---:|---:|---:|---:|---:|---:|
| L2L Conan（原生动作） | 7.094259 | 0.099602 | 7.266816 | 0.101545 | 0.173284 | 0.020790 |
| L2L Fallon（原生动作） | 8.080833 | 0.140326 | 8.268756 | 0.142928 | 0.199416 | 0.027740 |
| L2L Stephen（原生动作） | 6.484761 | 0.118793 | 6.638364 | 0.121084 | 0.161006 | 0.023947 |
| L2L Trevor（原生动作） | 5.471074 | 0.166756 | 5.623629 | 0.168365 | 0.138949 | 0.029540 |
| DualTalk（原生动作） | 66.003166 | 0.216280 | 66.166885 | 0.218324 | 1.347036 | 0.037435 |

| 方法 | SID Exp | SID Pose | Var Exp | Var Pose | rPCC Exp | rPCC Pose |
|---|---:|---:|---:|---:|---:|---:|
| L2L Conan（原生动作） | 3.337065 | 1.949145 | 0.030809 | 0.002957 | 0.044059 | 0.075866 |
| L2L Fallon（原生动作） | 3.048272 | 2.178443 | 0.036581 | 0.003787 | 0.181860 | 0.215971 |
| L2L Stephen（原生动作） | 2.020645 | 2.185899 | 0.025718 | 0.003160 | 0.101896 | 0.162534 |
| L2L Trevor（原生动作） | 2.988023 | 0.810084 | 0.026537 | 0.001047 | 0.046347 | 0.458612 |
| DualTalk（原生动作） | 1.835040 | 1.930289 | 0.143892 | 0.005231 | 0.569896 | 0.731422 |

来源：[report_vs_local_all.csv](../experiments/p2_vico_avatarforcing_added_20261111/report_vs_local_all.csv)的原生DECA／DECA transfer proxy行，保留各自空间标签。

### 2.4 ListenFormer原生BFM补充

这是原生BFM系数L1×100，不能当成上面EMOCA／DECA的Fréchet FD。

| 指标 | 本地结果 |
|---|---:|
| BFM_L1x100_angle | 10.299858 |
| BFM_L1x100_exp | 12.398529 |
| BFM_L1x100_trans | 5.477706 |

ViCo原始结果和执行说明：[AF加入后的完整同表](../experiments/p2_vico_avatarforcing_added_20261111/README.md)；[既有完整ViCo评测](../experiments/p2_vico_full_eval_20261111/README.md)。

## 3. Seamless历史固定200

200条固定测试输入，7个RGB配置及DualTalk原生动作评测均已结束。L2L每套生成198条，另2条官方首帧追踪无身份，记N/A。

### 3.1 统一视频八项

| 方法 | 评分状态 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | FVD ↓ | Emo-Acc ↑ | Emo-Div 生成 / GT | ΔResp ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Avatar Forcing（audio-only） | partial | 14.914447 | 0.637524 | 0.401719 | 36.140350 | 333.901553 | 0.386192 | 2.606061 / 3.883838 | 0.899865 |
| DyStream | partial | 15.458883 | 0.662413 | 0.382920 | 23.262600 | 243.277961 | 0.412665 | 3.346734 / 3.874372 | 0.615545 |
| ListenFormer（本地重训） | partial | 15.652007 | 0.654384 | 0.411067 | 41.935674 | 306.309211 | 0.462517 | 1.969697 / 3.848485 | 1.082845 |
| L2L Conan＋GPAvatar | partial | 7.522471 | 0.340840 | 0.654973 | 97.876254 | 543.743436 | 0.367276 | 3.258883 / 3.883249 | 0.483780 |
| L2L Fallon＋GPAvatar | partial | 7.456151 | 0.333557 | 0.660628 | 95.155192 | 536.191616 | 0.352753 | 3.106599 / 3.883249 | 0.455473 |
| L2L Stephen＋GPAvatar | partial | 7.451084 | 0.332786 | 0.654541 | 96.708253 | 539.018273 | 0.361788 | 2.994924 / 3.883249 | 0.456728 |
| L2L Trevor＋GPAvatar | partial | 7.416772 | 0.327568 | 0.659255 | 102.391116 | 555.118996 | 0.360452 | 3.040609 / 3.878173 | 0.600809 |
| DualTalk（原生动作） | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### 3.2 有效评分覆盖

| 方法 | 请求数 | 生成视频 | PSNR有效对 | FVD有效对 | 情绪有效身份 | ΔResp有效阶段 |
|---|---:|---:|---:|---:|---:|---:|
| Avatar Forcing（audio-only） | 200 | 200 | 200 | 200 | 198 | 198 |
| DyStream | 200 | 200 | 200 | 200 | 199 | 199 |
| ListenFormer（本地重训） | 200 | 200 | 200 | 200 | 198 | 198 |
| L2L Conan＋GPAvatar | 200 | 198 | 198 | 198 | 197 | 197 |
| L2L Fallon＋GPAvatar | 200 | 198 | 198 | 198 | 197 | 197 |
| L2L Stephen＋GPAvatar | 200 | 198 | 198 | 198 | 197 | 197 |
| L2L Trevor＋GPAvatar | 200 | 198 | 198 | 198 | 197 | 197 |

各列支持单位不同：有效对／视频身份／阶段不能混为同一个样本数。更多有效帧、有效时长和错误原因见各方法原始结果JSON。

### 3.3 原生DECA动作诊断

DualTalk和L2L各200条动作。这里列出已有的原生／代理动作评分，不与RGB八项或ViCo的EMOCA表合并排名。

| 方法 | FD Exp | FD Pose | P-FD Exp | P-FD Pose | MSE Exp | MSE Pose |
|---|---:|---:|---:|---:|---:|---:|
| L2L Conan（原生动作） | 8.336379 | 0.174132 | 8.591460 | 0.179463 | 0.198679 | 0.034210 |
| L2L Fallon（原生动作） | 7.881584 | 0.170078 | 8.142534 | 0.175893 | 0.193356 | 0.034003 |
| L2L Stephen（原生动作） | 7.500165 | 0.150559 | 7.744441 | 0.156270 | 0.180037 | 0.029785 |
| L2L Trevor（原生动作） | 8.088742 | 0.171564 | 8.312550 | 0.174773 | 0.193087 | 0.030980 |
| DualTalk（原生动作） | 78.669650 | 0.184773 | 78.908168 | 0.187920 | 1.601329 | 0.032301 |

| 方法 | SID Exp | SID Pose | Var Exp | Var Pose | rPCC Exp | rPCC Pose |
|---|---:|---:|---:|---:|---:|---:|
| L2L Conan（原生动作） | 2.760343 | 1.816217 | 0.036680 | 0.004285 | 0.024726 | 0.006641 |
| L2L Fallon（原生动作） | 2.969338 | 2.517270 | 0.038563 | 0.004301 | 0.160452 | 0.061737 |
| L2L Stephen（原生动作） | 2.025437 | 2.416776 | 0.027523 | 0.003350 | 0.112611 | 0.033188 |
| L2L Trevor（原生动作） | 1.577723 | 1.096684 | 0.031485 | 0.001176 | 0.154764 | 0.109782 |
| DualTalk（原生动作） | 2.339003 | 3.092291 | 0.194615 | 0.006898 | 0.536743 | 0.259852 |

来源：[双200视频精确值](../experiments/p2_reference_methods_dualtest_20261112/image_metrics.csv)、[原生动作精确值](../experiments/p2_reference_methods_dualtest_20261112/native_motion_metrics.csv)、[逐方法JSON目录](../experiments/p2_reference_methods_dualtest_20261112/results)。

## 4. ResponseNet历史固定200

200对固定测试输入，7个RGB配置及DualTalk原生动作评测均已结束。L2L每套生成199条，另1条官方首帧追踪无身份，记N/A。

### 4.1 统一视频八项

| 方法 | 评分状态 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | FVD ↓ | Emo-Acc ↑ | Emo-Div 生成 / GT | ΔResp ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Avatar Forcing（audio-only） | complete | 16.435502 | 0.649357 | 0.322038 | 17.048602 | 208.851374 | 0.586233 | 2.310000 / 3.345000 | 0.358986 |
| DyStream | complete | 16.410669 | 0.640691 | 0.341683 | 17.530999 | 184.480845 | 0.555191 | 3.080000 / 3.345000 | 0.238613 |
| ListenFormer（本地重训） | complete | 17.567966 | 0.675292 | 0.288216 | 28.199962 | 243.884933 | 0.657076 | 1.640000 / 3.330000 | 0.442052 |
| L2L Conan＋GPAvatar | partial | 10.381020 | 0.528748 | 0.437828 | 70.845493 | 368.577893 | 0.307710 | 4.231156 / 3.346734 | 0.414377 |
| L2L Fallon＋GPAvatar | partial | 10.362497 | 0.527659 | 0.428358 | 71.127401 | 376.872932 | 0.357583 | 2.854271 / 3.346734 | 0.402361 |
| L2L Stephen＋GPAvatar | partial | 10.281026 | 0.526000 | 0.424883 | 71.747117 | 395.821227 | 0.297135 | 3.175879 / 3.346734 | 0.366518 |
| L2L Trevor＋GPAvatar | partial | 9.967088 | 0.496574 | 0.479971 | 76.587869 | 397.302041 | 0.398691 | 3.417085 / 3.346734 | 0.203236 |
| DualTalk（原生动作） | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### 4.2 有效评分覆盖

| 方法 | 请求数 | 生成视频 | PSNR有效对 | FVD有效对 | 情绪有效身份 | ΔResp有效阶段 |
|---|---:|---:|---:|---:|---:|---:|
| Avatar Forcing（audio-only） | 200 | 200 | 200 | 200 | 200 | 200 |
| DyStream | 200 | 200 | 200 | 200 | 200 | 200 |
| ListenFormer（本地重训） | 200 | 200 | 200 | 200 | 200 | 200 |
| L2L Conan＋GPAvatar | 200 | 199 | 199 | 199 | 199 | 199 |
| L2L Fallon＋GPAvatar | 200 | 199 | 199 | 199 | 199 | 199 |
| L2L Stephen＋GPAvatar | 200 | 199 | 199 | 199 | 199 | 199 |
| L2L Trevor＋GPAvatar | 200 | 199 | 199 | 199 | 199 | 199 |

各列支持单位不同：有效对／视频身份／阶段不能混为同一个样本数。更多有效帧、有效时长和错误原因见各方法原始结果JSON。

### 4.3 原生DECA动作诊断

DualTalk和L2L各200条动作。这里列出已有的原生／代理动作评分，不与RGB八项或ViCo的EMOCA表合并排名。

| 方法 | FD Exp | FD Pose | P-FD Exp | P-FD Pose | MSE Exp | MSE Pose |
|---|---:|---:|---:|---:|---:|---:|
| L2L Conan（原生动作） | 9.484411 | 0.085405 | 9.697587 | 0.087368 | 0.229025 | 0.016337 |
| L2L Fallon（原生动作） | 5.754180 | 0.069703 | 5.974404 | 0.071767 | 0.158083 | 0.013876 |
| L2L Stephen（原生动作） | 6.808539 | 0.060216 | 7.002859 | 0.062237 | 0.171120 | 0.012060 |
| L2L Trevor（原生动作） | 7.516109 | 0.109908 | 7.713268 | 0.111144 | 0.191817 | 0.019383 |
| DualTalk（原生动作） | 65.531238 | 0.107532 | 65.736229 | 0.108767 | 1.348566 | 0.018727 |

| 方法 | SID Exp | SID Pose | Var Exp | Var Pose | rPCC Exp | rPCC Pose |
|---|---:|---:|---:|---:|---:|---:|
| L2L Conan（原生动作） | 2.795377 | 2.542720 | 0.038104 | 0.004811 | 0.057346 | 0.195446 |
| L2L Fallon（原生动作） | 2.988096 | 2.858853 | 0.038774 | 0.004067 | 0.011049 | 0.235767 |
| L2L Stephen（原生动作） | 2.773230 | 2.939229 | 0.029628 | 0.003214 | 0.000523 | 0.233937 |
| L2L Trevor（原生动作） | 1.992260 | 0.387110 | 0.032935 | 0.001163 | 0.117943 | 0.132154 |
| DualTalk（原生动作） | 3.777600 | 2.530766 | 0.175546 | 0.005911 | 0.165798 | 0.190621 |

来源：[双200视频精确值](../experiments/p2_reference_methods_dualtest_20261112/image_metrics.csv)、[原生动作精确值](../experiments/p2_reference_methods_dualtest_20261112/native_motion_metrics.csv)、[逐方法JSON目录](../experiments/p2_reference_methods_dualtest_20261112/results)。

## 5. 全量测试暂停进度

暂停时间：2026-11-13 14:57:14（北京时间）；两项全量任务均已停止，自动恢复关闭。以下是完成回执覆盖数，不是全量分数。

### 5.1 Seamless完整6624

| 方法 | 原生动作 | 已生成视频 | 视频N/A | 已评分请求 |
|---|---:|---:|---:|---:|
| Avatar Forcing（audio-only） | — | 1283 | 0 | 1200 |
| DyStream | — | 631 | 0 | 300 |
| ListenFormer（本地重训） | 987 | 985 | 0 | 200 |
| L2L Conan＋GPAvatar | 975 | 386 | 2 | 200 |
| L2L Fallon＋GPAvatar | 975 | 386 | 2 | 200 |
| L2L Stephen＋GPAvatar | 975 | 386 | 2 | 200 |
| L2L Trevor＋GPAvatar | 975 | 386 | 2 | 200 |
| DualTalk（原生动作） | 988 | N/A | — | 待全量动作汇总 |

共4,443份有效生成视频、2,500条已评分请求（含L2L共8条生成N/A）。尚无完整6624的最终FID／FVD等汇总结果。历史固定200已包含在上述数字中，不再累加。

### 5.2 ResponseNet完整586

新增386对尚未开始推理。以下全部来自历史固定200缓存；完整586最终成绩尚未生成。

| 方法 | 已有原生动作 | 已有视频 | 视频N/A | 本轮可复用评分请求 |
|---|---:|---:|---:|---:|
| Avatar Forcing（audio-only） | — | 200 | 0 | 200 |
| DyStream | — | 200 | 0 | 0 |
| ListenFormer（本地重训） | 200 | 200 | 0 | 200 |
| L2L Conan＋GPAvatar | 200 | 199 | 1 | 200 |
| L2L Fallon＋GPAvatar | 200 | 199 | 1 | 200 |
| L2L Stephen＋GPAvatar | 200 | 199 | 1 | 200 |
| L2L Trevor＋GPAvatar | 200 | 199 | 1 | 200 |
| DualTalk（原生动作） | 200 | N/A | — | 待全量动作汇总 |

DyStream历史200对已经有完整成绩（见第4节），完整586任务的评分指纹不兼容，需要重评；因此“本轮可复用评分请求为0”不代表此前没有测过。

全量原始状态：[Seamless6624](../experiments/p2_reference_methods_seamless6624_20261112/status.json)、[ResponseNet586](../experiments/p2_reference_methods_responsenet586_20261112/status.json)；[产物去重统计与暂停记录](../experiments/p2_reference_methods_seamless6624_20261112/EVALUATION_INVENTORY_20261113.md)。

## 6. 来源、精确值与论文对照

本文件仅汇总已有主测试配置的本地实测结果，没有重新推理、重新评分或启动训练。已完成的封存结果保持原文件。

“测试执行完成”不等于“论文report数值已复现”：ListenFormer检查点、L2L人物权重／渲染器、AF输入与动作表示、DualTalk代理几何等差异已在方法说明中标明。ViCo论文对照CSV保留reported／local／comparability及来源备注，不据不同协议的差值归因于模型或官方实现。

- 本汇总精确数值长表：[REFERENCE_METHODS_RESULTS_20261113.csv](REFERENCE_METHODS_RESULTS_20261113.csv)。
- 本汇总来源摘要与核验记录：[REFERENCE_METHODS_RESULTS_20261113.sources.json](REFERENCE_METHODS_RESULTS_20261113.sources.json)。
- ViCo论文report／本地逐项对照：[report_vs_local_all.csv](../experiments/p2_vico_avatarforcing_added_20261111/report_vs_local_all.csv)。
- 固定双200原始最终结果：[RESULTS.md](../experiments/p2_reference_methods_dualtest_20261112/RESULTS.md)；[完成清单](../experiments/p2_reference_methods_dualtest_20261112/COMPLETE.json)。
- 各方法模型、权重、输入和推理入口：[逐方法交接索引](../related/method_handoffs/README.md)。
