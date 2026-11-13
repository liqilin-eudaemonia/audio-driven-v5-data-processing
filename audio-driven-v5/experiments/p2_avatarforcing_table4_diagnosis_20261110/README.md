# Avatar Forcing / ViCo：Table4差异定位

**后续更正（完整复现阶段）：** 现有`/public/0_DATA/2_Avatar/ViCo/ViCo-Dataset.zip`内同时存在`videos/cropped`和`videos/original`；此前仅沿用前者，遗漏了压缩包中的原始层。已从已有归档恢复test43双方原视频，原始宽视野数据无需重新下载。原始层样例960×1080并带音轨。完整推进见[后续复现记录](../p2_avatarforcing_table4_full_reproduction_20261110/README.md)。

本轮同时核对了 Avatar Forcing 论文／正式补充材料／官方推理代码，以及 ViCo 官方与 DIM 的评测代码。原43条结果保持封存。本报告把实测确认的差异与作者未披露的配置分开，未按接近论文分数的程度选择设置。

## 当前结论

三项已经证实的问题：**可用ViCo裁脸画面与作者示例的画面范围不同；ViCo官方、DIM和Avatar Forcing并非一套评分协议；人脸检测器改变全局cuDNN状态，结合官方整段min/max导出，会放大数值路径差异。** 当前8项是按照论文文字和公开代码重建的组合，作者没有发布Table4完整评分入口。运行状态的严格干预结果见第4节。

不能把当前差距全部归于权重失效，也不能据此认定评分框架算错。已经排除了一部分常见实现错误；公开材料仍不足以唯一确定作者完整量化运行配置。

## 1. 画面范围：取得了作者实际发布的视频作为对照

从[CVPR正式补充包](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Ki_Avatar_Forcing_Real-Time_CVPR_2026_supplemental.zip)提取了两条listening比较视频。其音轨与本地以下测试片段相关性分别为0.9876和0.9852，偏移分别约22ms和5ms；对应相同源对话时段，不只是人物相同。

| 作者示例 | 对应本地test UUID | 时长 | 作者输出脸宽占画面 | 本地输出脸宽占画面 |
|---|---|---:|---:|---:|
| 03_listening_01 | ClNUxPsUByg_000210_000221.0 | 11秒 | 37.3% | 84.8% |
| 03_listening_02 | Ew2z_sYABE0_000063_000083.0 | 20秒 | 39.8% | 83.6% |

作者输出保留上半身／周围背景，本地官方256视频是紧裁脸。将256放大为512不能恢复裁掉的内容。我们虽然执行了官方预处理方法，但受输入图像边界限制，得到的实际视野仍不同。这里确认的是发布示例与当前执行的区别，不能断言示例就是Table4全部量化输出。

进一步做了纯视野控制：保持同一张作者画面中的脸部像素和关键点对应不变，只移除周围画面，再走同一个SPECTRE裁剪／编码函数。两例表情系数的平方L2偏移为**4.12和8.51**。SPECTRE并非对裁剪背景完全不敏感；这种变化足以明显影响FD，不能只对齐最终图像宽高。

同一评分器对两条发布视频的Exp FD分别为59.76和16.13，本地两例分别37.91和34.04，未出现一致的提升方向。**不能挑16.13接近16.64的一例来宣布整表已复现，也不能把两条例子的平均与Table4全测试集平均作等价检验。** 正式视频含标题、压缩与视野差异，量化结果仅用于诊断。

证据：[音轨匹配](AUTHOR_DEMO_AUDIO_ALIGNMENT.json) · [发布示例原图1](published_supplement/03_listening_01.png) · [发布示例原图2](published_supplement/03_listening_02.png) · [完整读出](AUTHOR_DEMO_READOUT.json) · [纯视野控制](FRAMING_PROBE.json)。

## 2. 三条评测来源必须分开

| 项目 | ViCo原官方框架 | DIM发布代码 | Avatar Forcing论文／本轮重建 |
|---|---|---|---|
| 动作提取 | Deep3DFaceRecon／BFM | EMOCA_v2_lr_mse_20 | 论文明确SPECTRE；本轮遵循 |
| 动作维度 | Exp64，angle3、trans3 | Exp50＋pose6 | Exp50＋pose6 |
| 动作误差入口 | 官方issue15给出逐帧L1实现 | FD／P-FD／MSE／SID／Var／rPCC | Table4只列FD／rPCC／SID／Var |
| SID | 不等同下游论文的实现 | GT拟合KMeans40／20，全局直方图 | 论文写15／9；本轮采用15／9，GT拟合与全局汇总是重建选择 |
| Var | 不是本次8项的来源 | 实际print_metrics拉平帧与系数；另有未用于此处的求和方差helper | 附录写沿时间算方差、再按特征平均；跨片段汇总未写明 |
| FD汇总 | 与本次表达系数FD不是同一入口 | 逐片段算后等权平均 | 附录给公式，但未明确片段平均或全帧汇总；本轮主值继承DIM片段平均 |
| 帧率／时序 | 数据30fps | 预处理30fps；tester中listener从t1、speaker从t0 | 论文视频25fps；本轮25fps同时间戳全片段 |
| 随机采样 | 另一个baseline流程 | tester每条10候选，用GT全56维FD择优 | 官方推理默认10 NFE；NFE不是候选数；Table4未披露十次择优 |
| 测试对象 | 完整数据test43，另有OOD42；挑战赛包另算 | 按audio键写文件，会覆盖部分listener映射；loader仍保留重复行 | 本轮保留独立UUID43条；AF最终有效清单未发布 |

AF Table4标†的RLHG／L2L／DIM是引用旧数字。引用同一数据集与这些数字，不能反推AF自己使用了完整DIM测试程序。其官方main、page两分支、releases、tags及正式补充包均未找到Table4评分代码；相关issue也未找到作者对完整设置的说明。

尤其要区分FD的含义：**原始ViCo论文的FD是Feature Distance，即逐帧L1特征距离，论文表中还乘100；AF Table4的FD是Fréchet分布距离。** 这不是同一个指标换了提取器。证据为`writing/01_RLHG_ViCo_ECCV2022.pdf`第10–11页与ViCo官方issue15。

论文版本已核对：用户本地18页PDF、arXiv v2和CVPR正式8页补充PDF的上述D.5说明一致。正式补充包是PDF、说明页和视频，不含评分脚本。

在**同一批封存系数**上实测：

| 仅改变计算方式 | Exp数值 | Pose数值 |
|---|---:|---:|
| FD逐片段等权平均 | 28.0164 | 0.07472 |
| FD全帧合并一次计算 | 12.9709 | 0.01123 |
| Var逐系数跨全帧平均 | 0.3557 | 0.01136 |
| Var按DIM拉平计算 | 1.6398 | 0.02290 |
| Var每片段时间方差后平均 | 0.06227 | 0.001390 |
| SID论文15／9 | 3.0585 | 2.5630 |
| SID套用DIM40／20 | 3.9699 | 3.7611 |

这些是诊断，不据此选择最接近原表的一行。DIM未改动函数与结构化实现的16个标量核验误差为0。说明算术内核对得上公开代码，**选择哪一种汇总仍是独立问题**。

原始43条有40个唯一audio键。模拟DIM覆盖映射时，43个loader行只对应40个不同pair；再按原30fps长度模拟5–1024帧限制，得到41行／39个不同pair。这个模拟不是作者真实有效清单：实际HuBERT对齐长度未提供。此模拟Exp FD为28.38，单帧时序差异为28.08，均未解释16.64的差距。不将这些历史代码行为强行套给25fps的AF。

来源：[ViCo官方issue15](https://github.com/dc3ea9f/vico_challenge_baseline/issues/15) · [DIM实际评分](https://github.com/Boese0601/Dyadic-Interaction-Modeling/blob/main/code/mymetrics.py) · [DIM指标函数](https://github.com/Boese0601/Dyadic-Interaction-Modeling/blob/main/code/metrics/eval_utils.py) · [AF附录D.5](https://arxiv.org/html/2601.00664v2#A4.SS5)。本地证据：[函数／数据范围交叉计算](METRIC_CROSSCHECK.json) · [完整方差分解](VARIANCE_DECOMPOSITION.json) · [正式补充包目录](SUPPLEMENT_INVENTORY.json)。

## 3. 生成条件：静音波形与关闭音频分支不是一回事

本轮使用speaker音视频＋listener首帧，listener音轨全零。这是明确的listening-only重建假设。论文一般预处理写了分离／分配双方音轨，官方README推荐IIANet／ClearVoice，本轮没有执行这一步；Table4没有写明该测试中listener音轨与guidance的具体处理。

实测零波形经过Wav2Vec2／投影后得到25×512个非零值，RMS约0.269；官方CFG的缺省分支则是全零特征。因此，静音波形＋avatar CFG2仍施加一种条件。

短1秒例子中，关闭该分支使表情时间方差明显增加。但随后按三种态度、两种时长分位固定6条，成对重跑12条视频，结果**不支持这是统一修复**：

| 六条成对测试 | avatar CFG2 | avatar CFG0 |
|---|---:|---:|
| Exp FD | 23.3822 | 23.4630 |
| Exp每片段时间方差平均 | 0.04058 | 0.03440 |
| Exp rPCC | 0.03111 | 0.02360 |
| Pose FD | 0.07459 | 0.07452 |
| Pose rPCC | 0.15802 | 0.13231 |

表情FD三条改善、三条变差。不能因短例子变好，就把CFG0说成论文真正设置。此成对测试使用相同的显式后端状态、权重、输入与噪声种子，是条件消融，不是新的正式Table4复现。

证据：[固定六条与预声明选择](conditioning_ablation/protocol.json) · [成对完整结果](conditioning_ablation/RESULTS.json) · [零波形与原始入口验证](INFERENCE_PARITY.json)。

## 4. 已排查的实现错误与复放异常

- 官方源码预处理与适配器输入最大差异5.96e-8，同进程同种子模型输出差异0（最短例子）。
- SPECTRE原v0.8.1 MobileNet源码与部署骨干加载同一checkpoint后，特征及最终系数差异0（8帧控制）。
- 封存短视频重新提取SPECTRE与原缓存完全一致；645个编码状态严格加载，残差加回及时间上下文检查通过。
- 327个生成核心参数与Wav2Vec加载核验通过，仓库所选源码blob与当前官方一致，未改网络结构。初始化时Wav2Vec旧命名警告随后由显式映射修复，并非还留着两个随机卷积权重。
- 官方README推荐Torch2.0.1／torchvision0.15.2／CUDA11.8，本地生成实际为Torch2.10／torchvision0.25／CUDA12.8、H200。已有权重与源码核验不等于完成原版本端到端数值等价验证；这个环境差异需要披露。官方也未提供将公开pth明确绑定Table4具体划分／DPO版本／评测回执的配置，不能因文件来自官方就推断这些均相同。
- 原43条Exp FD的平均项约23.44、协方差项4.58，差距主要体现在平均表达状态。相同提取器下，生成的片段内时间方差约为GT的59%（Exp）和27%（Pose）。
- 没有缺失脸帧；去掉GT多脸片段后的Exp FD反而为29.17，没有证据表明多脸片段是整体差距主因。
- 一条同配置跨流程复放出现整体亮度变化：`UCA1A5GqCdQ_001213_001223.1`，Exp FD从64.69变到40.09，RGB差异近似全局1.326倍增益。其他五条平均像素差约1–2/255。

人脸检测器会将`cudnn.benchmark`设为True，覆盖guard的初始False。官方导出在截断[-1,1]后按**整段实际min/max再次拉伸**，因此极值变化会作用于全片亮度。

**严格干预已经验证了这条因果链。** 使用原生成脚本、相同缓存输入与权重，在模型前向前只固定`cudnn.benchmark=False`，21段随机噪声的SHA完全相同：

| 原脚本执行条件 | 导出前范围 | 原始均值 | 输出对应关系 |
|---|---|---:|---|
| 检测器初始化后实际benchmark=True | [-1, 0.5080017] | -0.3134209 | 与原封存视频字节／SHA完全一致 |
| 前向前显式固定benchmark=False | [-1, 1] | -0.3132067 | 与简化直调诊断视频字节／SHA完全一致 |

由范围计算的增益为`2/1.5080017=1.326258`，实测三通道增益约1.324–1.327。说明参数／种子相同，仍可能因实际计算路径及导出拉伸得到不同亮度，再影响SPECTRE读出。不能将该现象解释为checkpoint损坏，也不能把简化直调与完整入口当作自动等价。**完整原入口可字节复现，原43条封存成绩未被改写。**

同样，11月7日source-only适配没有初始化这一路人脸检测器，11月10日full-visual适配会初始化；两版的全部差异不能单独归因于新增speaker视觉。后续条件消融必须同时固定实际后端状态。六条成对实验已在两臂相同的显式状态下执行，但其简化调用不作为原入口正式成绩。

同进程先False后True的早期探针没有重现全幅差异，不能单独拿来排除后端影响：算法缓存／初始化路径并不相同。最终结论以完整脚本的上述单变量干预及视频SHA对应关系为准。

新增独立[稳定入口](../../scripts/p2_avatarforcing_table4_generate_stable.py)，在所有人脸预处理之后、每次前向之前重新固定benchmark／deterministic／TF32，逐条保存实际状态与导出范围。保留官方min/max映射，不同时更改导出公式来混淆归因。**该入口已实际跑过异常片段，输出与严格固定False对照逐字节一致。** 验证状态和命令见[稳定入口记录](../p2_avatarforcing_table4_stable_runtime_20261110_v2/README.md)。本入口仅是运行可控性修正，不代表已取得作者私有Table4设置。

证据：[骨干独立核验](BACKBONE_PARITY.json) · [封存／复放像素对照](PAIRED_BASELINE_REPLAY_AUDIT.json) · [亮度仿射拟合](PHOTOMETRIC_AFFINE_FIT.json) · [数值探针](NUMERICAL_PROBE.json) · [True先执行的独立进程探针](NUMERICAL_PROBE_TRUE_FIRST.json) · [原脚本字节复现](EXACT_SOURCE_REPLAY_AUDIT.json) · [严格干预／因果核验](EXPORT_CAUSAL_AUDIT.json)。首次严格干预因空闲显存不足被guard拒绝，日志保留；后续容量恢复后完成，没有停止其他作业。

## 接下来该做什么

1. 新运行应使用已显式固定后端并记录导出范围的独立入口，禁止把简化诊断直调混作原入口正式成绩。当前修正不等于完整43条已重跑，也不意味着Table4数值会匹配；原成绩继续保留。
2. 用原始较宽视野的视频恢复speaker／listener画面，按AF说明处理双路音轨，完成少量与正式补充示例相同片段的验证。两条源YouTube的oEmbed仍可访问，但尚未重新下载原始媒体，见[源地址与可用性](RAW_SOURCE_AVAILABILITY.json)。
3. 分别冻结ViCo官方、DIM和AF三种协议。AF未披露的汇总／测试清单／checkpoint与DPO绑定继续明确标作推断；不靠调公式、删片段或选种子去凑Table4。

无需依赖联系作者才能继续做工程与受控验证。但在公开材料未给全、输入画面也未等价前，不能宣布Table4完全复现。
