# 逐篇实验、输入输出与表格来源核查

日期：2026-11-07。核查本地21篇独立论文的任务、方法条件、实验与相关附录，并补列官方文字补充及图/正文比较。来源未披露与引用推测明确保留，不视为已确认；资源现状与原论文实际使用的checkpoint分开记录。

## GLARE

任务：生成与speaker语音及其韵律相关、类型和时刻适当的视觉倾听反应。

[本地PDF](../../writing/00_GLARE_local.pdf)；输入：speaker audio＋listener reference image；另有previous motion context及固定reference video clip Tref（p16）；训练reaction标注不作推理输入。输出：listener视频，不生成listener audio。

训练：RealTalk＋Seamless整理约147h；9:1划分；p16 baseline650epochs，p17 Table7 final850epochs。

| 表／页 | 测试数据 | 比较方法 | 指标 | 来源判定 |
|---|---|---|---|---|
| 2 / p7,8 | 作者整理RealTalk、Seamless各自测试集 | L2L、DIM、RLHG、ListenFormer、DyStream、GLARE | PSNR/SSIM/FID/FVD16/Var/LPIPS/rPCC/DI-Sync/R-F1/R-tIoU/R-ATD/R-FID | p7明确全部比较方法在其数据上训练和测试；未给各baseline初始化/具体checkpoint |
| 3 / p8 | 同上，分六类反应 | DyStream、GLARE | R-F1/R-tIoU/R-ATD/R-FID | 同本论文训练评价；基线精确输入适配未披露 |

图中／正文中的其他比较：

- Fig3/4（p8,9,20）：RealTalk/Seamless各自作者整理样例；L2L、DIM、RLHG、ListenFormer、DyStream、GLARE。沿正文作者目标数据训练；不是官网截帧直接引用的声明。

证据边界／原文未披露项：反应类别匹配要求tIoU≥0.5；非本文整段类别F1。R-ATD α2/β0.5；R-FID仅反应区间。公开annotations不含精确源切片映射。p16固定同一Tref用于全部推理，来源/身份未披露；不能简化成严格只有首帧的同输入基线。

## ViCo / RLHG

任务：建立ViCo benchmark并以speaker视听信号、listener身份与态度生成倾听头像。

[本地PDF](../../writing/01_RLHG_ViCo_ECCV2022.pdf)；输入：speaker流式音频＋speaker动作＋listener首帧特征＋态度标签。输出：listener动作系数，再以独立renderer可视化；不生成音频。

训练：ViCo Dtrain；Dtest人物见过，Dood人物未见；300 epochs。

| 表／页 | 测试数据 | 比较方法 | 指标 | 来源判定 |
|---|---|---|---|---|
| 3 / p10,11 | ViCo Dtest、Dood | Random、Simulation、Simulation-attitude、RLHG | L1 Feature Distance×100，angle/expression/translation，分态度 | 作者实现自身与非参数控制，无外来学习基线指标引用声明 |
| 4 / p10,11 | ViCo Dtest、Dood | RLHG | FID/SSIM/PSNR/CPBD | 独立零样本renderer评分，不能解释为训练一个新renderer |
| 6 / p13,14 | ViCo Dtest/Dood人工偏好与态度匹配 | RLHG | GT/生成/相当偏好比例；态度识别precision均值与方差 | 作者生成结果与GT人工比较；非外来模型指标引用；每人30偏好与90态度样本 |

图中／正文中的其他比较：

- Fig5（p12,21）：ViCo test/OOD，固定reference与attitude；Random、Simulation、Simulation-attitude、RLHG。作者控制和自身生成，另FigA4仅自身模态消融。

证据边界／原文未披露项：Table5含audio-only/visual-only/wrong-audio/mirroring消融；FD在此为L1，不是Fréchet。

## Learning to Listen / L2L

任务：自回归生成同一speaker输入下多种合理、非确定性的listener面部动作。

[本地PDF](../../writing/02_Learning_to_Listen_CVPR2022.pdf)；输入：speaker动作＋speaker分离音频＋过去listener动作（自回归）。输出：listener DECA/FLAME动作；写实可视化为额外人物renderer，不生成音频。

训练：自建72h六频道数据；按listener分别训练codebook/predictor，70/20/10划分，跨listener模型聚合。

| 表／页 | 测试数据 | 比较方法 | 指标 | 来源判定 |
|---|---|---|---|---|
| 1 / p6,7 | 自建in-the-wild L2L | NN-motion、NN-audio、Random、Median、Mirror、Delayed-Mirror、LFI、Random-Expression、L2L-Random-Walk、L2L | L2/Fréchet FD/Var/SI/P-FD/PCC，expression/rotation | p7明确LFI在自身数据重训；检索、随机和镜像为作者控制；自身训练。未声明直接引用外表 |
| arXiv补充3–6/7 / parxiv13,arxiv14,arxiv15,arxiv16 | L2L四个listener人物各自test与TLCC分析 | NN-motion、NN-audio、Random、Median、Mirror、Delayed-Mirror、LFI、Random-Expression、L2L-Random-Walk、L2L | 主表同套指标；Table7为TLCC lag | 补充明确主表平均四个listener（A/B/C/D），六频道中另两人因guest hosts数据不足排除；LFI在作者数据按原config重训，含双方motion/audio条件 |
| arXiv补充D正文 / parxiv14 | LFI原数据与原train/test划分 | LFI、L2L | FD/P-FD/Var；仅motion设置 | 因LFI只提供facial annotation无原音视频，作者重训LFI和自身motion-only消融m；不是官方LFI权重推理 |

图中／正文中的其他比较：

- Sec5.3正文人工研究（p8）：一位随机listener，人工精选显著时刻后随机50条8s test；150评审；NN-motion、L2L。作者mesh可视化AB；另有a+m消融与GT对照；不同于四listener自动表聚合。

证据边界／原文未披露项：Table2为模型消融；补充已取得17页arXiv版本web/L2L_arxiv.pdf。Fig7显式GT min-L2 best-of1至200是多样性分析，不能据此推断主Table1也用了GT择优。补充另有listener-agnostic模型结果，不能冒充person-specific官方权重。

## PCH / PCHG

任务：以正则driver和增强renderer参加ViCo talking/listening双任务挑战。

[本地PDF](../../writing/PCH.pdf)；输入：驱动audio＋reference image；listening额外attitude。输出：talking/listening视频；不生成目标音频。

训练：ViCo challenge；driver训练，预训练PIRender及背景融合。

| 表／页 | 测试数据 | 比较方法 | 指标 | 来源判定 |
|---|---|---|---|---|
| 1 / p2 | ViCo官方challenge leaderboard | PCH、sysu_hcp、iLearn、Avatar-team、Digital_Human、THU-Talking、cheese、en_train、LIMMC | talking: PSNR/FID/LMD；listening:PSNR/FID/ExpFD | 明确截取官方最终leaderboard；不是作者逐个复现其他参赛队 |
| 2/3 / p4 | 作者ViCo validation | PCH | PSNR/FID/LMD | 模型设计、batch和ensemble消融，主要为talking任务 |

证据边界／原文未披露项：不能把此论文全部消融当作listener实验；leaderboard团队名称应保留，但不是当前应重训的具名公开方法。

## MFR-Net

任务：兼顾反应多样性、listener身份与态度的multi-faceted listening head generation。

[本地PDF](../../writing/06_MFR_Net_ACMMM2023.pdf)；输入：speaker音频＋动作＋listener身份特征＋态度＋latent状态。输出：listener 3DMM＋PIRender视频；无音频生成。

训练：ViCo训练集；renderer以身份loss重训。

| 表／页 | 测试数据 | 比较方法 | 指标 | 来源判定 |
|---|---|---|---|---|
| 1 / p5,6 | ViCo Dtest/Dood | RLHG、PCH、MFR-Net | L1 Feature Distance×100，分态度 | p5明确每种方法在ViCo训练集训练、test/OOD评价 |
| 2/3 / p6,7 | ViCo Dtest/Dood及人工样本 | RLHG、PCH、MFR-Net | SSIM/CPBD/PSNR/FID/CSIM/Diversity；人工naturalness/diversity/identity/attitude | 作者训练评价；具体初始化/核心权重未披露 |

图中／正文中的其他比较：

- Fig3（p6）：ViCo test/OOD固定listener与attitude；RLHG、PCH、MFR-Net。作者目标数据训练与生成，身份renderer按其设置。

证据边界／原文未披露项：CSIM在本表方向及特征定义需保留原文，不能与其他ArcFace余弦表自动同向。Table4为自身消融。

## ELP

任务：用情绪条件下的离散motion空间生成listener表情、头姿与眨眼。

[本地PDF](../../writing/08_Emotional_Listener_Portrait_ICCV2023.pdf)；输入：speaker音频与动作＋预测或指定情绪；参考图用于renderer。输出：listener表情/头动/眨眼＋写实视频；无listener音频生成。

训练：ViCo与L2L分别训练；TalkingHead-1KH训练one-shot renderer。

| 表／页 | 测试数据 | 比较方法 | 指标 | 来源判定 |
|---|---|---|---|---|
| 1上半 / p5,6 | ViCo | NN-motion、NN-audio、Random、DLS-Random、RLHG、PCH、L2L、ELP | FD/V-D/SI-D/RPCC/WTLCC/STS，另blink WTLCC | PCH与L2L明确ViCo重训；PCH基于RLHG重实现。RLHG的两个带* FD单元引用其官方报告，其余格不能一起归为引用 |
| 1下半 / p5,6 | L2L原数据集 | NN-motion、NN-audio、Random、DLS-Random、RLHG、L2L、ELP | 同上，expression/pose分开 | 正文说PCH/RLHG在L2L重训，但实际下半表没有PCH行；L2L带*的FD/V-D/SI-D/RPCC引用L2L官方报告，WTLCC/STS等未带*项来源另记未明确 |
| 2 / p7 | ViCo和L2L分开，latent U/V控制 | NN-motion、NN-audio、DLS-Random、ELP | FD/V-D/SI-D/RPCC/WTLCC/STS/blink WTLCC | 作者在两种latent空间重复控制实验，不是额外开源模型；对应设置重跑 |
| 3 / p8 | L2L，latent分类头大小消融含L2L对照 | L2L、ELP | FD/V-D/SI-D/RPCC/WTLCC/STS/blink WTLCC | ELP各H为作者训练；L2L一行具体权重/是否沿用已报值未逐格声明 |
| 4 / p7,8,9 | 不同情绪场景，42人，数据集逐条范围未披露 | RLHG、PCH、L2L、ELP | 5分量表；表情/pose/blink/overall，Real与Mesh分列 | 作者人工调查；部分qualitative使用官网已有结果并手动对齐，不能默认每段均作者独立推理 |

图中／正文中的其他比较：

- Fig4/5（p7）：ViCo样例及L2L人物mesh/写实结果；RLHG、PCH、L2L、ELP。RLHG/PCH官网样例手动对齐；L2L视觉采用作者另训person-specific Vid2Vid；不能当官方L2L renderer。

证据边界／原文未披露项：FD正文名为Frechet却定义为L1；需按公式/实现解释，不随名称混同。Table2为latent消融。

## CustomListener

任务：通过文本属性与行为先验控制listener响应，并保持与speaker协调。

[本地PDF](../../writing/15_CustomListener_CVPR2024.pdf)；输入：speaker视频动作＋audio/转写相关信息＋自定义listener文本＋reference。输出：listener 70维动作＋PIRender视频；不生成音频。

训练：ViCo、RealTalk重新词转写及文本标注；自身diffusion训练。

| 表／页 | 测试数据 | 比较方法 | 指标 | 来源判定 |
|---|---|---|---|---|
| 1/2 / p6,7 | ViCo Dtest/Dood | RLHG、PCH、L2L、MFR-Net、CustomListener | L1 FD/RTLCC/RWTLCC/FID-delta-motion/SND/V-D；SSIM/CPBD/PSNR/FID | RLHG/PCH/L2L带*，明确重训；MFR-Net带†引用MFR原文Table1/2。ELP不在定量表，因其100维与本文64维expression不兼容 |
| 补充5/6 / psupp2,supp3 | RealTalk Dtest | RLHG、PCH、L2L、CustomListener | L1 FD/RTLCC/RWTLCC/FID-delta-motion/SND/V-D；SSIM/CPBD/PSNR/FID | 补充B明确RLHG/PCH/L2L全部在RealTalk重训；MFR/ELP因无代码没有RealTalk定量。PCH/RLHG态度由自动情绪映射正/负/中性 |
| 4 / p8 | ViCo/RealTalk随机25个speaker视频，28人 | RLHG、PCH、L2L、CustomListener | 1–4分：GT相似/自然/协调/平滑 | 作者在所选视频生成各方法结果并人工评分；生成沿正文目标数据重训，未逐段给权重 |

图中／正文中的其他比较：

- Fig5及官方补充C（p7,8,supp3,supp4,supp5）：ViCo；ELP官网对应speaker样例；RLHG、PCH、ELP、CustomListener。RLHG/PCH作者重训；ELP无代码，采用ELP已展示speaker视频另生成自身结果，不是作者跑ELP。

证据边界／原文未披露项：已取得官方CVF五页supp：web/CustomListener_supp.pdf；A.2披露GPT依据speaker转写与用户属性预测文本；不能把所有文本推理都概括为用GT。ELP视觉对比取其展示源speaker视频后另生成自身结果，并非独立跑ELP。

## DIM

任务：先联合学习speaker/listener motion的dyadic表征，再迁移到listener与speaker动作任务。

[本地PDF](../../writing/17_DIM_ECCV2024.pdf)；输入：listener分支：speaker音频＋speaker动作＋reference；speaker分支：自身speech。输出：EMOCA expression50＋pose6；listener写实视频采用另训renderer；不生成speech。

训练：CANDOR850h预训练；ViCo、LM_Listener下游；BiWi speaker分支。

| 表／页 | 测试数据 | 比较方法 | 指标 | 来源判定 |
|---|---|---|---|---|
| 1 / p9,10,11 | ViCo | Random、NN-audio、NN-motion、ELP、RLHG、L2L、L2L-CANDOR、DIM | Fréchet FD/P-FD/MSE/SID/Var/rPCC（Exp/Pose） | 全部统一EMOCA重训；ELP作者从头重实现。L2L-CANDOR另含预训练，不能与普通L2L混同 |
| 2 / p9,10,11,12 | LM_Listener（Trevor单listener扩展集） | Random、Mirror、NN-motion、ELP、RLHG、L2L、L2L-CANDOR、DIM | 同上 | RLHG/L2L重训；ELP重实现；共享EMOCA，非直接采用原L2L人物权重 |
| 4 / p14,23 | BiWi | MeshTalk、FaceFormer、CodeTalker、DIM-Speaker | LVE/FDD | 明确重训，EMOCA转换FLAME 5023顶点；不同于原先23370顶点协议 |
| 附录2 / p22 | ViCo test中8个约14s例，50位Prolific参与者 | ELP、RLHG、L2L、DIM | 0–5分：正确反馈、动作表情多样平滑；逐视频及平均 | 作者在本论文统一EMOCA/renderer方案下的重训或重实现结果，独立人工研究 |

图中／正文中的其他比较：

- Fig4（p12）：ViCo样例，所有方法共享该pair微调的PIRender；RLHG、ELP、L2L-CANDOR、DIM。图注明确L2L是CANDOR预训练；不能拿Fig4去证明原表普通L2L的视觉结果。

证据边界／原文未披露项：p12/22：每个test人物/配对微调renderer，属于需披露的适配。公开代码的常量speaker/十次GT择优/有效test列表与主表对应未确认，见旧AUDIT。

## ReactFace

任务：在线生成多种适当的面部反应，并以多个可接受真实反应衡量适当性。

[本地PDF](../../writing/26_ReactFace_TVCG2025.pdf)；输入：speaker原始face视频＋speech＋listener参考图（render）。输出：多个适当listener FaceVerse58维动作及视频；不生成listener speech。

训练：REACT2023混合NOXI/RECOLA；1594/562/806，30s，不含UDIVA。

| 表／页 | 测试数据 | 比较方法 | 指标 | 来源判定 |
|---|---|---|---|---|
| 2 / p10,11,12,13 | REACT2023 test | Mirror、Random、NN-motion、NN-audio、Trans-AE、LFI、L2L、RLHG、ReactFace | FRDvs/FRVar/FRDiv/FRRea(FVD)/FRCorr/FRSyn/FPS | p12明确重实现LFI、Ng/L2L、Zhou/RLHG并遵循其训练设置；PIRender重训适配FaceVerse |
| 8 / p15 | REACT2023中8个clip的24个视频对，41人 | L2L、RLHG、ReactFace | AB偏好：Realism/Diversity/Appropriateness/Sync，另与GT比较 | 作者人工调查，Ng et al为L2L、Zhou et al为RLHG；不是引用原方法已有人工分数 |
| 9 / p12,15 | RLD / ViCo，483视频、67speaker/76listener，具体有效test列表未披露 | Trans-AE、L2L、RLHG、ReactFace | FRRea/FRVar/FRDiv/FRSyn；不报告FRCorr | 作者额外训练ReactFace，用普通Lrec替代缺失的appropriate邻居损失Larec；遵循RLHG原3DMM系统。其他行逐项checkpoint/训练细节未独立披露，不能将主REACT表状态机械套用 |

图中／正文中的其他比较：

- Fig7（p11,13）：REACT2023；L2L、RLHG、ReactFace。作者重实现对应生成；同其主比较的FaceVerse/PIRender适配。

证据边界／原文未披露项：页码为18页版本；原文件现已有效，Table2/8/9与作者替代版已重新核对。Table1仅符号表。已补第15页RLD Table9与人工Table8；RLD改标准重建损失并沿原RLHG 3DMM体系，不能混同REACT主表FaceVerse58。多适当邻居协议不同于单GT。

## DiffListener

任务：结合显式运动与隐式细节修正生成高分辨率、逼真的listener视频。

[本地PDF](../../writing/34_DiffListener_CVPR2025.pdf)；输入：speaker audio＋动作及listener reference。输出：显式MediaPipe运动＋扩散写实listener视频；不生成音频。

训练：ViCo、RealTalk；VFHQ预训练renderer；512px。

| 表／页 | 测试数据 | 比较方法 | 指标 | 来源判定 |
|---|---|---|---|---|
| 1 / p6,7 | RealTalk：train1489/test407 | RLHG、PCH、L2L、DiffListener | SSIM/PSNR/FID/L1 FD/FID-delta-motion/SND | RLHG/PCH/L2L明确重训 |
| 2 / p6,7 | ViCo：train398/test43/OOD42 | RLHG、PCH、L2L、MFR-Net、CustomListener、DiffListener | 同上 | RLHG/PCH/L2L重训；MFR-Net与CustomListener带*直接引用其论文；正文明确CustomListener带GT运动文字条件 |
| 官方补充6 / pcvfsupp3,cvfsupp4 | RealTalk随机20个speaker视频；26人 | RLHG、PCH、L2L、DiffListener | 1–5分：VisualQuality/Vividness/Naturalness/SimilarityToGT | 作者重训后生成并人工打分；A.3明确RLHG/PCH均去掉emotion标签；不是继承他人人工结果 |

图中／正文中的其他比较：

- Fig2/补充Fig6（p5,cvfsupp2,cvfsupp3）：RealTalk/ViCo比较样例，逐样例数据集未全部标出；RLHG、L2L、PCH、DiffListener。作者重训；RLHG/PCH在该比较中不使用emotion标签。

证据边界／原文未披露项：内部运动57维不代表评价也是该空间；评价重新提取统一3DMM。官方补充A.3明确RLHG/PCH在ViCo和RealTalk均去掉emotion标签重训，与含attitude的原始设置不同；其Table6为额外人工比较。两个本地DiffListener文件指纹相同，只计一篇。

## INFP

任务：由已知双方音轨统一驱动头像说话、倾听及角色转换。

[本地PDF](../../writing/35_INFP_CVPR2025.pdf)；输入：双方已知音轨＋reference；可选额外motion style视频。输出：interactive头像视频，不生成双方speech。

训练：自建DyConv训练motion latent和generator。

| 表／页 | 测试数据 | 比较方法 | 指标 | 来源判定 |
|---|---|---|---|---|
| 1/2 / p6,7,8 | DyConv test | DIM、INFP | SSIM/PSNR/FID/SyncScore/LPIPS/CSIM/SID/Var；MOS | DIM用官方代码在DyConv重训，speaker/listener分支及renderer另训 |
| 3 / p7,8 | ViCo | L2L、RLHG、DIM、INFP | Fréchet FD/MSE/SID/Var | RLHG/L2L/DIM明确继承DIM Table1；INFP自身直接迁移不额外修改；旧报告误称此为Table2，应为Table3 |
| 4 / p7,8 | HDTF随机20视频 | SadTalker、AniTalker、EchoMimic、INFP | SSIM/PSNR/FID/SyncScore/CSIM | 作者在选定20条上比较；各外来方法具体checkpoint/是否再训练未明确 |

图中／正文中的其他比较：

- Fig4（p6）：DyConv；DIM、INFP。DIM官方代码在DyConv重训；作者自身交互结果。
- 官方补充Fig4（pcvfsupp1,cvfsupp2）：ViCo listener与HDTF talking样例；L2L、RLHG、DIM、SadTalker、AniTalker、EchoMimic、INFP。listener比较沿正文继承DIM结果；自身直接迁移；talking具体checkpoint未报告。

证据边界／原文未披露项：正文说core model访问受限；公开网站与官方生成权重不是一回事。

## ListenFormer

任务：以非自回归Transformer生成倾听动作，缓解误差累积和长序列解码问题。

[本地PDF](../../writing/ListenFormer_ Responsive Listening Head Generation with Non-autoregressive Transformers.pdf)；输入：speaker音频＋speaker motion＋listener reference。输出：listener动作与PIRender视频；无speech生成。

训练：ViCo与L2L分别训练；ViCo70维，L2L DECA56维；不在L2L报告图像指标。

| 表／页 | 测试数据 | 比较方法 | 指标 | 来源判定 |
|---|---|---|---|---|
| 1/2 / p5,6 | ViCo Dtest/Dood | RLHG、PCH、DSPN、MFR-Net、ListenFormer | L1×100；SSIM/CPBD/PSNR/FID/CSIM | 表注MFR*明确引用MFR原文，其余比较方法在作者系统复现；这两表没有L2L行 |
| 3 / p5,6 | L2L | RLHG、DSPN、L2L、ListenFormer | expression/angle的L1与Fréchet FD | 未给逐行权重/人物聚合范围；不得推断是官方Trevor模型或将ViCo表注自动扩展到此表 |
| 4 / p7 | ViCo人工30视频 | RLHG、PCH、DSPN、ListenFormer | naturalness/diversity/identity/synchrony/attitude | 作者生成与15人用户研究 |

图中／正文中的其他比较：

- Fig5/Sec4.3（p7）：ViCo；RLHG、PCH、DSPN、ListenFormer。正文描述与表中作者系统复现一致，定性不含MFR独立推理证据。

证据边界／原文未披露项：DSPN全名Responsive Listening Head Synthesis with 3DMM and Dual-Stream Prediction Network。Table5后为自身消融。

## SpeakerVid-5M

任务：构建视听人物/对话数据集与VidChatBench，并训练联合AV自回归回应baseline。

[本地PDF](../../writing/38_SpeakerVid_5M_2025.pdf)；输入：dyadic: initiator音视频＋reference；conditioned: GT ASR＋描述caption。输出：responder音视频；需要reference声音支持声纹控制。

训练：SpeakerVid-5M预训练／joint AV／高质量SFT。

| 表／页 | 测试数据 | 比较方法 | 指标 | 来源判定 |
|---|---|---|---|---|
| 2 / p8,9,10 | VidChatBench 500对未见speaker ID | SpeakerVid-5M | FID/FVD/PSNR/SSIM/ArcFace/CLIPdialog/Syncconf/FIDEmotion/SIM-o | 全部为作者自身conditioned/dyadic和audio/spatial/noise变体，无外来方法比较行 |

证据边界／原文未披露项：不能把数据集比较Table1当作方法性能表；CLIPdialog依赖5条候选文本及排序，需取原候选才能复现。

## ARIG

任务：在长程上下文和对话状态条件下逐帧生成交互头像motion。

[本地PDF](../../writing/40_ARIG_ICCV2025.pdf)；输入：user音视频＋avatar已知音频＋reference＋上下文状态。输出：avatar视频，不生成自身speech。

训练：RealTalk训练，迁移single-role。

| 表／页 | 测试数据 | 比较方法 | 指标 | 来源判定 |
|---|---|---|---|---|
| 1 / p6,7 | RealTalk | DIM、ARIG | rPCC/CSIM/SyncScore/PSNR/SSIM/FID/SID/Var | DIM明确在作者数据重训 |
| 2 / p7,8 | ViCo | RLHG、L2L、DIM、INFP、ARIG | Fréchet FD/rPCC/SID/Var | RLHG/L2L/DIM/INFP带*，明确继承DIM与INFP；ARIG生成视频后70维3DMM，兼容性需单列 |
| 3 / p7 | HDTF | SadTalker、Hallo、EchoMimic、ARIG | PSNR/SSIM/FID/CSIM/SyncScore | 作者比较；具体外来权重和重训来源未披露 |
| 官方补充2 / pcvfsupp2 | RealTalk随机20视频，25人 | DIM、ARIG | 1–5分：自然／协调／多样／lip同步 | 作者以其RealTalk重训DIM生成比较并开展人工调查；补充A.2区分内部motion262维与主文反提评价70维 |

图中／正文中的其他比较：

- Fig5（p6,7）：DyConv官网两条demo；DIM、INFP、ARIG。INFP直接取其官网结果；缺user视频，先据user音轨合成user再驱动ARIG；与真实user输入不同。
- 官方补充Fig5/6/7（pcvfsupp2,cvfsupp3,cvfsupp4,cvfsupp5）：RealTalk/HDTF/ViCo分开；DIM、SadTalker、Hallo、EchoMimic、RLHG、L2L、INFP、ARIG。DIM在RealTalk作者重训；ViCo旧行沿正文引用链，未提供每样例checkpoint。

证据边界／原文未披露项：INFP定性使用官网两条demo，缺user GT视频时先合成user视频，不能写成独立官方INFP推理。

## DiTaiListener

任务：以speaker视听及可选文本控制生成高保真listener视频，并平滑长视频段间过渡。

[本地PDF](../../writing/41_DiTaiListener_ICCV2025.pdf)；输入：speaker音频＋motion＋reference＋可选listener描述文本。输出：listener视频，含段间Edit；无speech生成。

训练：RealTalk基于EasyAnimateV5.1-12B训练20k；ViCo5k；Gemini＋Llama标注。

| 表／页 | 测试数据 | 比较方法 | 指标 | 来源判定 |
|---|---|---|---|---|
| 2/3 / p5,6,7 | ViCo / RealTalk | RLHG、L2L、ELP、DIM、DiTaiListener | FID/FVD/LPIPS/PSNR/SSIM | ViCo有ELP，RealTalk无ELP；正文称测试baseline，但未逐项给出训练／checkpoint；不得仅由数值推断重训 |
| 4 / p8 | ViCo EMOCA | ELP、RLHG、L2L、DIM、INFP、DiTaiListener | Fréchet FD/P-FD/MSE/SID/Var | INFP†明确直接取INFP Table3；ELP/RLHG/L2L/DIM数值与DIM Table1一致但无同等明确†声明，登记为数值匹配线索 |
| 5 / p8 | RealTalk EMOCA | RLHG、L2L、DIM、DiTaiListener | 同上 | 无直接引用标记；具体baseline来源未明确 |
| 1 / p5,14 | 50份有效人工答卷，样例见附录 | ELP、RLHG、L2L、DIM、DiTaiListener | Feedback/Diversity/Smoothness/Overall多选一偏好比例 | 作者匿名随机顺序人工调查；样本生成沿其主比较设置，逐条权重来源仍未披露 |
| CVF补充6 / pcvfsupp2 | ViCo长于4s的子集 | Prompt-Travel、Teacher-Forcing、DiTaiListener | FlowConsistency/MotionHistoryEnergy/S3/NIQE | 作者比较长视频拼接/递推策略，不是三个独立预训练生成器；w/o Edit为自身消融。此表在新取得CVF3页supp内，初始本地15页版本没有 |

图中／正文中的其他比较：

- Fig3/5（p6,7,8）：ViCo普通/长视频样例；RLHG、L2L、ELP、DIM、Prompt-Travel、Teacher-Forcing、DiTaiListener。旧3DMM结果由PIRender按其协议渲染；两种长视频策略与自方法比较，不是额外公开完整模型。

证据边界／原文未披露项：Table1人工含ELP/RLHG/L2L/DIM；附录14–15页披露过滤及caption。文本来自真实listener观察，默认文本条件是否推理可得需说明；有w/o text消融。

## Avatar Forcing

任务：以因果diffusion forcing和偏好优化生成实时、表达性更强的交互头像。

[本地PDF](../../writing/43_Avatar_Forcing_CVPR2026.pdf)；输入：user音频/视频动作＋avatar预定音轨＋reference。输出：avatar视频，无speech生成。

训练：RealTalk训练motion AE及diffusion forcing；DPO。

| 表／页 | 测试数据 | 比较方法 | 指标 | 来源判定 |
|---|---|---|---|---|
| 1/2 / p6,14 | RealTalk及人工 | FLOAT、INFP-reimpl-AvatarForcing、Avatar-Forcing | latency/rPCC/SID/Var/FID/FVD/CSIM/LSE-D/C；人工preference | INFP*作者在FLOAT运动空间重实现并训练；FLOAT非交互参考；未把它标成官方INFP |
| 3 / p6 | HDTF随机50视频 | SadTalker、Hallo3、FLOAT、INFP-reimpl-AvatarForcing、Avatar-Forcing | FID/FVD/CSIM/LSE-D/C | INFP*同上；其余baseline初始化来源未明确 |
| 4 / p6,8 | ViCo | RLHG、L2L、DIM、INFP-reimpl-AvatarForcing、Avatar-Forcing | Fréchet FD/rPCC/SID/Var | RLHG/L2L/DIM†明确直接引用DIM Table1；自身与INFP*为作者实验 |

图中／正文中的其他比较：

- Fig6/14/15（p7,16,17,18）：RealTalk交互/HDTF talking/ViCo listening分开；INFP-reimpl-AvatarForcing、SadTalker、Hallo3、FLOAT、RLHG、L2L、DIM、Avatar-Forcing。INFP*为作者FLOAT空间重实现；旧listener结果来源沿正文DIM引用，未逐帧给来源。

证据边界／原文未披露项：附录14页SPECTRE expression50＋pose6；不仅说明维数，还需核对提取器与EMOCA兼容；latency排除了部分音频处理。

## ECHO

任务：通过长程语境和角色解耦调制改善已知双方音轨下的情绪适当交互头像。

[本地PDF](../../writing/45_ECHO_Interactive_Head_2026.pdf)；输入：双音轨（含avatar预定speech）＋user视觉历史＋reference；上下文理解。输出：avatar视频，不生成speech。

训练：Seamless300h+；LoRA；16 A100。

| 表／页 | 测试数据 | 比较方法 | 指标 | 来源判定 |
|---|---|---|---|---|
| 1 / p10,11,12 | RealTalk IHG | DIM、INFP、ARIG、Avatar-Forcing、ECHO | rPCC/SID/Var/FID/FVD/CSIM/LSE-D/C | 多项与ARIG/AvatarForcing原表完全一致，主文未明确逐行重跑/引用；INFP数值匹配AvatarForcing INFP*，不能擅称官方 |
| 2 / p13 | HDTF随机50视频 | Hallo、SadTalker、Hallo3、INFP-reimpl-AvatarForcing、ARIG、Avatar-Forcing、ECHO | FID/FVD/CSIM/LSE-D/C | INFP*明确来自AvatarForcing；其他行需保留未披露来源 |
| 3 / p14 | ViCo | ARIG、RLHG、L2L、DIM、DualTalk、INFP-reimpl-AvatarForcing、Avatar-Forcing、ECHO | Fréchet FD/rPCC/SID/Var/MSE | INFP*来源AvatarForcing；ARIG†注明76维3DMM，其余56维FLAME，仅供参考；其他行多与既有表匹配但无明确引用声明 |
| 4 / p13,14 | RealTalk情绪；50样本25人 | DIM、DualTalk、ECHO | Qwen3-Omni E-Consis / human E-Appro | 作者自动及人工评测，需核对外部baseline样本来源 |

图中／正文中的其他比较：

- Fig3（p11,12）：RealTalk两条交互样例；Per-FLOAT、DIM、DualTalk、ECHO。已渲染原PDF图片核对行标签：Per-FLOAT、DIM、DualTalk；不是AvatarForcing。Per-FLOAT未解释具体实现，图内[51]/[37]/[31]编号与当前参考文献不一致，保留标签不猜映射。
- Fig5（p13,17）：DyConv官网两条交互样例；INFP、ARIG、StreamAvatar、ECHO。图注明确各旧方法当时未开源，沿官网视觉比较；没有StreamAvatar定量行，不能捏造其ViCo分数。

证据边界／原文未披露项：DyConv及Seamless还有定性展示；不把定性示例当新增同口径定量测试集。

## OmniResponse

任务：在完整对话流中在线自主生成时间同步的文本、声音与视觉回应。

[本地PDF](../../writing/50_OmniResponse_NeurIPS2025.pdf)；输入：speaker当前/历史音视频与utterances、过去双方facial/context。输出：listener文本＋音频＋facial/video，同步在线生成。

训练：ResponseNet；Phi3.5＋Chrono-Text＋TempoVoice，LoRA。

| 表／页 | 测试数据 | 比较方法 | 指标 | 来源判定 |
|---|---|---|---|---|
| 2 / p7,8 | ResponseNet官方test长对话 | GPT-4o、GPT-4、GPT-o1、Qwen-7B-Chat、Claude-Sonnet-4、Gemini-2.5-Flash、DeepSeek-R1、Moshi、ReactFace、RLHG、OmniResponse-LSTM、OmniResponse-AV-LLM、OmniResponse | METEOR/BERTScoreF1/ROUGE-L/Distinct2；LSE-D/UTMOSv2；facial FD/FVD | 作者构建LSTM/AV-LLM；调用各单模态系统做比较，但主文未明确所有baseline的重训/权重/调用prompt，不能补写 |
| 4 / p10 | ResponseNet，49人，每人16个随机顺序clip | OmniResponse-LSTM、OmniResponse-AV-LLM、OmniResponse | AB偏好：SpeechContent/AudioQuality/VisualQuality/AVSync | 作者自行构建两个baseline并开展人工研究；不能复制到本地短pair协议；正文最低偏好类别与表格有笔误，应以原表明确数值为准 |

证据边界／原文未披露项：官方696长对话并非J短pair586，也不是历史200筛选；task含整轮回复，不能直接引用这些数字到短backchannel表。p8明确Claude-Sonnet-4版本2025-05-14、DeepSeek-R1版本2025-05-28；其他baseline训练/调用细节未逐项披露。

## DyStream

任务：统一说话/倾听角色的高效流式视频生成；音轨作为条件而非生成输出。

[本地PDF](../../writing/DyStream.pdf)；输入：双方已知audio＋reference；音频lookahead按online/offline区分。输出：speaker/listener视频，不生成audio。

训练：HDTF＋RealTalk及内部数据；AE另有内部数据；生成器只公开来源数据训练。

| 表／页 | 测试数据 | 比较方法 | 指标 | 来源判定 |
|---|---|---|---|---|
| 1 / p5,6 | HDTF100；正文另称内部100统一测试池 | Hallo3、Sonic、SadTalker、Real3DPortrait、INFP-reimpl-DyStream、INFP-I、DyStream | Sync-C/content-similarity/quality/dynamic/latency | INFP*作者重实现，INFP-I对其改训练与推理的inpainting变体；其他方法具体checkpoint未逐项说明 |
| 2 / p6 | RealTalk test38条，3–14s | INFP-reimpl-DyStream、DyStream | MediaPipe FD/MSE/SID/Var；k40 expression/k20 pose | 与自己重实现的INFP比较，不是官方INFP或从DIM继承；不能混EMOCA表 |
| 4 / p5,8 | H200 online推理性能，精确测速test清单未披露 | INFP-reimpl-DyStream、DyStream | Step/FPS/AudioPacketDelay/Sync-C | INFP虽表中未加星，但sec4.3明确为作者reproduced版；作者测速不同steps，非原INFP官方权重速度 |

证据边界／原文未披露项：Sec4.3–4.5默认offline一次编码可看未来；online在4.6另评，不因名称Streaming把主表都写为严格因果。

## OmniHuman-1.5

任务：从给定语音/图像通过agentic语义规划生成动作合理的人物动画。

[本地PDF](../../writing/OmniHuman-1.5.pdf)；输入：人物图像＋给定voice track，可选text/多人物mask。输出：speech驱动动画视频；不自主生成speech。

训练：15000h、其中70%音频dropout；高质量100h微调。

| 表／页 | 测试数据 | 比较方法 | 指标 | 来源判定 |
|---|---|---|---|---|
| 1/2 / p8,9,10 | 自建single-subject150例 | OmniHuman-1、OmniHuman-1.5 | IQA/ASE/Sync及HKC/HKV；人工GSB与瑕疵率 | 主要为自方法消融和与OmniHuman1比较，基线权重细节未披露 |
| 3 / p10,11 | 自建multi-subject57例 | InterActHuman、OmniHuman-1.5 | IQA/ASE/Sync-D/HKC/HKV；人工DrivingAccuracy/GSB | 作者实验，无逐项重训声明 |
| 4左 / p8,12 | CelebV-HQ100 | SadTalker、Hallo、EchoMimic、Loopy、Hallo3、OmniHuman-1、OmniHuman-1.5 | IQA/ASE/Sync-C/FID/FVD | 遵循既有设置，不等于明确使用官方权重或重训，来源未披露 |
| 4右/Fig5 / p8,12 | CyberHost269视频119身份 | SkyReels-A1、FantasyTalking、OmniAvatar、MultiTalk、OmniHuman-1、OmniHuman-1.5 | 同上＋HKC/HKV、Top1；匿名CS-A/B/C/D另有GSB | 作者实验；四个商业系统匿名无法映射到公开名称，禁止猜测 |

图中／正文中的其他比较：

- Fig7（p13,14）：多种语义动作样例；OmniHuman-1、OmniHuman-1.5。作者视觉比较；额外reflection仅定性展示，定量时关闭。

证据边界／原文未披露项：这是语音驱动动画背景比较，不是speaker音频条件下独立listener视听反馈；商业匿名不计为可获取明确方法。

## OmniPersona

任务：利用多轮视听历史和persona参考生成身份持续、语境相关的自主联合AV对话。

[本地PDF](../../writing/OmniPerson.pdf)；输入：双方多轮音视频历史＋listener reference image＋4–6s reference voice。输出：avatar audio/video及文本回复；多轮历史与persona条件。

训练：Seamless训练；Qwen2.5Omni＋OVI1.1；48k steps。

| 表／页 | 测试数据 | 比较方法 | 指标 | 来源判定 |
|---|---|---|---|---|
| 2 / p7,18,19 | SpeakerVid-5M / VidChatBench协议 | SpeakerVid-5M、Qwen2.5-Omni+OVI、OmniPersona | FID/FVD/PSNR/SSIM/ArcFace/CLIPdiag/Syncconf/FIDemo/SIM-o | SpeakerVid行与SpeakerVid原文Table2最终dyadic行一致；未明确声明引用/再跑；Qwen+OVI为作者级联基线 |
| 3 / p7,8 | OmniResponse原协议 | OmniResponse-LSTM、OmniResponse-AV-LLM、OmniResponse、OmniPersona | METEOR/BERTScoreF1/ROUGE-L/Distinct2/LSE-D/UTMOSv2/FVD | 三baseline与OmniResponse Table2逐项一致，但未找到明确引用声明，数值匹配只作为线索 |
| 4/5 / p7,8,17 | Seamless自选60条约50s；人工20对20人 | F5-TTS、LiveAvatar、OmniAvatar、OmniPersona | FVD/ArcFace/SIM-o/FIDemo/FD；人工EA/AVS/OQ | 分音频和视频评价；是否官方权重/微调设置主文未逐项明确，不能以OVI可下载代替OmniPersona已发布 |
| 13 / p18,19 | SpeakerVid-5M，同主表集合及reference，统一SpeakerVid生成的response text | SpeakerVid-5M、Qwen2.5-Omni+OVI、OmniPersona | FVD/ArcFace/Syncconf/FIDemo | 作者同response文本控制实验；不等于共享generator/训练过程；SpeakerVid具体模型来源未披露 |
| 15 / p19 | Seamless官方test自动Anger/arousal筛出的20条 | LiveAvatar、OmniPersona | FIDemo/Expression FD；另自身w/o MQ | 作者比较，自动情绪筛选不是人工真值；仅适用于这20条高怒样本，不外推全部情绪 |
| 16 / p19,20 | 单H200，5s/125帧448px完整pipeline | OVI、OmniAvatar、LiveAvatar、OmniPersona | 总时间/峰值显存 | 作者测速含各方法condition/prompt及VAE；OmniPersona另含MLLM，亦单列移除MLLM时间；不是流式首帧延迟 |

图中／正文中的其他比较：

- Fig5/6及附录A（p8,9,15）：SpeakerVid/Seamless，另提OmniResponse展示；SpeakerVid-5M、Qwen2.5-Omni+OVI、LiveAvatar、OmniAvatar、OmniPersona。同主比较方法与消融；附录提及concurrent-work视频但未命名，不能猜成某个模型。

证据边界／原文未披露项：匿名ICLR2027在审版本，不按文件名当OmniPerson独立模型。附录Table13同response text控制；Table10/11是线性probe而非独立方法。

