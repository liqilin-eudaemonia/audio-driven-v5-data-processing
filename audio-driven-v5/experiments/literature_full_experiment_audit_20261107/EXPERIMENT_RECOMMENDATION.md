# 当前论文应比较什么、怎样实际执行

依据2026-11-07当前中文初稿和三组评测确认版。以下是实验建议，**没有启动新实验，也没有冻结测试名单、事件检测阈值或人工量表**。

## 方法总数的含义

当前68组比较表记录、20组图／正文比较中出现 **85个名称**：24个核心任务相关候选、31个外围方法／基础模型／级联条目，以及30个控制／团队／论文内变体／未明确身份图中标签。完整逐项记录在 [资源矩阵](RESOURCE_MATRIX.md)，不是只列最常见的几个方法。

24个核心候选为：

- 视觉listener 14个：RLHG、PCH、L2L、MFR-Net、ELP、CustomListener、DIM、ReactFace、DiffListener、ListenFormer、DiTaiListener、GLARE、DSPN、LFI。
- 已知自身音轨的交互视频7个：INFP、ARIG、Avatar Forcing、ECHO、DyStream、DualTalk、StreamAvatar。
- 自主联合音视频3个：SpeakerVid-5M、OmniResponse、OmniPersona。

外围31个的完整名单也不能删除，但通常不应全部塞入listener主表：17个目标语音驱动动画、3个独立3D说话模型、Moshi/F5-TTS两个音频方法、7个文本模型、Qwen2.5-Omni＋OVI级联及独立OVI。DIM-Speaker是DIM的分支，单列在其余30项中；原论文另有Prompt Travel／Teacher Forcing两种长视频策略和未明确身份的Per-FLOAT图中标签。匿名商业CS-A/B/C/D及general Base Model没有身份，单独说明而不猜模型名称。表格中的自身组件／参数消融也保留在论文说明中，不把每个参数配置计成新模型。

这里的范围为21篇独立PDF及列出的官方文字补充。全部编号表的角色见[TABLE_CENSUS](TABLE_CENSUS.md)，来源未披露、视频未命名或图中身份冲突都明确保留，不靠计数猜补。

## 24个核心候选目前如何落地

| 当前资源层次 | 方法 | 下一步与可写结论 |
|---|---|---|
| 官方核心模型有入口：4个 | L2L、DyStream、Avatar Forcing、DualTalk | 可以进入适配／推理验收；只有L2L和DyStream有本地历史运行证据。L2L和DualTalk主要输出动作，不能冒充可直接统一写实视频。Avatar Forcing已核实官方Drive核心入口，但尚未实际下载验证本次模型。 |
| 有官方训练路径，需重训／恢复资源：6个 | RLHG、DIM、ReactFace、ListenFormer、GLARE、LFI | 不能统称“只能引用”。ListenFormer有历史作者权重链接但用户此前确认过期；其余未核实核心checkpoint下载。需要预处理适配、训练和独立推理验收。 |
| 部分模块，需补全实现：1个 | PCH | 官方不是完整独立系统，不能与完整train/test仓库等同。可以按论文重实现，明确实现者与差异。 |
| 目前未获得可执行原生成器：13个 | MFR-Net、ELP、CustomListener、DiffListener、DiTaiListener、DSPN、INFP、ARIG、ECHO、StreamAvatar、SpeakerVid-5M、OmniResponse、OmniPersona | 当前使用原协议文献参考，或投入明确标记的重实现；不能在自己的J测试集旁复制其旧分数。SpeakerVid已有处理和部分评分工具；ECHO已找到官方占位仓库；OmniResponse模型仍TODO；这些都不等于原生成器可用。 |

本表描述当前证据和资源，不是声称每个仓库已被重新运行。官方链接、issue及限制逐项在资源矩阵中给出。

## 建议的实际实验矩阵

| 位置 | 方法／设置 | 输入与输出 | 实际评价内容 | 尚需完成 |
|---|---|---|---|---|
| 共同输入联合AV主表 | 本文完整模型＋三组件消融；另加Qwen2.5-Omni→OVI | 当前speaker audio＋listener首帧→listener AV；级联由预测回复／caption驱动OVI | 固定同一pair下的视觉／声音质量，整段事件P/R/Macro-F1，语境适当性与参与感，AV协调 | 级联需适配“允许不发声／短反馈”、时长／对齐／分段策略；固定模型和prompt，不提供GT事件、GT caption或GT回应音轨 |
| 视觉listener表，按条件分组 | GLARE重训；DyStream适配；Avatar Forcing适配 | GLARE含固定Tref及previous motion，需披露来源；DyStream other为speaker/self为零；Avatar Forcing user视觉为零、self audio为零 | 视觉质量、视觉事件与适用人工项；音频生成列N/A | GLARE训练完成权重、Tref来源／初始化与数据适配；两种self-zero推理实际验收；保留模型原本的额外条件说明 |
| 原生额外输入的视觉／motion补充 | L2L、DualTalk；资源允许再补RLHG、DIM、ReactFace、ListenFormer、LFI/PCH | 按原方法保留speaker motion、attitude、历史等；DualTalk可能双音轨＋partner motion | 共同视频指标仅限有已验收renderer的方法；纯动作另表 | 不能把输入置零后仍称完整原生模型；人物权重迁移、重训、表示和renderer差异分别披露 |
| 已知目标声音的动画补充 | INFP/ARIG/ECHO文献；DyStream/Avatar Forcing可实测；外围动画按实验目的挑选 | 额外给真实listener audio时，只检验已知语音驱动动画 | 视觉、目标audio条件下的同步；不是自主声音反馈质量 | 与主表分组；自适应audio输出不存在时记N/A，不能以GT音轨作为其生成结果 |
| 原协议文献参考 | 24候选中拿不到原生成器者及完整外围名单 | 各原论文输入／数据／训练设置 | 原论文原表，标清重训／引用／重实现／未披露 | 按 [协议映射](PROTOCOL_MAP.md)引用，无法对齐不计算本文相对提升 |
| 组件基础模型对照 | 独立OVI＋本文去Feature／去Event等 | OVI需要预测／固定文本而非直接speaker理解；消融共享明确训练预算 | 验证现有生成器用于任务的缺口与三个组件的作用 | 选择共享文本或端到端两种问题时需分开解释，不能用不同条件冒充单组件因果证据 |

以上优先级来自任务匹配与当前资源的综合判断，不是相关论文已经证明的排名。主表不宜因“可下载”就选满所有目标audio驱动说话模型；这些方法可用于renderer或已知目标语音补充，但不能独立回答本文的backchannel决策问题。

## 如果只能引用，优先选哪一套数字

不存在一套能够覆盖全部模型且与本文J短pair自动可比的现成表。建议按问题选择原协议：

- **ViCo统一EMOCA动作表**：RLHG、L2L、ELP重实现优先追溯DIM Table1；INFP用自身Table3。ARIG和Avatar Forcing引用的旧行追到DIM/INFP，不能当多个独立复现实验。DIM主表精确test、输入和GT候选择优仍有公开证据缺口，不能称已经严格复现。
- **ViCo BFM/L1系数＋视频表**：RLHG原文、MFR-Net、CustomListener、DiffListener和ListenFormer分别保留其原表。MFR数值被Custom/Diff/ListenFormer沿用时，应引MFR结果源头；文本条件、renderer、test/OOD聚合与距离尺度必须一致。
- **RealTalk视觉listener表**：CustomListener官方补充Table5/6、DiffListener Table1、DiTaiListener Table3/5各有自身处理；GLARE Table2是GLARE重新整理后的训练／测试片段。这些不是一个通用“RealTalk test”。
- **REACT多适当反应表**：ReactFace Table2是REACT2023混合数据的邻居协议；RLD另看Table9，后者缺appropriateness邻居真值并换损失／3DMM体系，不能把两表合在一起。
- **短语音自主AV的文献背景**：SpeakerVid Table2是VidChatBench；OmniResponse Table2是长对话；OmniPersona Table2/3/4分别属于三个协议。我们的586/200 ResponseNet J短pair没有资格直接继承长对话表的分数。
- **交互视频与目标audio动画表**：Avatar Forcing Table1/3/4、DyStream Table1/2与ECHO Table1–4分别保留。INFP原版、Avatar Forcing重实现INFP*、DyStream重实现INFP*是三个不同身份。

每个方法对应的精确表号、页码、特征表示和缺口见PROTOCOL_MAP；“数值相同但作者未说明来源”保持该状态，不升级成确认引用或确认官方checkpoint推理。

## 可以复用哪些评测工具

| 入口 | 可复用部分 | 必须保留的限制 |
|---|---|---|
| [本地统一listener评测](../../../OmniResponse/evaluation/listener_metrics/README.md) | 已有配对视频、适用音频指标与覆盖率；可作为共同外部评测基础 | 本地整合工具不是OmniResponse作者原表实现；短回应UTMOS不适用时报告N/A |
| [GLARE官方eval](https://github.com/lzk901372/glare/tree/main/eval) | 六类reaction的R-F1/R-tIoU/R-ATD/R-FID | 需要预先检测／平滑的CSV，官方代码不是完整检测器；事件一对一且tIoU≥0.5，R-ATD实现×100，不是本文整段出现F1 |
| [ViCo官方evaluation](https://github.com/dc3ea9f/vico_challenge_baseline/tree/main/evaluation) | 图像/身份/清晰度等及renderer适配 | FD在早期论文指L1距离；依赖权重和特征实现须锁定 |
| [DIM官方代码](https://github.com/Boese0601/Dyadic-Interaction-Modeling) | EMOCA FD/P-FD/MSE/SID/Var/rPCC实现可参考 | 默认输入与十次GT择优已有审计问题；不能只复制默认tester就声称对齐原表 |
| [ReactFace官方metric](https://github.com/lingjivoo/ReactFace) | 多适当反应邻居、DTW/TLCC与多样性 | 要取到原GT邻居/FaceVerse；与单GT事件分类是不同问题 |
| [SpeakerVid官方metrics](https://github.com/Dorniwang/SpeakerVid-5M-Code/tree/main/metrics) | FID/FVD/ArcFace/情绪/Sync/SIM-o及benchmark资源 | 需原500例与文本候选；处理权重不是论文生成器 |
| [DualTalk metric.py](https://github.com/ZiqiaoPeng/DualTalk/blob/main/metric/metric.py) | 动作统计实现与公式可参考 | 当前SID调用顺序与函数定义相反，必须先修复并记录版本；不能直接复用其SID数值 |

共同测试应固定来源清单、裁剪／音量／fps／分辨率、生成种子与采样预算、失败/缺脸/静默处理、检测器和聚合方式。禁止用GT挑最佳样本却只给其他方法一次采样。500h选集含部分来源dev，最终test必须核对实际来源隔离，不能只看split字符串。

本文已确认三组benchmark方向；类别、阈值与人工问卷仍由后续实验确定。本调研没有把GLARE的时间IoU规则擅自加入本文整段事件F1。

补充：OVI1.1单次最长10s，本文pair可到11.95s，需为全部方法固定共同窗口或公开切块/拼接协议，不能只截短某个baseline后与完整GT混评。早期按“48条非零＋模型表现最好152条”挑选的200条是探索集，不能作为无偏正式主测试；本建议不替代作者后续已选择或将冻结的随机测试清单。
