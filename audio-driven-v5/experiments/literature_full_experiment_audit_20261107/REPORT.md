# 全部相关论文实验与资源核查

日期：2026-11-07。本报告扩展旧L2L专项核查，回答每篇论文任务、输入输出、测试数据、表格结果来源、方法并集、资源与可比协议。已核查21篇独立PDF的任务／方法条件、实验比较和相关附录，另取得并审阅官方文字补充。保留所有已发现的原文未披露、匿名方法和版本冲突；这是一份文献／资源核查，不声称已经复现全部论文。 本次没有启动新训练或GPU推理。

先看[实际实验建议](EXPERIMENT_RECOMMENDATION.md)：在当前“短speaker audio＋listener首帧→listener AV”任务下，应分共同输入联合AV主表、视觉listener表、额外条件补充表和原协议文献表。没有任何一张现成论文表可以让我们把J短pair分数直接并入。

## 当前成果入口

| 用户问题 | 记录与证据 |
|---|---|
| 每篇任务、输入输出、训练数据、测试集、比较对象与来源 | [21篇逐表记录](PAPER_DETAILS.md) · [机器可读目录](PAPER_CATALOG.json)；当前68组比较表记录＋20组图／正文比较，页码为PDF页序 |
| 全部方法是否漏记、在哪篇哪张表出现 | [85个名称的反向索引](METHOD_OCCURRENCES.json) · [分类与资源登记](METHOD_REGISTRY.json) · [分类计数](METHOD_COUNTS.json) |
| 哪些有代码、权重和评测，哪些需重训或只能暂作引用 | [完整资源矩阵](RESOURCE_MATRIX.md) · [机器可读矩阵](RESOURCE_MATRIX.json) · [重点issue与代码证据](RESOURCE_DELTA.md) |
| 如果引用数字，应引用哪篇、怎样对齐 | [逐方法协议映射](PROTOCOL_MAP.md)；包含来源链、表示、距离定义、划分及额外条件 |
| 我们到底应先实测哪些方法 | [建议实验矩阵](EXPERIMENT_RECOMMENDATION.md)，含全部24个核心候选和外围方法的取舍依据 |
| 每条方法结果到底来自哪里 | [392条逐方法来源CSV](ROW_PROVENANCE.csv) · [结构化JSON](ROW_PROVENANCE.json)，ELP按指标格拆分引用范围 |
| 表格是否遗漏、其他图中比较与版本证据 | [全部编号表角色清单](TABLE_CENSUS.md) · [原文／补充／修正证据](SOURCE_GUIDE.md)：本地116张编号表＋额外14张，共130张；比较实验按68组表＋20组图/正文登记 |
| 原文件与阅读／验收范围 | [PDF清单与指纹](INVENTORY.json) · [当前进度](PROGRESS.json) · [结构校验](COVERAGE_CHECK.json)；结构校验不证明全文阅读或语义来源正确 |

## 已能确定的主要结论

85个规范化名称中，24个是核心任务相关候选：14个视觉listener、7个已知自身音轨的交互视频、3个自主联合AV。另有31个外围方法／基础模型／级联条目及30个控制、团队、变体和未明确身份图中标签。**85不是85个必须重训的独立模型，24也不是24个都有现成权重。** ViCo/RLHG、PCH/PCHG等别名合并，原表拼写／引用编号冲突保留在SOURCE_GUIDE；匿名商业系统保持未识别状态。

| 核心候选的当前资源 | 方法 |
|---|---|
| 官方核心权重入口已核实，需按输出和输入验收：4个 | L2L、DyStream、Avatar Forcing、DualTalk。L2L/DualTalk主要为动作；Avatar Forcing的两个核心文件实际来自Drive；只有L2L和DyStream有本地历史推理证据。 |
| 有官方训练路径但核心模型尚缺／旧链接不可用：6个 | RLHG、DIM、ReactFace、ListenFormer、GLARE、LFI。它们不能统称只能引用；ListenFormer的历史作者链接此前已由用户确认过期。 |
| 部分模块，需补全／重实现：1个 | PCH，不能按完整训练仓库或完整checkpoint登记。 |
| 当前未获得可执行原生成器：13个 | MFR-Net、ELP、CustomListener、DiffListener、DiTaiListener、DSPN、INFP、ARIG、ECHO、StreamAvatar、SpeakerVid-5M、OmniResponse、OmniPersona。保留搜索范围和限制，不作全网绝不存在的断言。 |

这些状态分别依据官方仓库、模型目录、下载脚本、作者页或issue，不依据“论文说将开源”或“存在一个Code按钮”。SpeakerVid开放数据处理及评分；OmniResponse模型仍TODO；ECHO虽有官方仓库但目前只有README/许可证；DiTaiListener作者页的Code/Model为占位。完整逐项链接见资源矩阵。

## 结果来源链及关键协议差异

1. **DIM→INFP→ARIG／Avatar Forcing旧ViCo行**：DIM在EMOCA56维下重训RLHG、L2L并重实现ELP；INFP Table3继承DIM；ARIG Table2继承DIM/INFP；Avatar Forcing Table4的旧baseline引用DIM。L2L普通行与CANDOR预训练†行不同。旧审计误称INFP Table2，本轮已更正为Table3。
2. **MFR-Net→CustomListener／DiffListener／ListenFormer**：这些论文中的MFR引用结果应追溯原MFR表，不能称各自独立复现。CustomListener补充Table5/6的RealTalk则明确重训RLHG/PCH/L2L，不能拿ViCo值填RealTalk。
3. **ELP按单元区分**：ViCo重训L2L/PCH；原L2L表部分带星号单元引用原报告，其他单元未给相同声明。FD叫Fréchet却用L1定义，不能按名称与DIM Fréchet FD等同。
4. **INFP的三个身份**：原INFP、Avatar Forcing在FLOAT latent重实现的INFP*、DyStream作者重实现的INFP*不可合并。ECHO继承某些旧数字的线索和明确引用声明分开，数值相等本身不证明来源。
5. **ReactFace至少两套协议**：Table2为REACT2023／FaceVerse58与多个适当邻居；作者替代版Table9为RLD即ViCo，沿RLHG 3DMM、移除appropriate邻居损失。额外Table8人工研究亦已补入。不能用一套主表描述概括所有测试。
6. **L2L原始人物与下游重训不同**：17页arXiv附录明确四listener聚合；六频道不代表六套模型都进主表。其LFI原数据对照是motion-only重训；best-of200只明确用于专项多样性图，不应推广到主Table1。
7. **目标音轨与任务输入**：INFP/DyStream/ARIG/Avatar Forcing/ECHO原生多为已知双方音轨→视频；本文要生成未知listener声音。self置零可成为视觉适配，但不变成音频生成模型。Avatar Forcing作者issue5确认user视觉可置零，仍需实际适配验收。
8. **GLARE数据与事件指标**：其RealTalk/Seamless是重新整理后的片段与划分；原表全部baseline在其数据训练／测试。p16还使用固定reference video Tref与previous motion context，来源须披露，不能简写为严格只有首帧。baseline650epochs与Table7 final850epochs分别记录。官方reaction eval需已平滑CSV、事件一对一tIoU≥0.5，R-ATD实现×100。本文已确认整段类别出现F1，属于另一评价。
9. **当前公开评测不必然等于发表协议**：DIM默认路径的常量speaker输入、十次GT候选择优与精确test仍有缺口。DualTalk当前metric.py确有SID参数顺序颠倒：用预测拟合聚类、对GT计熵；不能据此断言发表时全部数字都这样产生。
10. **补查定性与补充后的修正**：ECHO Fig5还有StreamAvatar；Fig3原图另有Per-FLOAT、DIM、DualTalk，图内文献编号过期，Per-FLOAT保持身份未明。DiTai新版CVF补充Table6包含Prompt Travel与Teacher Forcing长视频策略；DiffListener补充A.3明确RLHG/PCH去emotion后重训。不能因只看主定量表而遗漏这些比较。
11. **联合AV三套benchmark**：SpeakerVid的VidChatBench500、OmniResponse原长对话、OmniPersona自选Seamless60条约50秒均不是我们J短pair。OmniPersona旧baseline数字与原表一致时只记数值匹配，作者未披露的调用／训练来源保持未知。

## 证据与版本边界

writing递归登记24份PDF：22份研究文件／21个独立指纹，加1份CVPR模板和1份本论文新生成的单页方法图。两份DiffListener字节相同，计一次。ReactFace初始本地PDF解析为0页，曾改读[作者提供18页版本](https://wcxie.github.io/Weicheng-Xie/pdf/TVCG-2024-Luo.pdf)。本轮发现工作区原文件已变为有效18页版本，本审计没有修改源PDF；已重查其Table2/8/9与作者版数值、方法一致，文件／提取文本并非字节相同，版本证据见[重新核对记录](web/REACTFACE_VERSION_RECHECK.json)。初始指纹保留在清单previous_versions中。CustomListener官方5页补充及L2L17页arXiv补充另有来源和指纹记录。

当前写作文件的三条贡献已获作者认可并写入，早期目标快照所说尚未写入已过时。本调研采用当前数据／模型pipeline／benchmark三方向；不改论文任务或擅自冻结类别、阈值、人工量表和最终测试ID。500h选集含部分来源dev，最终test隔离要核查真实来源。

官方页面及搜索快照保存在web目录。早期API有504／HTML错误，web/INDEX.json明确标记；这些失败快照不作为成功repo/issue JSON证据。新浏览证据包括README、issue、模型文件目录、DualTalk metric代码与ECHO官方占位仓库。链接可见、完整payload可取、本地模型推理通过和原表严格复现是四种不同事实。

## 核查结论的适用范围

- 已将本地所有研究PDF、编号表的角色以及已发现的图／正文比较关联到目录；额外官方文字补充的来源与指纹另列。匿名商业模型、Per-FLOAT冲突标签及未命名concurrent-work视频明确保留未知，不进行身份猜测。目录和来源CSV可逐项反查。
- “论文未披露”是调研结论，不是擅自补成官方权重推理的理由。97条方法出现记录仍未逐行披露训练／checkpoint；另有数值匹配但无明确引用声明的行。它们不是97个新方法，也不能据资源存在推断当时怎样运行。
- 资源分类按核查时官方入口与issue证据给出，不声称不存在任何私下模型或未来发布。HTTP错误、未加载的动态评论不作“作者没有回复”的证据。本次不下载每个大模型、不启动新训练或推理，也不要求作者答复才能给出当前可执行建议。
- 若实际启动统一比较，还需独立完成数据适配、checkpoint下载与生成验收、固定test与检测器、训练预算和采样策略。原论文没有报告的精确test或初始化不能靠这份调研补出来；文献表和新实验结果应分开。

要求逐项核对见[完成审计](COMPLETION_AUDIT.md)。结构校验只证明清单／链接／指纹一致，不把绿勾当作原协议复现。

历史入口：[旧L2L核查](../l2l_literature_protocol_audit_20261106/REPORT.md) · [统一测评交接](../../related/P2_参考方法统一测评与L2L重训暂停交接_20261106.md) · [DIM协议审计](../l2l_literature_protocol_audit_20261106/dim_protocol/AUDIT.json)。旧记录保留历史版本，新发现以本报告及链接的逐表记录为准。
