# writing目录：Learning2Listen比较实验的模型来源核查

**2026-11-07补充：** [L2L原论文补充、公开实现和本地训练的逐项差异，以及此前判断纠正](PAPER_GROUNDED_REPRODUCTION_AUDIT_20261107.md)。

核查范围：递归扫描writing下15份PDF；两份DiffListener逐字节相同，因此为14份独立PDF。以下页码均按PDF文件页序，不是期刊排版页码。额外读取了DiffListener官方CVF补充材料A.3。

结论：多个ViCo基线使用在目标数据集重训的L2L；未找到明确声称直接采用Conan/Fallon/Stephen/Trevor某个人物checkpoint进行ViCo零样本评测的说明。重训并不自动说明初始化、是否从零训练、共享一个模型或按身份训练多个模型，这些未知项不能补写成事实。

| 工作 | L2L指标对应的模型／数据 | 证据页 |
|---|---|---|
| [GLARE](</private/liqi_workspace/project_code/audio-driven-v5/writing/00_GLARE_local.pdf>) | 明确写所有比较方法均在其数据集训练和测试；未披露L2L具体人物初始化或checkpoint。 | p.7 |
| [ELP / Emotional Listener Portrait](</private/liqi_workspace/project_code/audio-driven-v5/writing/08_Emotional_Listener_Portrait_ICCV2023.pdf>) | ViCo表的L2L明确重训；L2L原数据集表部分带星号指标直接沿用原论文报告，不是指定某个人物checkpoint的说明。 | p.6 |
| [CustomListener](</private/liqi_workspace/project_code/audio-driven-v5/writing/15_CustomListener_CVPR2024.pdf>) | 明确重训L2L于ViCo；Table1星号也标为重训。没有指定Conan/Fallon/Stephen/Trevor中的权重。 | p.6 |
| [DIM](</private/liqi_workspace/project_code/audio-driven-v5/writing/17_DIM_ECCV2024.pdf>) | 用官方L2L代码、统一EMOCA动作表示重训；Table1/2含L2L普通版及CANDOR预训练的L2L†。LM_Listener数据只含Trevor Noah，不能把该数据身份误作选用官方Trevor权重的证据。 | p.10；p.11 |
| [DiffListener](</private/liqi_workspace/project_code/audio-driven-v5/writing/34_DiffListener_CVPR2025.pdf>) | 正文和补充A.3都明确在ViCo及RealTalk重训L2L，遵循原方法实验设置；未披露哪套人物初始化、是否从零、确切checkpoint。 | p.6；补充p.3 |
| [INFP](</private/liqi_workspace/project_code/audio-driven-v5/writing/35_INFP_CVPR2025.pdf>) | 明确继承DIM的定量和定性结果；Table2的L2L行匹配DIM Table1普通L2L行（非†）。未自行选用四个人物checkpoint测试。 | p.7 |
| [ListenFormer](</private/liqi_workspace/project_code/audio-driven-v5/writing/ListenFormer_ Responsive Listening Head Generation with Non-autoregressive Transformers.pdf>) | L2L方法指标出现在Table3的L2L原数据集，ViCo的Table1/2无L2L行；正文没有披露L2L baseline的具体checkpoint、人物子集或汇总范围，不能断言是四套中的哪套。 | p.6 |
| [Learning to Listen original](</private/liqi_workspace/project_code/audio-driven-v5/writing/02_Learning_to_Listen_CVPR2022.pdf>) | 原论文按特定listener分别训练码本和预测器；train/val/test=70/20/10，定量结果聚合所有listener模型。不是四套权重跨ViCo的选择规范。 | p.6；p.6 |

## 逐篇原文证据

### GLARE

明确写所有比较方法均在其数据集训练和测试；未披露L2L具体人物初始化或checkpoint。

- [PDF第7页](</private/liqi_workspace/project_code/audio-driven-v5/writing/00_GLARE_local.pdf>)：“All methods are trained and tested using our dataset, respectively.”

### ELP / Emotional Listener Portrait

ViCo表的L2L明确重训；L2L原数据集表部分带星号指标直接沿用原论文报告，不是指定某个人物checkpoint的说明。

- [PDF第6页](</private/liqi_workspace/project_code/audio-driven-v5/writing/08_Emotional_Listener_Portrait_ICCV2023.pdf>)：“We retrain PCH [19] and L2L [31] on the ViCo datasets [61]”

### CustomListener

明确重训L2L于ViCo；Table1星号也标为重训。没有指定Conan/Fallon/Stephen/Trevor中的权重。

- [PDF第6页](</private/liqi_workspace/project_code/audio-driven-v5/writing/15_CustomListener_CVPR2024.pdf>)：“We retrain PCH [15], RLHG [44] and L2L [25] on ViCo[44].”

### DIM

用官方L2L代码、统一EMOCA动作表示重训；Table1/2含L2L普通版及CANDOR预训练的L2L†。LM_Listener数据只含Trevor Noah，不能把该数据身份误作选用官方Trevor权重的证据。

- [PDF第10页](</private/liqi_workspace/project_code/audio-driven-v5/writing/17_DIM_ECCV2024.pdf>)：“We retrain RLHG [84], L2L [50] and ELP [66] on the ViCo dataset [84].”
- [PDF第11页](</private/liqi_workspace/project_code/audio-driven-v5/writing/17_DIM_ECCV2024.pdf>)：“pre-trained on the CANDOR [57] dataset.”

### DiffListener

正文和补充A.3都明确在ViCo及RealTalk重训L2L，遵循原方法实验设置；未披露哪套人物初始化、是否从零、确切checkpoint。

- [PDF第6页](</private/liqi_workspace/project_code/audio-driven-v5/writing/34_DiffListener_CVPR2025.pdf>)：“We retrain RLHG, PCH and L2L on both ViCo and RealTalk dataset.”
- [PDF第3页](</private/liqi_workspace/project_code/audio-driven-v5/experiments/l2l_difflistener_protocol_audit_20261106/07114_supp.pdf>)：“following their original experimental setups.”

### INFP

明确继承DIM的定量和定性结果；Table2的L2L行匹配DIM Table1普通L2L行（非†）。未自行选用四个人物checkpoint测试。

- [PDF第7页](</private/liqi_workspace/project_code/audio-driven-v5/writing/35_INFP_CVPR2025.pdf>)：“inherited both quantitative and qualitative results presented in DIM.”

### ListenFormer

L2L方法指标出现在Table3的L2L原数据集，ViCo的Table1/2无L2L行；正文没有披露L2L baseline的具体checkpoint、人物子集或汇总范围，不能断言是四套中的哪套。

- [PDF第6页](</private/liqi_workspace/project_code/audio-driven-v5/writing/ListenFormer_ Responsive Listening Head Generation with Non-autoregressive Transformers.pdf>)：“Table 3: Quantitative results on feature-level metrics with different methods on L2L dataset.”

### Learning to Listen original

原论文按特定listener分别训练码本和预测器；train/val/test=70/20/10，定量结果聚合所有listener模型。不是四套权重跨ViCo的选择规范。

- [PDF第6页](</private/liqi_workspace/project_code/audio-driven-v5/writing/02_Learning_to_Listen_CVPR2022.pdf>)：“each codebook and predictor model is trained on a specific listener”
- [PDF第6页](</private/liqi_workspace/project_code/audio-driven-v5/writing/02_Learning_to_Listen_CVPR2022.pdf>)：“Quantitative results are aggregated over all listener models.”

## 两条关键溯源关系

- INFP第7页明确继承DIM结果；第8页的L2L表情FD=33.93、pose FD=0.06，与DIM第11页普通L2L行一致，而非CANDOR预训练的L2L†行（31.03、0.06）。
- DIM第9页写LM_Listener数据是单一listener Trevor Noah；这说明数据集身份，不能用来证明作者直接加载官方Trevor人物权重。
- ListenFormer第6页Table1/2为ViCo，方法行没有L2L；L2L行在同页Table3，数据集是L2L。正文没有明确给出该baseline人物权重或聚合范围。

## 未找到L2L定量对比行的本地PDF

- `writing/DyStream.pdf`
- `writing/50_OmniResponse_NeurIPS2025.pdf`
- `writing/01_RLHG_ViCo_ECCV2022.pdf`
- `writing/OmniPerson.pdf`
- `writing/OmniHuman-1.5.pdf`
- `writing/author-kit-CVPR2027-v1-latex/main.pdf`

DyStream和OmniResponse中出现的L2L主要是参考文献或指标来源；RLHG PDF检索到的Learning to listen还包含2008年的同名教育文献，不能当作CVPR2022的L2L模型结果。

## 对本地实验的含义

此前四套官方人物权重各跑ViCo43条属于零样本迁移实验，且为DECA空间的独立动作评分。它们不能替代这些目标数据重训的L2L基线，也不能直接与BFM／EMOCA空间的论文分数混排。正式复现需要进一步固定目标训练划分、系数表示、码本与预测器训练／初始化、模型选择及渲染与评分口径；本次核查未启动任何L2L训练或测试。

机器可读证据和文件SHA256见[REPORT.json](REPORT.json)，完整提取文本在texts/。DiffListener补充材料保存在[官方补充PDF](../l2l_difflistener_protocol_audit_20261106/07114_supp.pdf)，下载来源与摘要见同目录SOURCE.json。
