# feishu_docs 方法范围核对（2026-11-11）

此前“本轮全量完成”仅指已接入的可运行子集，不代表飞书主表全部完成。主表去重、排除用户指定不测的Avatar Forcing后共20项；4个方法家族已有结果（DualTalk仅原生动作），16项仍未完成。

## 来源与去重

- 主清单：[评测v2.md](../../../feishu_docs/评测v2/评测v2.md)。两张表的重复条目合并；ViCo baseline与RLHG合并。
- L2L的Conan/Fallon/Stephen/Trevor是同一方法的四套人物权重。GPAvatar是渲染后端，不计作新增倾听生成基线。
- 第42/74行的lm-listener独立计数；它指向sanjayss34/lm-listener，不能用CVPR2022 L2L的Trevor结果充当完成。
- 文档把ListenFormer归在可直接推理组；实际已有结果来自本地重训，权重来源差异必须保留。

## 主表逐项状态

| 方法 | 状态 | 说明 | 来源行 |
|---|---|---|---:|
| L2L / Learning to Listen | 已有结果 | 四套公开人物权重均有43条结果；GPAvatar渲染及统一评测完成；不是四个独立方法 | 10 |
| ListenFormer | 已有本地重训结果 | 43条结果；使用本地200轮重训权重，不是作者公开checkpoint | 11 |
| DyStream | 已有结果 | 43条视频及图像、统一EMOCA指标完成 | 12 |
| ViCo baseline / RLHG | 暂缓训练/未完成 | 主表明确暂缓重新训练；此前本地盘点只有诊断训练结果 | 14 |
| DIM | 暂缓训练/未完成 | 主表明确暂缓重新训练；此前本地盘点缺可用核心生成权重 | 15 |
| GLARE | 独立训练流程/未完成 | 主表要求用我们的数据训练；属独立训练流程，本轮未完成其正式评测 | 17 |
| DualTalk | 仅原生动作结果 | 官方权重43条原生动作结果完成；RGB及统一RGB→EMOCA为N/A | 20 |
| DiTaiListener | 待核查资源与接入 | 此前未取得可运行完整生成资产，仍需核查接入 | 25 |
| DiffListener | 待核查资源与接入 | 此前未取得可运行完整生成资产，仍需核查接入 | 26 |
| ELP | 待核查资源与接入 | 此前找到论文但未取得可运行生成资产 | 27 |
| PCH | 待核查资源与接入 | 内部别名mm2022；此前缺核心生成checkpoint | 29 |
| MFR-Net | 待核查资源与接入 | 此前找到论文但未取得可运行生成资产 | 30 |
| CustomListener | 待核查资源与接入 | 此前项目Code链接指向论文；未取得可运行生成资产 | 31 |
| ReactFace | 待核查资源与接入 | 此前缺核心checkpoint和渲染资产 | 32 |
| ARIG | 待核查资源与接入 | 此前仅找到项目页面仓库，尚未接入生成器 | 33 |
| ECHO | 待核查资源与接入 | 此前找到论文但未取得可运行生成资产 | 34 |
| SpeakerVid-5M | 待核查资源与接入 | 此前仓库为数据整理/指标，未取得可运行生成器 | 37 |
| OmniResponse | 待核查资源与接入 | 存在评测框架/数据接入，不等于生成模型已接入；此前缺完整生成代码/权重 | 38 |
| lm-listener（文档称L2L-trevor） | 待生成接入 | 文档明确指向sanjayss34/lm-listener；不是CVPR2022 L2L的Trevor权重。此前已有官方预训练链接线索，尚未完成本地生成接入 | 42 |
| Latent Behavior Diffusion | 待核查资源与接入 | 此前找到论文但未取得可运行生成资产；附图为FRCorr/FRDist等反应评价体系 | 43 |

资产缺口说明沿用2026-11-10本地盘点。本次核对的是文档范围与完成记录，没有重新检索所有远端发布页；“本地未取得”不能写成“官方没有开源”。
此前待测登记实际上保留了这些16项，但上一轮汇报没有把它们与已完成子集一起说明。[原待测登记](../../p2_vico_additional_optimization_20261110/PENDING_METHODS.csv)还额外包含Writing中出现的INFP。

## Writing 与其他范围

[Writing.md](../../../feishu_docs/Writing/Writing.md)还讨论OmniHuman-1.5、OmniCharacter、OmniAvatar，并在指标说明中提到INFP；这些未被计入上述20个主表实验行，也没有本轮已测结论。AnyTalker、OmniPersona出现在主表分组/指标来源栏，不能仅凭表头就判定其为已指定的独立ViCo实验行。
Learning and Evaluating Human Preferences for Conversational Head Generation、Multiple Appropriate Facial Reaction Generation等在Writing中用于评价方法讨论，需与生成基线区分。
主表还引用ResponseNet交接，包含586对样本及评测接入；这不是本轮ViCo test43，也不应宣称一并完成。

## 指标范围也未全部覆盖

飞书表头还包括FVD、LPIPS、UTMOS_v2、Emo-Acc、Emo-Div、ΔResp、MLLM反应检测和人工评价。本轮完成的是README中明确列出的图像指标与EMOCA动作指标，没有把这些额外指标一起跑完。没有生成音频的方法，其音质指标应标N/A。

后续执行范围应保留全部未完成项：优先核查lm-listener的已知预训练入口及其余资源缺口；ViCo/DIM保持文档的暂缓训练状态；GLARE沿用独立训练任务，不擅自启动或改变其他任务。
