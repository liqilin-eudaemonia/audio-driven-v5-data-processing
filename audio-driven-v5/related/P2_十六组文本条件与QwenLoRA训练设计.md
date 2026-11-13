# 十六组文本条件与Qwen LoRA训练设计

2026-10-27。用户已确认：固定/具体时段标注两种文本，各有AV、AV+Event、AV+Feature、AV+Event+Feature四组；第二阶段可开启Qwen条件编码LoRA接收同组生成与辅助损失梯度，不做规划文本SFT。共8种配置的第一阶段及对应8种第二阶段。

## 固定约束

- 使用重新筛选的29,975段后处理50h及512×512，读取本子集清单，不遍历全量缓存决定成员。
- 原标注内容原样保留，包括单字中文。模板/时间/字段沿用此前英文指导形式；不翻译发声词。
- fixed条件：固定Gemma文本、输入Qwen hidden；guided条件：同一标注计划同时进入Gemma和Qwen第二次forward，使用全部输入和计划token。A1（只改变Gemma）仍是额外消融，不计入16组。
- 第一阶段Qwen冻结，使用既有input或新增input_plan缓存；第二阶段在线forward，移除提取与融合路径上的hidden detach，不能读取冻结hidden代替在线条件。
- Gemma、Qwen音频/视觉塔、LTX主体冻结；LTX text-attention LoRA各组均开启。
- AV权重1，Event权重0.05，视频/音频Feature各0.1。无Event组保留detach(U)的事件probe，不让其损失更新Connector/Qwen。
- 无Feature组不构建decoder、不读取teacher、不建立对应参数组；有Feature组新建后处理teacher身份，不复用旧音频teacher。

## 第二阶段边界

从相同文本/损失组的第一阶段检查点初始化已有Connector、LTX LoRA、事件头和可选decoder，新增Qwen LoRA；重新初始化第二阶段优化器，不能把阶段切换伪装为严格断点续训。

Qwen LoRA初版采用文本模型self-attention的q/k/v/o投影，rank8、alpha8、lr1e-5；多模态输入embedding与音视频塔冻结。原模型参数不写入实验checkpoint，只保存本项目模块和适配器。阶段内断点续训严格核对数据、配置、源码、模型资产、父检查点及rank数，保存optimizer/RNG/step。

AV→Qwen、Event→Qwen（只在Event组）、Feature→Qwen（只在Feature组）分别测试；Event不更新下游LTX LoRA，Feature不更新下游LTX LoRA。禁止只改requires_grad却被connector.detach或no_grad截断。

## 实现计划

新增独立实现包与脚本，不改写历史四组发布版本或正在执行的缓存代码。内容包括16份配置/矩阵索引、统一模型与loss开关、在线Qwen LoRA、后处理Feature teacher缓存、逐样本文本数据接口、训练与checkpoint/TensorBoard入口以及16组梯度/恢复验证。每100步及结束保存。

缓存任务已单独推进；本请求授权准备代码，不自动启动16组正式训练。训练步数、资源预算和验证数据尚待后续指定；提供需显式填写预算的配置，避免继承数据视图的旧1000步。

状态：16份配置与统一训练代码已实现，相关CPU回归88项通过，真实Qwen在线梯度及静音后teacher GPU提取通过；真实LTX＋Qwen三路loss单样本反传/优化也已通过，边界见[实现与运行](P2_十六组训练实现与运行.md)。尚未开展16组正式训练或得出效果结论。

第二阶段增加训练量，因此比较s2与s1不能单独证明LoRA收益；归因实验需另配相同新增步数的继续冻结Qwen对照。用户已确认的16组规模不因此自动扩大。
