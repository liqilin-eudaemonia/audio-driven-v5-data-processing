# L2L复现依据复核与此前训练决策纠正（2026-11-07）

用户要求以论文依据确定复现方案，拒绝凭训练曲线临时更改方法。当前所有本地L2L训练已停止；原轮last停在27749步，短预热predictor诊断停在569步。均保留权重、优化器、RNG和日志，不进入正式基线，也不自动恢复。

## 依据及版本

1. [L2L原论文，CVPR2022版](../../writing/02_Learning_to_Listen_CVPR2022.pdf)，PDF p.4/p.6。
2. [作者arXiv完整论文及补充，17页](original_paper_recheck_20261107/arxiv_full.pdf)，[来源及哈希](original_paper_recheck_20261107/arxiv_full.pdf.source.json)。补充B位于PDF p.12–13，listener-agnostic实验位于p.14。下载自 https://arxiv.org/pdf/2204.08451 。不同论文版本若有差异应继续注明，不能混成一套配置。
3. [官方仓库README](https://github.com/evonneng/learning2listen#training-from-scratch)，本地README第156行明确说明VQ loss先升后降是正常现象，典型训练约2天/4GPU；必须VQ收敛后才训练predictor（第158行）。时长是作者设备上的经验，不是我们机器上的训练预算或收敛阈值。
4. [官方VQ配置](https://github.com/evonneng/learning2listen/blob/main/src/vqgan/configs/l2_32_smoothSS.json)、[predictor配置](https://github.com/evonneng/learning2listen/blob/main/src/configs/vq/delta_v6.json)、[优化器工厂](../../compare/learning2listen/src/vqgan/vqmodules/gan_models.py)、[调度实现](../../compare/learning2listen/src/utils/optim.py)。本地固定commit a0825d58b91bae63a5550bac115b5f7ba03ff36c，兼容性改动见compare/deployment/l2l_compat.patch。
5. [DIM论文及补充](../../writing/17_DIM_ECCV2024.pdf) PDF p.9–11、p.21–23：统一EMOCA表示，在ViCo重训L2L；公开材料未补齐全部基线训练细节。既有[协议审计](dim_protocol/AUDIT.json)仍有效。
6. [DiffListener官方补充A.3](../l2l_difflistener_protocol_audit_20261106/07114_supp.pdf) p.3：ViCo与RealTalk重训L2L，遵循原方法设置；这句话并不能解决L2L论文和公开代码之间的差异。

## 已核对的论文—代码—本地差异

| 项目 | 论文/作者依据 | 公开实现/本地状态 | 结论 |
|---|---|---|---|
| 两阶段训练 | L2L补充p.12–13，README要求VQ收敛后冻结再训predictor | 本地架构和两阶段框架存在 | 不能仅凭1000步优于常量均值就宣称VQ已收敛 |
| VQ预热 | 补充p.12：4000 steps | 官方JSON：100000 steps | 存在明确版本差异，不能混用或任意改1000再称复现 |
| predictor预热 | 补充p.13：4000 steps | 官方JSON：100000 steps | 同上；569步短预热诊断不计为复现 |
| VQ优化器 | 补充p.12：Adam | 官方工厂：AdamW | 需声明选择论文版还是发布实现版 |
| VQ恢复调度 | 官方gan_models.py中，加载epoch>500权重时会set_init_lr(4.5e-6) | 本地完整恢复刻意保持保存的scheduler而非再次执行该分支 | 恢复一致性正确不代表已复现作者实际长期学习率历史；作者启动/重启历史需核实 |
| VQ隐藏维度 | 补充p.12：hidden512、12层、8头 | 官方JSON：hidden256、12层、8头 | 不能称默认JSON严格逐项匹配论文补充 |
| 其他VQ设置 | 主文p.6及补充：K200、dz256、32帧VQ输入、batch32 | 默认代码大体对应 | 使用64帧训练对/两32帧半窗需说明所处阶段 |
| 训练长度 | 补充p.12–13分别1000 epochs（原数据规模） | JSON num_epochs5000000/500000，本地曾固定200000 updates/min150000 | 不能将epoch等同step，也不能将本地预算冒充作者协议 |
| listener建模 | 主文p.6特定listener；补充p.14明确支持agnostic跨人训练 | 本地混合ViCo多人 | 有方法层依据，但ViCo具体配置与身份划分仍需单独声明 |
| 头姿表示 | 主文p.4：Euler head rotation，按每视频rest direction对齐 | 本地特征直接拼DECA encoder pose6，没有显式Euler转换/rest对齐 | 现缓存不能直接声称完整匹配论文预处理；需核对原数据/作者处理路径 |
| DIM对比空间 | DIM p.9–10明确所有基线统一EMOCA后重训 | 本地是DECA50+global3+jaw3 | 本地DECA分数不能并入DIM EMOCA原表直接排名 |
| ViCo划分 | 对标须使用同一已公开train/test及样本范围 | 用户选择397保守方案，train351/val46，隔离1个共享speaker区间 | 合理的独立实验选择，明确偏离官方398条，不冒称完全相同划分 |
| 输入/帧率 | L2L补充p.12–13：16kHz mel、运动30fps、音频4倍序列、speaker motion+audio、无listener GT历史输入 | 已有128维原始mel、4t对齐、30fps，仍需逐项冻结版本 | 是有依据的适配；GT不得参与输出选择 |

## 撤回此前不充分的判断

- loss先上升本身不能证明训练坏了：作者README已明确提示这一轨迹正常。27749更新也尚未走完发布配置的100000预热，不能只凭该段曲线认定必然不收敛。
- 初始化码本RMS小于encoder RMS、某批次梯度范数差距大，只是数值观察，不是bug根因证明。原量化器本身就使用[-1/K,1/K]的均匀初始化。
- 将码本改成训练潜变量初始化、将warmup缩到1000，是工程/方法变体消融，不是来自L2L论文、DIM或DiffListener的已证明复现设置。短实验即使改善，也不能直接转为复现baseline。
- min150000/max200000、均值重建改善5%、unigram改善1%、方差区间都是本地预算/健全性检查，不是论文训练停止或最终结果等价的证据。
- 与官方单步更新一致、恢复状态完全一致，证明执行/恢复正确；不能代替预处理/版本协议核对，更不能证明模型已收敛。

## 正确的执行顺序

1. 首先选定目标：复现发布的L2L代码版本，还是复现DIM在ViCo上的重训baseline。两者表示与配置可能不同，不能混称。
2. 为每项数据处理、网络/损失、优化器/调度、训练长度/选择、推理采样和指标，记录PDF页码/官方文件行号/固定commit。冲突与未公开项单列；任何本地选择都明确写出。
3. 优先用已有的官方Conan测试特征、配对官方权重和统计检验环境、预处理与原始重建/推理流程；这一步不训练，也不触碰我们的ViCo test43。它是实现核验，并不等于已复现论文整表。
4. 核对DECA头姿表示与rest对齐；若目标是DIM，先落实EMOCA数据链路及其指标/统计，不能拿现DECA缓存冒充。
5. 之后再固定一套有明确依据、披露全部差异的重训协议和有上限的计算预算。预算内未完成收敛，应报告未完成，不临时改方法去追分。不按test选择超参数/seed/样本。
6. VQ收敛验证后才冻结并训练predictor，最终用规定的source-only推理和相同评分协议测试。

**当前仅完成依据复核，训练保持停止；尚未补齐的细节不以猜测填充。**

## 官方发布模型的原数据锚点检查（只推理，0更新）

已执行verify_official_anchor.py：配对官方Conan VQ、官方统计及官方Conan测试前32序列，调用官方load_test_data及原模型forward。输出有限，量化loss0.281534、归一化重建项4.450700，token解码往返最大误差2.265e-5。该结果只是单批实现检查，不能与论文全测试表指标直接比较。

发布checkpoint自身记录epoch3561、optimizer n_steps1546342。说明不能把本地27000步与作者最终训练量等同，也不能推断ViCo必须同样更新数；原数据规模、恢复学习率路径及目标数据差异都应先核对。实际记录见[OFFICIAL_ANCHOR_CHECK.json](original_paper_recheck_20261107/OFFICIAL_ANCHOR_CHECK.json)。
