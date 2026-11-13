# 飞书 ViCo 比较表的来源与可比范围

**进一步确认（2026-11-09）：** 已追溯首次发布代码并检查官方补充、全部 issue 和公开资源。FD／SID／Var 的代码规则、test 划分及预训练评测入口的十次择优已经确定；剩余的是原表绑定的 GT／有效帧底稿及实际运行分支。[逐项确认结果与证据](../p2_vico_dim_public_confirmation_20261109/README.md)。十次补测仍未完成，当前单次成绩不改名为十次成绩。

已读取用户提供的 [Writing 本地副本](../../feishu_docs/Writing/Writing.md)及对应截图。**Avatar Forcing 表中的 RLHG、L2L、DIM 三行确实直接沿用 DIM，24 个数值全部一致；其余几张 ViCo 表不能合并成同一指标表。我们的当前结果可单列，但还不能称为严格相同协议。**

[逐格数据与文件 SHA](VICO_TABLE_AUDIT.json)保存截图、原论文及现有结果来源。本次未修改飞书导出的内容。

## 可以确认来自 DIM 的数值

[Avatar Forcing Table 4 截图](../../feishu_docs/Writing/图片和附件/image%2013.png)明确写 † 为继承自 DIM。对照[DIM Table 1 截图](../../feishu_docs/Writing/图片和附件/image.png)和原 PDF，三行四指标的 Exp／Pose 共24个数值全部一致。

以下每格为 Exp／Pose，发表值按原表保留：

| 方法与来源 | FD | rPCC | SID | Var |
|---|---|---|---|---|
| RLHG，DIM 转引 | 39.02／0.07 | 0.08／0.02 | 3.62／3.17 | 1.52／0.02 |
| L2L，DIM 转引 | 33.93／0.06 | 0.06／0.08 | 2.77／2.66 | 0.83／0.02 |
| DIM，DIM 转引 | 23.88／0.06 | 0.06／0.03 | 3.71／2.35 | 1.53／0.02 |
| INFP*，Avatar Forcing 表 | 17.52／0.07 | 0.01／0.07 | 2.19／3.20 | 2.10／0.03 |
| Avatar Forcing，原论文 | 16.64／0.05 | 0.01／0.01 | 3.12／3.00 | 2.80／0.03 |

† 的含义随论文变化：Avatar Forcing 的 † 是“从 DIM 继承”；DIM 原表的 † 是“CANDOR 预训练”。因此 Avatar Forcing 中 L2L† 对应 DIM **未加 CANDOR†的 L2L 33.93**，不是 DIM 的 L2L†31.03。不能按符号相同就选错行。INFP* 也不能换成另一论文未标星的 INFP 结果。

## 现有我们的结果

固定 ViCo test43，联合8000、seed137单次，结果来自[用户确认 Var 定义的版本](../p2_vico_dim_var_per_coefficient_20261108_v2/README.md)：

| 本地模型 | FD Exp／Pose | rPCC Exp／Pose | SID Exp／Pose | Var Exp／Pose |
|---|---|---|---|---|
| base Qwen | 25.89253／0.08714 | 0.01039／0.03493 | 4.55095／3.52459 | 0.84125／0.01401 |
| SFT2752 | 24.33909／0.06182 | 0.00600／0.03062 | 4.64889／3.66934 | 0.85460／0.01062 |

不能由这两张表作严格统一排名：本地使用 EMOCA 和 DIM 发布指标，原 Avatar Forcing 附录使用 SPECTRE，SID 聚类数为15／9，本地 DIM 路径为40／20。继承自 DIM 的三行没有因此被重新用 SPECTRE 评分。Writing 中“四项都用 SPECTRE”的概括，不能扩大到这些继承行。

剩余差异还有：原表有效测试范围未完全确定、我们从 RGB 反提取而部分原方法直接输出系数、当前单次而发布 tester 支持十次 GT 择优、作者实际 Var 汇总未确认。当前 GT Var 为0.904523／0.012038、论文0.93／0.01，接近但不证明同一协议。原十次择优队列本次未恢复。

## 其他截图为什么不能直接拼入

| Writing 图片 | 论文与数据集 | 指标区别 |
|---|---|---|
| image 1.png | DiffListener Table 2，ViCo test／ood | FD 明确是逐帧动作 L1，显示值乘100；不是 DIM 的 Fréchet |
| image 11.png | CustomListener Table 1，ViCo test／ood | FD 是70维动作 L1，显示值乘100；还含平移维度及不同相关性定义 |
| image 7.png | ELP Table 1，上半 ViCo、下半 L2L | 论文把 FD 命名为 Frechet、文字却描述 L1，显示值乘1000；不能用表头名称推定与 DIM 同公式 |
| image 13.png | Avatar Forcing Table 4，ViCo | FD 为分布 Fréchet；三行继承 DIM，其余按该论文新测路径 |
| image.png | DIM Table 1／2 | 上表 ViCo，下表 LM Listener，两个数据集不能混行 |

这些区别均核对了本地原论文的指标定义。DiffListener／CustomListener 的“FD 11.54”等数字不能简单除100后与我们的 Fréchet FD24.339比较；ELP的“4.20”也不能当作同一套L2L基线。

当前可交付的比较方式是：保留 DIM／Avatar Forcing 的发表引用表，单列我们的公开实现补测及协议差异。若需要真正统一训练与测评，应让可运行基线在同一500h训练池、固定测试输入和同一特征提取／评分器下重新生成、再统一计分。
