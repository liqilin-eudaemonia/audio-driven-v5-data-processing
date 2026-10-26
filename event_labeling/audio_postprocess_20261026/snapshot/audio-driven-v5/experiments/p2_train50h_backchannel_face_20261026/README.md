# 50 小时训练子集：大脸主体集＋稀有类别补充

**已完成：30,415 段，50.00111286 小时，即 50 小时零 4.006 秒。** 全部来自现有 `train`，用于快速训练实验；现有 `dev/test` 保持独立，不重新随机切分这些训练片段作为最终验证/测试。数据制备阶段没有启动训练。

**E1 训练入口（2026-10-26）：** 用户随后确认三机 24 卡、global batch=24、1,268 步。[独立训练视图与运行说明](../p2_e1_train50h_lora_tb_20261026/README.md) 记录本次正式弱标签训练准入，保留本目录的固定数据、标签和共享缓存。实际运行状态见该入口。

**随机 HTML 复核：** [Cursor 转发服务端 8876 页面](http://127.0.0.1:8876/listen_benchmark_review/train50h_random/)、[抽样与播放说明](review_random/README.md)。固定种子抽取整体随机 50 段＋七类各 5 段，本次去重后共 85 段；可播放 512×512 listener 裁剪音视频，查看人脸尺寸、原文事件时间线及标签 JSON。分类补充与整体随机入口分开标识。

**音频后处理版本（2026-10-26）：** 本50h固定范围的新标签、虚拟静音配方及音频latent缓存见 [后处理数据入口](../p2_train50h_audio_postprocess_20261026/README.md)。不新增处理后WAV；后续训练应使用新入口的 `view/config.json`，读取路径已接入新音频与事件缓存。原本目录继续保留处理前的数据与统计。

## 直接使用

数据根目录：

```text
/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_train50h_backchannel_face_20261026
```

| 文件／目录 | 内容 |
| --- | --- |
| [sample_ids.txt](sample_ids.txt) | 30,415 个唯一 sample_id，冻结训练范围 |
| [manifests/observations.json](manifests/observations.json) | 原生 P2 输入清单，official_split / experiment_split 均为 train |
| [manifests/targets.json](manifests/targets.json) | 原生 P2 listener 视频/音频目标清单，媒体路径与哈希保留 |
| [labels/canonical/](labels/canonical/) | 所选样本的七类训练弱标签，逐文件复制原始字节，不改描述、时间或监督策略 |
| [integrated_labels.jsonl](integrated_labels.jsonl) | 原始整合记录的逐字节摘录，包括开放描述、时间区间和证据来源 |
| [dataset.jsonl](dataset.jsonl) | 每段的源 manifest 行号/哈希、标签哈希、四类缓存位置/身份/载荷哈希 |
| [cache/](cache/) | 指向既有全量缓存的目录链接，未重复复制或重算大体积特征 |
| [samples.csv](samples.csv) | 每段人脸/事件统计、筛选分数、入选阶段 |
| [dataset_contract.json](dataset_contract.json) | 数据用途、split、来源与选择器身份 |
| [summary.json](summary.json) | 汇总、逐类数量和分位数 |
| [selection_policy.json](selection_policy.json) | 确切筛选门槛、排序公式与补充记录 |
| [split_audit.json](split_audit.json) | 独立留出集来源检查与文件哈希 |
| [validation.json](validation.json) | 全选集标签、缓存身份、split 与不重叠复核 |
| [COMPLETE.json](COMPLETE.json) | 完成记录、主要产物哈希 |

P2 读取入口仍是 `src/p2_listen/dataset.py`，本目录的输入清单限定这 30,415 段。这里的标签逐字节保留上游 `formal_training_admitted=false` / `feature_preparation_only=true` 阶段标记；本轮正式训练应使用上方独立训练视图的 `config.json`，其中已记录用户授权并建立对应的新事件缓存身份。原有 input/VAE/text 缓存继续复用。**数据制备本身没有修改训练配置、自动启动训练或绕过缓存兼容性检查。** 共享缓存包含全量样本，训练范围应以 manifests 为准，不通过遍历 cache 构建样本集，也不要清理共享缓存目录。

## 选了哪些数据

- **主体部分**：30,138 段，约 49.4825 小时。人脸短边 p10≥224 原始像素；有效人脸帧比例≥98%；没有已记录的原始、语义或时间区间问题；至少有一个非微笑类别候选。
- **摇头补充**：额外 277 段，约 0.5186 小时。人脸短边 p10≥160，其余质量条件相同。主体里原有 74 段摇头，补充后总计 351 段。此门槛共有 362 段可用，11 段因每个源会话最多 8 段的限制未加入。
- 原先预留约 45 小时主体＋5 小时补充空间。其余六类均已超过每类 2,000 段的补充下限，未额外降低人脸门槛；剩余时长继续从主体候选补足。
- 采用完整片段，未裁掉末段来凑整，所以比 50 小时多约 4 秒。平均每段 5.918 秒；不是按假设每段 7 秒固定取 25,800 条。
- 共 9,331 个源会话 pair、312 个既有连通组。同一 pair 最多 8 段；同一 pair 的源时间区间互不重叠，包括不同 listener 角色之间，避免重复计入相同会话时间。

人脸尺寸来自原始编码画面的 facial landmarks，不是 512×512 输入尺寸，也不是预裁框尺寸。本轮的人脸可见性指标是像素大小和检测有效率，没有新增模糊度模型评分。

| 指标 | 结果 |
| --- | ---: |
| 人脸短边 p10≥224 像素的片段占比 | 99.185% |
| 片段人脸短边 p10 的中位数 | 287.28 像素 |
| 片段人脸短边 p50 的中位数 | 303.09 像素 |
| 每段候选条数中位数 | 10 |
| 每段非微笑候选条数中位数 | 4 |
| 每段类别数中位数 | 3 |
| 整合候选总条数 | 495,203 |

## 七类覆盖

| 类别 | 含该类片段数 | 候选条数 | 来源 pair 数 |
| --- | ---: | ---: | ---: |
| 点头 | 13,286 | 36,887 | 6,701 |
| 摇头 | 351 | 406 | 344 |
| 微笑 | 24,028 | 319,826 | 8,474 |
| 大笑／笑声 | 6,860 | 26,229 | 4,086 |
| 皱眉 | 6,535 | 15,658 | 3,980 |
| 惊讶 | 11,527 | 41,975 | 5,732 |
| 音频回应／发声 | 25,331 | 54,222 | 8,593 |

一个片段可含多类，不能把上述片段数相加当作总样本数。七类全部覆盖，但没有为了追求七类数量相等而重复复制摇头数据。类别与事件均继承现有弱标签；候选条数不等于人工确认的真实动作发生次数。

## 筛选分数

在满足质量、train 划分和人脸门槛后，主体排序分数为：

```text
score = 0.5 × min(face_p10 / 384, 1)
      + 0.3 × 类别数 / 7
      + 0.2 × min(封顶事件密度 / 12, 1)

封顶事件密度 = 10 × 封顶候选数 / max(片段时长, 5)
封顶候选数 = 每个类别最多计 3 条，但 smile 最多计 1 条
```

这是用于本轮快速子集的透明启发式排序，不声称是数学意义上的全局最优数据集。分别照顾原始人脸大小、多类反应和事件丰富度，对同类条数封顶以减弱细碎描述的优势。只改变排序，不删除或改写标签；短片段的 5 秒分母下限也只用于排序。

先取约 45 小时主体，再按人脸大小优先补摇头；其余各类若不足 2,000 段，从人脸 p10≥192 的候选中补足；最后继续从 p10≥224 的主体池填满 50 小时。所有阶段均按 sample_id 去重并检查源区间不重叠、每 pair 上限。

## train / dev / test 隔离

沿用 `data_corpora/seamless_listen_local_full/catalog.json` 的现有划分，保留原 official train，不把 dev/test 样本移入 train。目录中的现有保留来源数为 train 62,474、dev 695、test 918 个 pair。

复核了现有 **dev 11,824 条、test 17,549 条** manifest 记录（含各音质处置池），与本子集的 sample_id、pair、连通组、session、规范化参与者均无交集。参与者键为 `Vxx_Pnnnn`，A/B 后缀被合并，防止同人不同后缀被当成独立人物。

既有数据政策已排除 522 个与 train 相关的 eval pair，以及 13 个与 test 相关的 eval pair；本次保持这 535 个排除记录有效。因此隔离结论针对现有清理后的 dev/test，后续不能再把这些已排除的官方源条目加回评测。没有查看 dev/test 的模型标签来调筛选阈值，仅读取来源身份做隔离检查。

## 代码、复现与验收

- [筛选入口](../../scripts/select_listen_train_subset.py)、[实现](../../src/listen_benchmark/train_subset.py)
- [独立验收入口](../../scripts/audit_listen_train_subset.py)
- [筛选回归测试](../../tests/test_listen_train_subset.py)

项目根目录下可生成新版本（已有输出禁止覆盖）：

```bash
python scripts/select_listen_train_subset.py \
  --stats experiments/listen_backchannel_face_stats_20261026 \
  --catalog data_corpora/seamless_listen_local_full/catalog.json \
  --output experiments/p2_train50h_backchannel_face_20261026_rebuild \
  --hours 50 --class-minimum 2000 --max-per-pair 8 --workers 16
```

仅试算选择结果，在上面命令末尾加 `--plan-only`，不会输出或改动数据。

```bash
python scripts/audit_listen_train_subset.py \
  experiments/p2_train50h_backchannel_face_20261026 --workers 16

python -m pytest -q \
  tests/test_listen_train_subset.py tests/test_listen_backchannel_face_stats.py
```

11 项测试通过。全部 30,415 段完成输出文件哈希、原始源行与整合记录哈希、原生输入/目标 schema、标签类别计数、标签原文文件哈希、缓存输入/目标/事件标签身份、时间区间、train 与留出集隔离校验。媒体及缓存载荷存在性逐条检查，完整载荷 SHA256 校验继承上游已完成的全量审计，未重复扫描全部大体积 payload。所有原始数据、共享缓存和原标签保持不变。
