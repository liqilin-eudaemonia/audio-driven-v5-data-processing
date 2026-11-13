# 当前 pair 切分代码入口：固定 J v1

当前使用 `listen-pair-segmentation-j-v1`。实际规则模块为 [listen_local_recovery_semantic_trial.py](../../scripts/listen_local_recovery_semantic_trial.py)，不是旧全量运行脚本默认使用的 V1。文件名和内部 revision `sentence-integrity-two-source-trial-6` 保持原样，以保留已验证代码的指纹；名称中的 trial 不表示应该改用别的历史模块。

本目录是当前切分方案的统一交接入口。算法与 46 份已归档源码均保持不变；本次整理补齐代码分工、参数、运行方法、输出口径和验证入口。

## 先找这几个入口

| 用途 | 入口 |
|---|---|
| 当前 J 切分规则 | [规则模块](../../scripts/listen_local_recovery_semantic_trial.py)：`SourceContext.propose()` |
| 原始 Seamless 全量持续切分 | [全量集群入口](../../scripts/run_listen_pair_j_full_cluster.py)、[配置](../../configs/listen_pair_j_full_cpu_cluster.json)、[64,622 条运行与恢复](../../data_corpora/seamless_listen_j_full_20261029/README.md)；用户已授权并启动 |
| 四机 CPU 分片与断点续跑 | [新集群入口](../../scripts/run_listen_pair_j_cluster.py)、[四机配置](../../configs/listen_pair_j_cpu_cluster.json)、[有限试跑与恢复说明](../../experiments/pair_j_cluster_pilot_20261029/README.md)；不生成审核视频 |
| 已验证的 J 运行入口 | [随机五条 run.py](../../experiments/pair_semantic_random5_20261029/run.py)，显式参数 `j`；只处理 selection.json 中的五条 |
| 切分后音质筛选与局部恢复 | [rerun_fixed_pairs.py](../../experiments/pair_boundary_real_audit_20261029/rerun_fixed_pairs.py)，由上面的入口设置 `repair=J`、输入和输出目录 |
| 版本、参数、源码 SHA256、验证证据 | [manifest.json](manifest.json) |
| 原样源码归档 | [code/](code/)，保留项目相对路径；46 份文件清单见 manifest |
| 校验项目源码与归档是否一致 | [verify.py](verify.py)，只读，不切视频、不加载音质模型 |
| 全部对照、统计与 HTML 链接 | [最新随机五条](../../experiments/pair_semantic_random5_20261029/README.md) |

`run.py` 是固定样本的复现实验入口，不是任意输入的通用 CLI。历史 [run_listen_local_full.py](../../scripts/run_listen_local_full.py) 仍 `import listen_local_recovery as policy`，直接运行它不会得到 J。新增集群入口显式调用固定 J，支持四机分片、精确缓存复用和逐候选断点续跑；用户后续授权八小时内完成原始全量，现由 `run_listen_pair_j_full_cluster.py` 持续处理，独立输出、不替换旧生产结果。带 `pilot` 的入口只用于有限试验。

## 代码分工与实际调用关系

| 文件 | 负责的内容 |
|---|---|
| [J：semantic_trial](../../scripts/listen_local_recovery_semantic_trial.py) | `safe_cut` 检查合法切口；`greedy_with_lookahead` 做短前缀后的长句前瞻；`recover_region` 保护已选完整句末并处理短尾；`recovery_is_complete` 判断恢复子段；`SourceContext.propose` 组织初始切分与局部恢复 |
| [H：head_attach](../../scripts/listen_local_recovery_head_attach.py) | `attach_heads` 补接未覆盖的同一人短回应；`quality_fallback` 在补接后未通过音质时退回原 pair |
| [L：localbc](../../scripts/listen_local_recovery_localbc.py) | `prepare_people` 做局部 backchannel 分类；`sentence_units`／`end_priority` 区分真正句末和推测分句；`role_regions` 保留角色尾部回应；提供基础贪心候选切分 |
| [基础 local_recovery](../../scripts/listen_local_recovery.py) | 源数据上下文、候选行结构、区间运算、角色与音质阻断区间 |
| [plan_listen_v8_audio_full](../../scripts/plan_listen_v8_audio_full.py) / [listen_v8_recovery](../../scripts/listen_v8_recovery.py) | 原声学切分与兼容性检查；前者重导出后者的 `acoustic_cut`，J 沿用该路径 |
| [run_listen_v8_audio_full](../../scripts/run_listen_v8_audio_full.py) | 沿用现有音质模型、精确区间缓存与最终路由，由实验 runner 调用 |

实际关系是 J 调用 H 和 L，H 也调用 L，L 依赖基础模块。归档中的 [F：v2](../../scripts/listen_local_recovery_v2.py) 和 [G：gap5](../../scripts/listen_local_recovery_gap5.py) 是历史修复版本；其相关逻辑已延续到后续实现，不能把这些历史文件误当作 J 的直接 import 链或新的运行入口。

运行流程：源转写／词时间／VAD／RMS → 局部角色与反馈判断 → **先切完整候选** → 原音质筛选 → 必要时局部恢复 → 保存终端结果与被替代的父候选。恢复子段不能确认完整时，保留原整段及原 QC 结论供复核。候选边界、音质路由和人工认可分别记录。

## 固定参数与保留的行为

| 项目 | 当前 J v1 |
|---|---|
| pair 最短／最长时长 | 2～11.95 秒，按实际裁剪区间计算 |
| 长停顿阈值 | 相邻词间隔超过 5 秒作为长停顿分界 |
| 句末选择 | 真正句末优先于推测分句；不因多覆盖几个词而追加下一句残尾 |
| 短前缀前瞻 | 限于不超过 3 个词、跨度小于 1.6 秒的前缀，后接 1～5 秒停顿；仅在前缀会挤断后续长句、后句可独立合法切分时触发 |
| 句首短回应补接 | 同一人、与后续正式发言间隔不超过 0.3 秒，并通过原有声学／角色／音质检查；失败退回原 pair |
| listener backchannel | 允许与 speaker 重叠；局部反馈判断不等同于实质换人 |
| 音质与恢复 | 原音质规则不变；无法确认恢复子段完整时保留原候选，不自动解除音质阻断 |
| 运行范围与资源 | 原样本为串行 CPU，初始试跑每机 1 worker；正式全量按时限逐档升至每机约 24、最多 32 个，每 worker 主要计算 2 线程、音质 CPU；不应用新的人脸裁剪准入 |

当前没有启用 ≥3 秒或 ≥4 秒的要求。后续若只做时长过滤，保留片段的起止不变；若把它加入切分约束，需另立版本重新验证。代码中的 1.6 秒语音跨度检查与最终 pair 最短 2 秒不是同一个量，不能统一替换这些常量来改最短时长。

## 校验与复现

以下命令从项目根目录执行。只检查当前代码是否仍为固定版：

```bash
python releases/listen_pair_segmentation_j_v1_20261029/verify.py
```

该命令核对项目源码和 `code/` 中的对应副本。规则模块 SHA256 为：

```text
8a7f326fafbda95382a986ce19a634270e42e915db050fd07bf89afaa6ec7f0a
```

需要重现已有五条的 J 计算时，使用下面的命令；本次整理未执行重切：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python -u experiments/pair_semantic_random5_20261029/run.py j
```

此命令读取已冻结的 [selection.json](../../experiments/pair_semantic_random5_20261029/selection.json)、五份源数据快照及现有归一化音频／音质证据；结果写入该实验的 `results/`。它会更新该实验的结果文件并按需计算未命中缓存的音质证据，不修改旧全量结果。`run.py h` 是历史 H 对照，当前使用 J 时不需要重新跑 H。

从准备快照、生成新旧媒体到页面验收的完整过程见 [实验记录](../../experiments/pair_semantic_random5_20261029/README.md)。其 `prepare.py` 复用已有 selection，不应为重现本轮而删除选样文件；`finish.sh` 会生成预览并更新共享 HTML。新增来源应使用独立实验目录和名单，显式选择 J、复用同一流程，保留本轮记录。

`code/` 是源码归档，不是包含原视频、模型、数据索引和运行环境的独立安装包。运行仍使用项目目录和既有资产。当前样本 runner 复用源快照，关闭场景切割并采用未裁剪双人画面；只 import J 类后直接走另一套默认源加载／场景检测流程，不等同于这次已经验证的完整入口。

## 输出与审核口径

主要输出在 [随机五条实验目录](../../experiments/pair_semantic_random5_20261029/)：

| 文件／字段 | 含义 |
|---|---|
| `results/<pair>/initial.json` | 音质筛选前的完整初始候选 |
| `results/<pair>/qc/*.json` | 逐段音质证据及来源；最终恢复注释以 result.json 为准 |
| `results/<pair>/result.json` 的 `clips` | 终端结果，可能通过、待复核或排除；不是全部都可以训练 |
| `superseded_attempts` / `initial_ids` | 被恢复／回退子段替代的父候选，以及初始候选身份；父子不能重复统计覆盖 |
| `audio_final.pool == keep_candidate` | 规则自动通过；不等于新增人工确认 |
| `semantic_recovery_hold` | 放弃不完整恢复子段，保留原候选与原音质状态 |
| `diagnostic.unresolved` | 未成段内容与恢复诊断，不应计作已保留 |
| [intervals.csv](../../experiments/pair_semantic_random5_20261029/intervals.csv) | 旧全量／H／J 各阶段的全部起止、原文和状态，含不变片段 |
| [comparison.json](../../experiments/pair_semantic_random5_20261029/comparison.json) / [review/manifest.json](../../experiments/pair_semantic_random5_20261029/review/manifest.json) | 统计、完整对照分组及媒体入口 |

HTML 的“整段待人工检查”对应 `semantic_recovery_hold`；“仍未自动通过”是其他尚未自动通过、单独补列供试听的候选。这两个展示标题不是音质严重程度，也不表示人工已确认有问题。

## 验证记录与已知取舍

- [最早两条 J 对照](../../experiments/pair_semantic_trial2_20261029/README.md)：San Diego、During、求职句、残尾及长句修复。
- [原抽样剩余五条](../../experiments/pair_semantic_remaining5_20261029/README.md)：包括 I like that.、I mean…、Yeah 等短回应取舍；用户已认可展示的方案，不能据此宣称所有短回应都会保留。
- [新增随机五条及 HTML 入口](../../experiments/pair_semantic_random5_20261029/README.md)：H→J 初始切分 140/150 不变；旧全量→J 最终通过 92/131 不变、131→113 段。两种基线不能混用。
- [数据校验](../../experiments/pair_semantic_random5_20261029/validation.json)、[浏览器校验](../../experiments/pair_semantic_random5_20261029/browser_validation.json)：全部差异与未通过候选已展示，116 个不同对照媒体播放通过；5 项 [切分回归测试](../../tests/test_listen_semantic_trial.py) 已通过。

固定 J v1 是固定当前切分与保守恢复基线，不是证明全量无误杀或所有差异都已人工确认。新阈值或新规则应另建版本并比较，保留本次源码及证据指纹。本次整理未重切数据、未启动全量、未切换生产或训练结果。
