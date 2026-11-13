# 当前500h训练缓存统一入口（2026-11-05）

**五项缓存均已完成并通过验收。** 使用当前train＋dev选集284,218段／500.00125913h：train 281,060＋dev 3,158，test不加入，源split保留。当前索引为[TRAINING_CACHE_INDEX.json](TRAINING_CACHE_INDEX.json)，绑定同一选集SHA及各完成回执。没有复制旧大文件，也没有删除当前选集成员。

| 缓存 | 完整覆盖 | 新提取／复用 |
|---|---:|---|
| Qwen H | 284,218 | 原始未SFT Qwen2.5-Omni-3B；直接／guided H各284,218 |
| Gemma文本 | 284,218 | 共享固定正负文本＋最新完整caption指导文本 |
| AV VAE | 284,218 | 整份复用276,780；新视频7,438／新音频6,542 |
| speaker Whisper | 284,218 | 复用277,676；新增6,542 |
| listener Feature teacher | 284,218 | 整份复用276,780；新视觉7,438／新音频6,542 |

各运行目录README列出对应独立Reader，四项增量Reader会选择核验后的旧payload或新payload。每个进程独立创建Reader。Whisper和teacher的COMPLETE.json记录增量；完整500h必须看MIXED_COMPLETE.json。

[全部缓存联合回读](ALL_CACHES_JOINT_READBACK.json)覆盖train/dev、整份复用、转正后部分复用和新增样本，Qwen／Gemma／AV原生训练输入准备通过，Whisper时间网格、teacher与当前H的时间／空间layout通过。各新payload在写入时做SHA及张量回读，最终逐条覆盖和缓存身份验收通过。AV复用旧payload曾全量重算SHA；Whisper/teacher旧文件沿用原完整验收与本轮来源/契约核验，读取时继续校验SHA。

Feature teacher出现过3个CPU供数超时块，原错误保存在其incidents/media_timeout_retry/original_failed。只对64个缺失成员改用原生单线程取帧和无损NumPy IPC重试，保持模型、输入、输出及缓存身份；全部补齐。恢复后的CPU fork审计曾等待锁，已改独立进程、Torch单线程后32进程审计完成。当前无未解决失败，不将部分结果冒充完成。

本缓存任务没有启动训练。旧训练入口、clip标签索引、时长分组与实验配置不能仅改路径就假定适配新成员：后续应显式接入当前混合Reader和完整选集，避免旧train-only过滤。具体实验启动情况属于对应训练运行，不能据此索引推断整机是否空闲。
