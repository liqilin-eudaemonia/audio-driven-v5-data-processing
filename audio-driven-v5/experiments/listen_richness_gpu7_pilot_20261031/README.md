# 7号卡：同一50段的行为提取prompt验证

2026-10-31。按用户最新要求，只对**50个不同片段**做真实模型试验；同一批比较4版prompt／输出协议。最初200段任务在模型加载期停止，没有完成200段推理。CPU曾准备369条候选，实际GPU输入始终是固定50条。

**结论：原句编号选择有效，但四版尚无一版通过完整语义验收，不能直接用于全量自动丰富度筛选。** v4的50/50是结构、引用和字段校验通过，不是动作提取准确率。

## 看结果

- [50段原文与四版提取对照](review50.md)
- [机器可读的完整对照](review50.jsonl)
- [具体问题与下一步prompt建议](prompt_findings.md)
- [四版资源与速度数据](comparison.json)
- 完整prompt：[v1](prompt_v1.txt)、[v2](prompt_v2.txt)、[v3](prompt_v3.txt)、[v4](prompt_v4.txt)。
- v4统计与原文依据：[export_v4/sample_stats.jsonl](export_v4/sample_stats.jsonl)、[export_v4/behavior_evidence.jsonl](export_v4/behavior_evidence.jsonl)、[export_v4/behavior_catalog.json](export_v4/behavior_catalog.json)。均为候选，没有训练准入。

## 实测

| 版本 | 修改重点 | 最多2次生成后的结构通过 | 首批50段生成时间 |
|---|---|---:|---:|
| v1 | 自由截取quote、行为／次数／时序关系一起提取 | 13/50 | 18.14秒 |
| v2 | 选原句编号，增加baseline，仍输出时序关系 | 31/50 | 63.17秒 |
| v3 | 取消关系图、强化JSON schema、固定动作名示例 | 47/50 | 12.35秒 |
| v4 | 去掉动作示例清单，强调动作必须有原文依据 | 50/50 | 10.62秒 |

这些不是纯prompt公平测速：v2同时修正了校验器不认识single=1的问题；v3增加了结构化生成约束。首次模型加载也更慢。v1首轮回复用修正single后的校验器重放有24/50通过，仍存在大量引用或次数问题。

v4首次49/50，重试1条通过；生成合计约12.49秒，含模型加载总计42.68秒。首次50条生成阶段GPU平均97.8%，峰值100%；含单条重试的生成阶段平均91.4%，峰值显存129.10GiB。不能用v2的高利用率掩盖重复生成造成的无效计算。

## 资源与输入

- 主机：172.29.189.119:22，dedicated-developjob-js-infer-machine-ws3qe；使用已有集群SSH密钥。
- 仅物理GPU7，UUID `GPU-4d5ee808-e829-c824-c506-cad80ff133f5`。其余0–6卡始终是其他任务，未停止或改动。
- 模型：`/private/ckpts/Qwen3-30B-A3B-Instruct-2507`，BF16，vLLM 0.19.0，TP=1。
- batch上限64、实际同时50条；调度token32768，max_model_len12288，单条输出上限8192，memory=.92，OMP/MKL线程8。
- 使用前缀缓存、chunked prefill、FULL_DECODE_ONLY CUDA Graph，图尺寸1/2/4/8/16/32/64。
- 从既有索引随机抽96个来源，每来源最多4个已完成caption，随后按点头／有声／静音／脸大小／旧框分层选50条。种子20261031；这不是按全体片段等概率抽样。原文输入在 `inputs50.jsonl` 和 `requests50_v*.jsonl`。
- 所有50条原caption文件哈希核对未改变；不重新打视频标签、不改变静音配方、不写WAV或改缓存。

任务结束后模型进程正常退出，GPU7回到7MiB、0%；0–6卡继续运行原任务。没有全量接续或后台常驻服务。

## 代码与状态

- 批量入口：[scripts/run_listen_richness_batch.py](../../scripts/run_listen_richness_batch.py)。启动前检查单卡可见性、指定GPU UUID及空闲显存，拒绝叠加其他任务。
- 提取／校验：[extraction.py](../../src/listen_richness/extraction.py)。保留SYSTEM_V1/V2/V3常量，当前默认为实验性v4。
- `stats50_v4.sqlite`保存本轮50条；旧DB按代码合同保留，不与新版混用。
- `pilot50_v*/attempts/`保存所有原始回复和失败原因，`protocol.json`记录模型／代码／推理配置，`resources.jsonl`为秒级采样。
- CPU回归31项通过。模型HTTP与离线批量使用同一JSON schema和原句编译器。
- `code_snapshot/MANIFEST.json`是本地源码快照，不代表GitHub备份。
