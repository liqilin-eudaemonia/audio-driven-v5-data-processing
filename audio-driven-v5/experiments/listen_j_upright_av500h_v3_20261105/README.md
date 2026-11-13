**当前选集已切换至train＋dev版本。** 本目录的全量v3统计继续复用，`selection/`保留旧train-only版以便追溯；当前284,218段／500.00125913h及train/dev来源数量见[新版入口](../listen_j_upright_av500h_train_dev_v3_20261105/README.md)。

# 新全量标签：冻结v3行为统计与AV约500h选集（2026-11-05）

**已完成并通过独立验收。** 本轮使用node8、node17、node12、node4共32张H200，完成全部1,107,313条新版caption的冻结v3提取、CPU核验汇总及AV约500h选集。最终交付以[DELIVERY_COMPLETE.json](DELIVERY_COMPLETE.json)为准。

| 部分 | 条数 | 小时 |
|---|---:|---:|
| 人脸p10≥224主集 | 264,063 | 462.62965440 |
| 有声补充 | 20,201 | 37.37104409 |
| 全部train选集 | **284,264** | **500.00069849** |

实际入口：[selection/train.jsonl](selection/train.jsonl)；配套[sample_list.csv](selection/sample_list.csv)、[index.sqlite](selection/index.sqlite)、[SUMMARY.json](selection/SUMMARY.json)和[最终独立验收](selection/INDEPENDENT_FINAL_AUDIT.json)。全部选中项保留最新原文和媒体／音频／裁框来源；包含33,276条普通caption复核标记、1,113条已转正样本。补充实际截断处p10为187.54970491px，准入规则下限仍是184px。500h按完整片段求和，超出500h约2.51秒。

全量统计：1,068,740条通过提取校验，38,573条带局部统计复核信息仍保留，无法解析0；130,347条原普通caption复核标记、1,690条转正记录均归账。2,163个块的输入／原响应／统计SHA、全部样本来源／caption／人脸／音频对应，以及每块首尾共4,326条CPU回放均通过，见[STATISTICS_AUDIT.json](STATISTICS_AUDIT.json)。

本轮32个GPU worker均已退出，见[GPU_RELEASE_CHECK.json](GPU_RELEASE_CHECK.json)。核查时部分GPU已在运行其他任务。训练张量缓存尚未重建。

- 全量输入：`../listen_j_labels_upright_20261105/labels.jsonl`，1,107,313条；SHA256 `0851714c52e17dec6c7b5145b82dc53998ab09af0cc7ceda0b9c807192c2b389`。另有9条视觉剔除、1条旧held，总目标1,107,323。
- 模型：`/private/ckpts/Qwen3-30B-A3B-Instruct-2507`；显式使用`production.contract()`冻结v3，prompt摘要`a6eeeef764e602e1c4258ef32e3d905b88c5564401ec3f6febf33a81f1b0473c`，temperature=0、8,192输出token、12,288总上下文，不截断。
- 四机：node8 `172.29.104.237`；node17 `172.29.15.91`；node12 `172.29.252.37`；node4 `172.29.225.51`。SSH22；用户所列服务端口20491／20342／20356／20359仅作为节点信息，独立TP=1副本无需跨机模型通信。配置中的hostname和全部GPU UUID来自本轮实测。
- 各卡D512：512并发、65,536批量token、两块512条输入形成1,024请求池，prefix cache、chunked prefill、decode CUDA Graph；BF16，显存预算0.92。沿用历史提速实测配置，不通过缩短输出或放宽规则提速。
- CPU四机各32线程核验真实caption／质量SHA、音频recipe、人脸测量、25fps检测证据；CPU批量预编码并验证与旧单条模板token逐个一致。使用manifest字节范围索引，避免递归扫描百万文件；不新增百万结果小文件。共享动态队列分配GPU任务。
- 只有一个协调器写输入SQLite及发布GPU队列；四机CPU准备器只写独立不可变包及回执，避免跨机SQLite WAL并发写。最终选集顺序扫描统计文件，避免几十万次随机打开共享文件。

## 本轮预填充显存调整

当前环境为vLLM 0.19.0、torch 2.10.0+cu129。历史D512的65,536预填充token在一次FA3调用中申请5GiB临时工作区，node8 GPU0因当时仅余4.91GiB而退出。已让当前批次完整落盘，保留166,912条结果，再将32卡统一切换为`E512_safe_prefill`：512并发、32,768批量token、两块512条请求池，配合`PYTORCH_ALLOC_CONF=expandable_segments:True`。8,192输出上限、12,288上下文、模型、prompt及统计／筛选合同不变，源码未改写。实测性能及完成情况以本轮resources／回执为准；没有将历史259.47条/秒当成本轮保证。详见`PREFILL_MEMORY_ADJUSTMENT.json`。

## 新版来源适配

旧正常样本仍走历史snapshot完整校验。转正样本按“原始来源→原裁框有效来源→原生无损转正来源→最终裁框”逐级验证，保留真实original_source，不改原标签或来源；质量、裁框、音频、参与者、时间和SHA均需对应。普通caption复核标记按用户授权保留，与硬生成失败分开；未修复病态循环／空caption／非正常结束拒绝进入GPU队列。初始全量manifest及发布验收SHA绑定在`INPUT_MANIFEST_VERIFIED.json`。

相关实现：`scripts/run_j_upright_av500h_v3.py`、`scripts/control_j_upright_av500h_v3.py`。冻结提取／统计继续使用历史`listen_richness.production`、`worker_cluster`、`merge_cluster`及原aggregate。CPU验证含真实转正／普通复核输入、来源损坏拦截、批量token等价及历史统计／断点回归，见`CPU_VALIDATION.json`与日志。

## 最终选集规则

仅取原train，保持原split/group。主集为人脸p10≥224px的全量样本，安静、零计数类别及普通caption复核标记不作额外剔除。

补充范围184≤p10<224px，采用的`counted=1`类别至少一个为speech、vocalization、audible laughter、humming、chuckling（均audio模态），净化listener波形非零且official_core_s>0。按p10降序、p50降序、采用类别数降序、sample ID升序取完整片段，直至主集＋补充达到500h。

不执行规划SFT严格文本筛查／有声捞回，不添加七类、每pair配额或动作丰富度门槛。次数未知为null，持续状态组不等于动作次数。500h按片段时长求和，不伪称去重后的原视频唯一时长。

## 状态与交付

- `RUN_STATUS.json`：本轮实际阶段、完成量、最近窗口速度及四机worker状态。
- `CPU_VALIDATION.json`、`INPUT_MANIFEST_VERIFIED.json`：开跑前验证。
- `config.json`：冻结合同、源码SHA、实际四机／GPU信息、选集规则。
- `producers/`、`preparation.json`、`producer_complete.json`：四机CPU输入准备及覆盖。
- `workers/`、`queues/extract/state.json`：32卡状态与队列。
- `resources/*.jsonl`：实测逐卡利用率／显存及CPU使用。
- `attempts/`：完整原响应与重试；`statistics/`及回执绑定各块输出。
- `tables/sample_stats.sqlite`、`sample_stats.jsonl`、`behavior_catalog.csv`、`coverage.jsonl`、`summary.json`：全量统计与覆盖。
- `selection/train.jsonl`：新版实际500h训练清单，含最新caption／来源／裁框／音频recipe／统计绑定。
- `selection/sample_list.csv`、`index.sqlite`、`SUMMARY.json`、`COMPLETE.json`：紧凑清单、字节索引、统计与完成验收。
- `TASK_COMPLETE.json`：全流程完成标记；没有该文件时仍为运行中或待处理。

本轮不覆盖旧500h清单、旧训练缓存、原始标签或媒体。不把输出清单称为已经重建完成的训练张量缓存。

## 消费新版结果时的字段口径

`labels.jsonl`和选中行保留导出时的原始字段，例如`behavior_statistics_status`及`training_admitted=false`，以便追溯；这些字段不代表外部新版统计尚未执行。实际最新提取状态看选中行的`statistics_status`、`statistics_file`、`statistics_offset`和绑定的caption SHA；实际成员以本目录`selection/`及完成回执为准。原aggregate中的通用`face_supplement_candidate>=160`提示不参与本轮AV500h准入，本轮明确使用184≤p10<224。

## 随机素材复核

[打开最终选集随机24条（8876）](http://localhost:8876/listen_benchmark_review/j_av500h_random24_20261105/)：主集M01–M16、有声补充V01–V08，组内固定随机，不按内容挑换。[抽样及真实浏览器验收](../listen_j_av500h_random24_review_20261105/README.md)。

用户已确认本轮按大致符合音视频的尺度使用，不再追加细节打标；精确时间／次数仅作参考，不阻塞本轮训练准备。见[确认记录](../listen_j_av500h_random24_review_20261105/quality_assessment/user_acceptance.json)。
