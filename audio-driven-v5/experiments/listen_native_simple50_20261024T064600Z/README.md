# 原生视听简短 prompt：50 段复测

用户明确要求只列出识别到的 backchannel 及其起止时间，不添加复杂限制。本轮使用 51 个英文词的任务说明，并将输出简化为 `events` 数组，每个事件只有 `start`、`end`、`modality`、`behavior`。完整实际请求见 `prompts/` 及逐样本 `request.json`。

同一批 50 段在 node8 的 8 张 H200 上重新运行；模型、8 fps 媒体、listener 独立音轨、0.25 秒原生视听时序参数、4096 token 预算和 temperature=0 与上轮一致。原 prompt 和长 prompt 在页面作为历史对照，不冒充本轮重跑。JSON 输出契约也有简化，所以这不是只改变一个文本变量的消融实验。

结果：50 次请求均完成，42 份通过校验，6 份时间区间无效，2 份达到输出上限后截断。所有原始输出保留；通过校验不等于语义正确。详见 `analysis.txt` 和 `comparison_stats.json`。

前置离线回归 25 项通过。`validation.json` 核对请求文本、模型输入身份、原生视听时间网格和实际生成结果。代码快照在 `code/`，结果页代码快照在 `report_code/`。

HTML：`data_previews/listen_benchmark_review/native_simple/`。默认对比原 prompt 和本轮简短 prompt，可勾选上一版长 prompt；校验失败但可解析的事件仍显示原文，零长度区间不能循环播放。没有补写 inspected coverage、证据或人工标签。

资源记录在 `resources.jsonl`，启动记录在 `launch.json`，结束释放记录在 `resources_after.json`。node8 的新占用任务按用户此前清理该机并使用 8 卡的授权停止，记录为 `node8_cleanup.json`。其他三台训练节点未改动。

用户指出利用率低后，另开 `../listen_native_prefetch8_20261024T065200Z/` 对照：4 个副本使用单 CPU 准备客户端，4 个副本使用 4 个客户端预准备媒体，全部保持 GPU batch=1；每个副本运行相同 8 段，总计 64 次请求。该吞吐测试与本目录的 50 段标注结果分开保存。
