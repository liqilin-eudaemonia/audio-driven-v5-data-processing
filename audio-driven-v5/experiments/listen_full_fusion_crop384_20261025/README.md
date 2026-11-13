# 全量原文整合自动接续

范围：722,706 段，1,024 个分片。当前已启动自动接续；完成情况以 `status.json` 为准。

- [规则、资源交接、目录和恢复说明](../../related/P2_全量打标自动整合接续说明.md)
- [实时监控](http://127.0.0.1:8876/listen_benchmark_review/full_fusion/)
- [状态](status.json)、[启动验证](launch_validation.json)、[配置](config.json)

原始结果只读：`../listen_raw_crop384_20261024T131300Z/raw/`。

最终候选保存到本目录 `labels/*.jsonl`，完整原文与证据在 `evidence/*.jsonl`，逐条问题在 `issues/*.jsonl`。七类仅作为附加语义元数据，保留原始自然语言。全量文件哈希与样本数核验后才生成 `index.json` 和 `COMPLETE.json`；带问题完成与全量无问题完成分开记录。
