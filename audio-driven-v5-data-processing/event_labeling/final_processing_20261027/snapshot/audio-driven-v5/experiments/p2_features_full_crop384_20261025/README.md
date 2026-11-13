# P2 全量特征自动接续

**用户已要求停止 20 分钟自动巡检和汇报；现已停用。** 特征抽取与校验继续运行，实时状态仍更新。停止记录见 [monitor/DISABLED.json](monitor/DISABLED.json)。

722,706 段，node17／node12／node4 各 8 卡，加上 node8（172.29.104.220）GPU 4–7，最多 28 张 GPU。上游全量原文整合与最终审计已完成，28 卡已接手特征抽取，每卡 2 个常驻进程，共 56 个。媒体生产并发已提高，并启用像素／文件哈希等价的[媒体导出提速版本](media_export_v2/README.md)。实时完成数以 status.json 为准。

- [流程、规则、资源调速与恢复说明](../../related/P2_全量特征抽取接续与资源巡检.md)
- [实时 HTML](http://127.0.0.1:8876/listen_benchmark_review/full_features/)
- [当前状态](status.json)、[巡检报告](monitor/latest.json)、[会话消息投递](monitor/last_notification.json)
- [运行配置](config.json)、[P2 缓存配置](p2_config.json)、[验证记录](launch_validation.json)、[全量时长核对](scope_audit.json)

特征保存到 `cache/{input,teacher,vae,events}/<sample_id>/`；共享文本在 `cache/text/`。全部核验后生成 `index.json`、`COMPLETE.json` 和原生 P2 总清单。原始视觉／音频／ASR 全部完成后开始计时，首份及后续每 20 分钟资源报告发送回当前 Codex 会话，日志与有界自动调速持续在后台运行。

没有启动训练。正式训练准入状态与媒体通过技术校验分开记录。GPU 常驻 VAE 与 input／teacher 预取的一致性检查已经通过，40 份缓存逐元素一致。每卡预设 2 进程、每进程 2 CPU 线程、预取 2 段；详见 [配置压测](../../related/P2_特征抽取28卡配置压测.md) 和 [capacity_preset.json](capacity_preset.json)。

CPU 控制器已切换到 [controller_v2](controller_v2/README.md)：校验待处理缓冲 64、实际校验进程仍为 8，原 GPU worker 与校验语义不变。恢复同一控制器时使用该版本入口。

2026-10-26：共享配置源码改动触发保护后，已配置[冻结模块续跑恢复](frozen_config_runtime/README.md)，使用原配置快照和原缓存身份，保留当前训练代码改动。实时恢复进度见该目录 deployment.json。
