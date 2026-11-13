# 倾听事件标注输入

[manifest.example.json](manifest.example.json) 是路径占位示例，不能直接运行。用户给数据后填入已切齐的倾听者视频和独立单声道音轨；不需要 speaker 转写。

使用方法、模型端点、复核和 P2 导入见 [完整运行说明](../../related/P2_倾听事件自动标注实现与运行.md)。真实数据到位前可运行 `scripts/check_listen_labeling_code.py` 检查合成媒体与模拟 HTTP 链路。

当前为 v2：先开放发现行为，再独立检查 P2 两类目标。自由描述格式见 [open_discovery.example.json](open_discovery.example.json)，这是格式示例，不是实际标注结果。

RealTalk 原画测试包可用 `prepare-realtalk` 生成带 crop/512 变换的视觉清单，随后 `--mode visual` 打标。无框样本单列，源混音不送入音频标注；命令及本次核验结果见运行说明第 9 节。

Seamless 双路包可用 `prepare-seamless --samples /path/to/items.json` 准备独立 listener 视听清单，随后 `--mode split` 或 `--mode omni`。浮点框明确取整后裁成 512；完整命令见运行说明第 10 节，实际模型问题见 [小样验证报告](../../related/P2_事件打标_真实数据与模型小样验证.md)。
