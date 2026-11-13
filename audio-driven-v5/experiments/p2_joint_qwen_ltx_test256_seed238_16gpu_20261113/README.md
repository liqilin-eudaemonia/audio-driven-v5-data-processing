# 四组Seamless256：第二轮seed238

2026-11-13按用户要求由双seed接续程序启动，仅node8／node12共16卡。与seed137固定同四组、同256条、相同模型与全部条件；只改变音视频生成seed为238，1024条重新生成和评分。

- [双seed完整交接与平均结果入口](../p2_joint_qwen_ltx_test256_two_seeds_20261113/README.md)
- [本轮实时状态](status.json) · [本轮结果](RESULTS.md) · [协议](PLAN.json)
- 条件缓存精确复用seed137，输出独立保存在本目录combinations/；不复用第一轮latent、视频或评分。
- 恢复优先用上述双seedfollowup入口，保留后续自动均值汇总。仅恢复本轮可在node12运行本目录run.py queue，外层接续会识别已有控制器。

首次采样实际seed、权重与规划核验由接续程序自动记录到上一链接目录的SEED238_LIVE_ACCEPTANCE.json。完整结束以本目录COMPLETE.json为准。
