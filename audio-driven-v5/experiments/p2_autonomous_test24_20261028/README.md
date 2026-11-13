# 原test24 · Qwen自主规划双次forward评测准备

**仅准备，尚未运行GPU规划/缓存/生成/计分。** 现有HTML中的结果仍是目标标注指导。

[完整方案、运行方式、对照限制与验收](../../related/P2_Qwen自主规划双次forward推理评测.md) · [CLI](../../scripts/p2_autonomous_eval.py) · [冻结协议](protocol.json) · [336项任务](jobs.json)。

同一原test24 × 2seed：当前旧4组、新版guided AV/AV＋Event共288项；新版guided AV＋Feature最终1249步待补48项。没有增加train样本或test成员。

每条样本只用speaker音频、listener首帧与时长生成一个全英文时段回应计划，跨权重/生成seed复用。规划同时进入Qwen第二次forward完整hidden及Gemma。与真实标注指导的缓存、输出和指标目录分离，不读目标标注补救规划。

运行后：`plans/<sample>/`保留原回答和校验记录；`cache/text/`、`cache/input_plan/`为新增两路条件；`outputs/<arm>/`为生成音视频；`metrics/<arm>/test24/`为原生评测结果；`reports/`保留完成、失败和等待记录。

准备不启动GPU，也没有挂接自动执行。须先安排单样本GPU S0，再执行24卡worker。当前训练继续运行。
