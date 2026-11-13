# Qwen自主规划：guided AV / AV＋Event 原test24

训练已按用户要求保存完整step976并暂停。两组均使用冻结Qwen guided训练的最终step1249权重，同一原test24和seed137/251，原计划96输出。

完整评测尚未完成。第一轮复杂system指令产生非法JSON；改用原生Qwen system＋user任务指令后，只有3/24条规划通过严格校验，其余出现零长度时间段等问题。原始回答和失败队列保留，未用目标标注补救。

用户要求先看视频后，单独执行这3条已通过规划的全部2组×2seed，共12个视频，已完成并发布。[打开自主规划预览](http://127.0.0.1:8876/p2_av_review/autonomous_preview_20261028/)。不是全test24评测，不代表整体模型效果，3条样本的配对指标现已完成，详见preview_metrics/report.md。规划仅有简单点头/眨眼，无发声规划；生成音轨原样保留，不能把有音轨等同有可听发声。

[范围](preview_scope.json) · [12项任务](preview_jobs.json) · [预览状态](preview_status.json) · [预览调度](preview_controller.py) · [页面构建](publish_preview.py) · [完整批次状态](status.json) · [训练断点](TRAINING_PAUSED.json)。

本批使用的新规划指令与完整原回答记录在run协议/各样本plans目录；早期失败记录在run.rejected_planner_v1、cluster24.rejected_planner_v1。没有静默改写、删去样本或把不合格规划当成合格结果。训练仍暂停，尚未恢复。

## 3条预览的计分结果

[配对报告](preview_metrics/report.md) · [完整JSON及逐条音轨诊断](preview_metrics/comparison.json)。12个自主规划输出已计分；真实标注指导取同两组权重的相同3条、相同seed，从已验证的原特征/检测缓存重新汇总。FVD共同3/3，GLARE去额外门槛3/3、严格1/3；UTMOS/SyncNet四条件双seed共同0/3，保留N/A及各输出实际分数。所有评分使用独立listener音轨。
