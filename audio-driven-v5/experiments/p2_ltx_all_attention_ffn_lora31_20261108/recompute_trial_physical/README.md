# 全attention／FFN LoRA重计算验证

从完整step96模型／Adam开始，四机31卡，最新500h及当前全部损失；node11 GPU0不参与，保留现有任务。

- 134GiB进程上限：四分之三重计算最长样本OOM，reserved峰值133.7754GiB。
- 137GiB进程上限：最长样本仍OOM，reserved峰值136.9902GiB。
- 移除额外进程显存上限：最长样本真实OOM，reserved峰值138.6660GiB；整卡139.81GiB，进程总占用139.79GiB，只剩16MiB，再申请78MiB失败。
- 全重计算基线及重复均正常，134／137轮均完成损失、梯度、模型更新、Adam和诊断逐值一致核验；物理容量轮在OOM后基线也正常完成。

四分之三重计算没有通过最长样本，不能报告其有效提速。保留全重计算，由driver从完整step96严格恢复。恢复验收见RESUMED_ACCEPTED.json，生产步数见上级status.json。

原始测量：partial_results.json；汇总：VALIDATION_RESULT.json。物理容量轮将候选放在基线之前，原汇总脚本仍假定第一项为基线，最后聚合时KeyError；这不影响已完成的OOM与基线测量。汇总已从原始结果重建，没有重写测量数据。

早期入口适配错误记录保留在recompute_trial和recompute_trial_retry；正式训练所有状态均由89／96完整检查点恢复，测试更新未写入生产模型。
