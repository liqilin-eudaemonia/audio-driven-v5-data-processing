# 三机24卡吞吐调优与四组并行训练

2026-10-27，用户确认node12 GPU7占卡任务已退出，要求最快获得四组结论。已实测资源释放，完整保存E0第105步后调优；正式训练样本/优化器进度未丢失，性能测试输出不计入正式预算。

## 最终采用：四组同时训练

AV、AV＋Event、AV＋Feature、AV＋Event＋Feature各占6卡（三机各2卡），共24卡。每卡microbatch=1，累积4个样本；各组global batch仍为24、seed仍为20261023、阶段预算仍为1268步。E0从本阶段105步继续，其他三组从各自原第1268步权重初始化第二阶段。

原始30,415段、原音频/事件/teacher缓存、固定Gemma文本、Qwen LoRA、各损失系数均保持不变。只改变跨卡分工与梯度累积；每个更新仍使用完全相同的24个全局样本序号、噪声seed和有效计数。

[实时进度](../parallel_queue/state.json)、[任务配置](../parallel_queue/jobs.json)、[调度源码](../../../scripts/p2_legacy_parallel.py)。初始GPU分配在每台机器上均为：E0=0/1，E1=2/3，E0F=4/5，E1F=6/7。

某组完成后，控制器会让未完成组保存当前完整checkpoint，再扩展到12卡或24卡，累积次数对应改为2或1，global batch保持24。每次拓扑迁移均生成新的run目录与`topology_migration.json`，保留原checkpoint；新checkpoint的parent元数据记录迁移来源/哈希和RNG映射。模型和优化器张量不修改。样本按全局ordinal重设CPU/CUDA及Feature generator的seed，不依赖rank分配。

三机各有后台控制器，共享锁仅允许一个调度，SSH guard只管理自身worker。单组失败从最近完整checkpoint恢复，不停止其他组；每组12次失败上限及20分钟无出步超时。每100步保存和TensorBoard继续生效。当前入口是`parallel_queue`，旧`queue/STOP`保留，防止旧串行队列重新启动。

## 实测依据

| 测试 | 结果 |
|---|---|
| 原24卡全重计算，同一批5步 | 平均9.16秒/步；四组逐个运行 |
| 24卡按长度减少重计算，同一批5步 | 平均9.95秒/步；显存增加但这批未提速，因此未用于正式训练 |
| 三组8卡并行 | 每组约14–18秒/步；每卡累积3个样本 |
| 四组6卡并行，同一批5步 | 各组中位数18.30–18.35秒/步，平均18.48–20.99秒/步；四组同时推进 |

单卡候选还覆盖全重计算、四分之三、三分之二、隔层、无重计算，以及关闭Qwen重计算。E1F最长11.95秒验证中：全重计算峰值约62.6GiB，四分之三约96GiB，三分之二约109–111GiB，隔层需约132–133.5GiB；完全不重计算超过128GiB测试限额。5.5秒短片段无重计算约125.8GiB。大显存配置经过验证但没有在24卡端到端测试中稳定获益，最终保留全重计算以提升并行调度的稳健性；显存占比不是决策目标，四组总完成时间才是。

24-rank逐段profiling显示每卡平均前向＋反向约4.32秒，梯度汇总处平均3.73秒；其中汇总自身最快rank仅约0.015秒，多数时间来自片段长度不同造成的等待。数据读取平均0.035秒，梯度指标与裁剪约0.061秒。增加每卡的连续样本数可降低等待比例，四组同时推进改善总吞吐。

这些是短测，含加载、同步、CPU/文件系统波动；不把五步中位数当作全程稳定速度。初步估计从并行正式启动起四组约7–9小时完成，需用后续真实步时修正。

## 正确性与追溯

- [8卡对照](../performance_world8/equivalence.json)：三份8卡副本，与同批原24卡训练完成5步后的模型/优化器张量逐位一致。
- [6卡对照](../performance_world6/equivalence.json)：四份6卡副本也全部逐位一致。迁移保留原参数和优化器，改变rank RNG列表长度；训练每个样本按同一ordinal重新设seed。
- [24卡重计算对照](../performance_distributed/first_comparison.json)：模型、优化器、24 rank RNG均逐位一致，但端到端未更快。
- [执行策略45项测试](tests.xml)通过，另7项并行调度GPU分配和6/12/24卡全局样本序号测试通过。
- [冻结源码/配置指纹](../parallel_queue/frozen.json)、`parallel_source_snapshot/`保留当前调度源码快照。旧串行配置与源码存于`prior_execution/`。
- `execution.json`和`p2_legacy_qwen_fast.py`属于已测试但未采用的候选；正式并行队列使用原`p2_legacy_qwen_lora.py`，LTX/Qwen均保留重计算。

手工查看当前输出路径，应读取`parallel_queue/state.json`的每组`output`；扩卡后路径会改变，旧路径继续保存历史检查点。当前TensorBoard目录为这些输出路径下的`tensorboard`。

停止整个并行队列使用`parallel_queue/STOP`。恢复前移除该文件，启动`p2_legacy_parallel.py --root <本实验绝对路径>`；不要移除旧串行队列的STOP。控制器会读取最近完整checkpoint和持久化GPU分配，严格恢复优化器和运行身份。

正式启动验收：四组均已连续完成至少5次更新，Qwen LoRA梯度非零，Feature两路损失在对应分支正常记录，TensorBoard持续出数。详见[启动验收记录](../parallel_queue/startup_verified.json)。实时步数以队列状态为准。
