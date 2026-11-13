**已实际启动并验收：31个rank/GPU对应、纯AV损失及两组梯度通过，21步窗口预热后4.29秒/步，峰值49.07GiB。node11 GPU7未用于训练。6500评测已按用户确认暂停，完成结果保留。TensorBoard：http://172.29.252.37:6007。实际进度见status.json；LIVE_ACCEPTANCE.json为实测验收，GPU_STATUS.json每30秒更新。**

# 7000步保存后，四机31卡启动原始Qwen AV-only

用户已授权：当前三机联合训练完成7000步完整保存后停止，随后在node12/node8/node4各8卡、node11 GPU0–6，共31卡启动独立AV-only训练。node11物理GPU7保留推理。

coordinator.py已后台启动，先等待当前status达到7000（此时原生循环正在存盘），写STOP阻止下一步更新；核对完整检查点SHA并等待父控制器退出。之后必须有PRELAUNCH_ACCEPTED且选定31卡连续两次空闲才启动。不终止其他任务，不使用node11 GPU7。QUEUE_STOP可在启动前取消接续，STOP用于停止本实验。旧9小时goal仍暂停。

数据：最终train281060＋dev3158、284218段／500.001259h，最新完整caption对应的原始未SFT Qwen H、Gemma及混合AV缓存。原始预训练初始化，Qwen条件，不加Whisper、Event或Feature。global batch31，每卡1条，duration496组批，真实sample/noise配对完整保留，尾部21条填充，最终9169步。每500步完整保存，all全重计算、预取2、512MiB FP64归约、GPU Adam。7000步父权重保留供推理／原实验恢复，不作为本消融初始化。

[队列状态](QUEUE_STATE.json) · [启动前验收](PRELAUNCH_ACCEPTED.json) · [配置](config.json) · [数据及32条真实缓存读取](preflight.json) · [23项测试](tests.xml) · [父7000保存验证（生成后）](PARENT_SAVED.json) · [实际开训验收（生成后）](RUNNING_VERIFIED.json) · [持续性能](PERFORMANCE_STATUS.json)。此前[全部284218条caption与Qwen/Gemma契约复核](../p2_current500h_av_only_20261107/RECHECK_FULL.json)沿用，同一冻结选集和缓存不重做。

当前是预约接续，实际开训以RUNNING_VERIFIED为准。原生入口首先做最长样本前后向，并保存0步完整状态；连续更新20步后监控写实际验收。监控检查纯AV公式、loss与两组梯度，并记录步速／显存。遇到训练故障记录QUEUE_ERROR，不自动无限重启；从本实验完整检查点严格resume可恢复模型／Adam／rank RNG。用户关于1000步保存的提问没有被当作修改指令，保存频率保持500。
