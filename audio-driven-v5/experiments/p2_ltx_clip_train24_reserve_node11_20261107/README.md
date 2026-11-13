# 三机24卡接续，node11整机用于推理测评

用户已选择空出node11整机。父四机31卡训练在6026步完整保存后正常退出；node11仅移除本训练的7个进程，原推理任务继续。新训练仅使用node12/node8/node4各8卡，global batch24、每卡1条，保留all全重计算和134GiB进程上限。

模型、Adam、前24份对应rank RNG从父6026步完整checkpoint迁移，父31份RNG保留在原目录。AV＋0.05 LTX hidden整段Event＋两路0.025 Feature、speaker Whisper条件、学习率、warmup、标签和所有缓存不变；无需重新打标或缓存。

父调度从3982步开始，6026步已经消耗2044个batch31；剩余93465个样本/noise配对全部保留，按时长重组24条一批，末批新增15条填充，共3895次更新，最终步数9921。绝不能用6026*31直接切父suffix。保持每500步完整保存，下次6500，最终9921另存完整状态。批次组成改变，样本和各自原始noise ordinal保留。

[实时状态](status.json) · [配置](config.json) · [迁移验收](MIGRATION.json) · [剩余数据全量检查](CPU_VALIDATION.json) · [58项CPU测试](TESTS.json) · [node11释放记录](NODE11_RELEASED.json) · [性能监控](PERFORMANCE_STATUS.json) · [TensorBoard](http://172.29.252.37:6006)。

控制器switch.py先核对父完整保存和进程退出，再验证剩余调度、数据/源身份，迁移并逐值核对模型/Adam/RNG，核对3机GPU空闲，再调用scripts/p2_ltx_resume24.py显式启动24个rank。恢复入口为本目录config.json和本目录完整checkpoint；不要将新的源码直接用于旧31/32卡运行。3个变更源文件已在source_before归档。

TensorBoard沿用node12:6006，复制父历史并从6027继续。monitor.py仅只读监控24张训练卡，不查询或占用node11。STOP会让本训练在当前更新完成后完整保存退出。旧9小时goal仍暂停，旧守护保持STOP。

## 实际运行验收

6026步模型／Adam／24份RNG逐值一致，6936个优化器张量全部有限、Adam step均6026。58项CPU测试通过；24个实际rank与node12/node8/node4的GPU0–7对应一致。已真实完成20次更新到6046，全部loss有限、加权公式正确、五组参数都有非零有限梯度；预热后平均5.86秒／步、中位数5.51秒，读取等待0.046毫秒，rank0峰值93.34GiB。此为启动验收窗口，长期速度看只读监控。

[实际运行](RUNNING_VERIFIED.json) · [24rank与GPU对应](LIVE_PROCESSES.json) · [采用验收](ADOPTION_ACCEPTED.json) · [源身份](LIVE_FINAL_VERIFIED.json) · [TensorBoard九项loss对照](TENSORBOARD_VERIFIED.json)。监控已区分冷启动加载与训练停滞，SSH瞬时失败仅记为telemetry_missing，不自动重启训练。
