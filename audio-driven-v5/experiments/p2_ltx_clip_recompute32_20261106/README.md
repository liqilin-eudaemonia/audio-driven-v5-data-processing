**已回退，禁止自动重启本候选。** 短程ABBA数值及速度验证通过，但正式训练在第329步停滞，故未满足持续稳定要求。原始异常未抓取到，不能把显存压力推断写成已确认OOM；已回到[原all运行](../p2_ltx_clip_train32_after_gpu67_20261106/README.md)，详情见[回退记录](ROLLED_BACK.json)。下方采用记录仅为故障前历史。

# 当前500h完整联合损失：四分之三重计算已采用

第305步完整状态迁移验收通过，已实际恢复至第328步（本次记录时间点），四机32卡持续训练。当前进度读取 `status.json`。每300步保存，下一周期检查点600。

完整ABBA：all两轮5.722696/5.887121秒，three_quarters两轮5.556717/5.547738秒；均值5.804908→5.552228秒，步时减少4.35%，吞吐提高4.55%。全部轮次损失、完整梯度、模型更新、Adam状态和梯度诊断逐值一致。各轮最长11.95秒样本两次真实更新均通过；32rank最大reserved峰值133.740234GiB，保留134GiB进程上限。

- [测试原始结果](../p2_ltx_clip_recompute_trial32_20261106/RESULT.json)
- [完整状态迁移](MIGRATION.json)
- [采用验收](ADOPTION_ACCEPTED.json)
- [实际恢复与监督核验](RUNNING_VERIFIED.json)
- [持续速度与GPU监控](PERFORMANCE_STATUS.json)

测试只决定执行策略，所有测试更新均丢弃。模型、Adam和32份RNG逐值迁移自原第305步；数据、损失、batch、sample/noise ordinal及保存频率不变。生产步时因样本时长不同不能直接用于计算相对加速比。

控制器在候选运行异常时会将最近完整状态恢复至原all运行；以控制器 `../p2_ltx_clip_recompute_trial32_20261106/state.json` 为实际状态，避免重复启动。

# 配置与恢复约定

当前状态以 `status.json`、`QUEUE_STATE.json` 和控制器 `state.json` 为准。

- 父训练：`../p2_ltx_clip_train32_after_gpu67_20261106`，第305步完整保存。
- 测试：`../p2_ltx_clip_recompute_trial32_20261106/RESULT.json`。
- 配置：`config.json`，只改变输出路径、重计算比例和显式profile元数据。
- 保持AV + .05 LTX hidden整段Event + .025视觉Feature + .025音频Feature + speaker Whisper、原始Qwen缓存、train+dev500h、global batch32、每300step保存。
- node12/node8/node4/node11各0–7卡，rank0在node12，master_port29838。
- 进程内存上限134GiB，FP64 SUM桶512MiB、预取2、GPU Adam状态不变。

严格入口：`scripts/p2_ltx_recompute_profile32_20261106.py`，底层复用 `scripts/p2_ltx_events.py`。加载本目录配置/权重的进程，须先执行 `from p2_ltx_recompute_profile_20261106 import install; install()`，再导入训练/推理入口。该profile锁定父配置SHA，仅允许本次重计算策略变化，不放宽原checkpoint校验。

控制器：`../p2_ltx_clip_recompute_trial32_20261106/CONTROLLER.json`，负责测试、迁移、继续训练；正常运行时不要重复启动。`MONITOR.json`和`PERFORMANCE_STATUS.json`记录性能监控。原检查点保留；测试更新全部丢弃；迁移必须验证模型、Adam及32份RNG逐值一致。
