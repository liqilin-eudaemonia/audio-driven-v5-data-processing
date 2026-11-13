# 完整50h Omni LoRA：自动对照、并行与扩卡

> **2026-10-28 用户最新安排：** 当前 AV＋Event＋Feature 冻结 Qwen 阶段完成第 1,249 步并保存最终权重后停止。后续 Qwen LoRA GPU 对照测试及训练已禁用；须等待用户再次要求恢复。三机已部署最终检查点停止守护，记录见 [停止安排](../p2_full50h_duration384_20261028/controllers/STOP_AFTER_CURRENT.json)。下文接续方案保留供后续恢复参考。

本目录接续`p2_full50h_duration384_20261028`的四组冻结Qwen最终权重。**当前代码与CPU验证已准备，GPU对照等待四个S1最终checkpoint齐备后自动执行。** 当前S1训练保持原样；新控制器等待期间不启动GPU任务。

[统一加速策略文档](../../related/P2_训练加速策略总览与自动接续.md)汇总缓存、384窗口时长对齐、四组并行、动态扩卡、实测依据与复用条件。

## 查看进度

- [接续总状态](status.json)：依次为`waiting_all_four_S1_finals`、`GPU_benchmark`、`formal_S2_training`、`complete`。
- [计划](plan.json)、[四组配置目录](configs)、[源码与配置冻结指纹](FROZEN.json)、[三机控制器](controllers/processes.json)。
- 测试开始后查看`benchmark/status.json`及`benchmark/DECISION.json`；正式训练开始后查看`training_queue/state.json`，其中包含各组实际输出路径、卡数、步数和错误。
- GPU测试期间的权重位于`benchmark/runs/`，不计入正式训练预算；正式权重位于`parallel_runs/`，TensorBoard位于对应输出下的`tensorboard/`。

## 测试和采用规则

四个S1最终权重齐备后，控制器阻止旧队列启动S2，再核验父权重及共同数据/源码/采样身份。对AV、AV＋Event、AV＋Feature、AV＋Event＋Feature分别进行24卡16步基线，然后四组各6卡同时各训练16步；检查每组第16步模型、优化器逐位一致，且Qwen LoRA有非零梯度。

最完整的AV＋Event＋Feature组还会验证6→12→24卡：在第16步后迁移到12卡训练第17步，再迁移到24卡训练第18步，与24卡连续运行结果比较。

比较的是四组全部完成的预计时间：串行为四组平均步时之和，并行为最慢组平均步时。只有预计耗时至少减少5%且正确性检查通过，才采用四组各6卡并行；否则自动开始24卡串行。短测同时保留实际墙钟时间；不预先保证本轮仍有旧实验的并行收益。

正式训练均从各自的S1最终权重重新初始化阶段二及优化器，不使用短测权重。29,975条、512×512、seed=20261023、global batch=24、每组1,249步和同一份384窗口计划不变。24卡时每卡累积1次；6卡累积4次；12卡累积2次。

## 扩卡、故障与恢复

采用并行后，完成组释放的卡分配给未完成组；控制器先要求完整保存，再建立新的12/24卡输出目录和迁移记录。模型、优化器张量不修改，只映射运行拓扑/RNG列表；每条样本仍使用同一ordinal对应的随机种子。

测试报错自动选择24卡串行。正式并行中同组连续多次失败（3次）会触发保存其他组，然后将未完成组改为24卡串行恢复，避免同一并行失败反复占住队列。保留错误报告和已有checkpoint；仍执行数据/源码一致性检查，校验失败不会被当作通过。

三台机器分别启动备用控制器，共享锁保证只运行一个调度器。控制器异常会记录到`controller_error.json`并重试，worker继续使用已有SSH进程组守护。原队列交接完成后保留原目录`STOP`，避免两套调度器抢卡；不要手工同时恢复原S2队列。

启动/恢复控制器（其余控制器存在时共享锁会避免重复调度）：

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python \
  /private/liqi_workspace/project_code/audio-driven-v5/scripts/p2_omni_accel_chain.py watch \
  --root /private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_full50h_omni_accel_20261028
```

本目录`STOP`控制这一接续任务；当前S1的调度仍由其原目录控制。所有父权重、完整50h缓存、原时长对齐采样计划及原阶段目录都是依赖，不要因交接完成而删除。

## 验证与代码

[接续与GPU测试](../../scripts/p2_omni_accel_chain.py)、[并行/扩卡/回退](../../scripts/p2_omni_accel_parallel.py)、[对应CPU测试](../../tests/test_p2_omni_accel_chain.py)。与采样及原调度测试合计42项通过。GPU训练仍使用原`p2_duration_train.py`，未修改正在运行的117项冻结源码/配置。

本地源码快照与启动验收分别记录在`code_backup/MANIFEST.json`、`ARMED_VERIFIED.json`。这些证明任务已准备并挂接，不等同于未来GPU实测已经通过。
