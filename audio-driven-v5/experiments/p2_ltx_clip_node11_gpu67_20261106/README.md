**已确认实际双卡更新：第46步已完成，WORLD_SIZE=2，6/7均参与计算，五组梯度正常。每50步保存，下次为50。详见RUNNING_VERIFIED.json。**

# node11 GPU6＋GPU7 双卡接续训练

当前为真正的双卡联合训练：CUDA_VISIBLE_DEVICES=6,7，torchrun nproc_per_node=2，WORLD_SIZE=2，local rank0→物理GPU6，local rank1→物理GPU7。两个进程共同优化同一模型，每张卡各累积16条样本，总global batch32。梯度通过原有FP64 SUM归并，AV/Feature按global batch归一化，Event按全局已知标签数归一化。

原单卡正常更新到step45后保存并退出。本轮从其完整checkpoint接续，下一次更新为46，预算仍8882，学习率/warmup沿用当前步数，每50次参数更新保存完整checkpoint，下一次为50。

原checkpoint保留在`../p2_ltx_clip_node11_gpu6_20261106/step_000045.pt`；本目录`step_000045.pt`是明确记录world1→2迁移后的版本。模型与Adam状态直接保留，只更新实验输出目录、world及仅涉及单机多GPU启动的源码身份。两份rank RNG由原rank0状态复制，训练仍逐全局sample ordinal重设CPU/CUDA及显式generator种子。下一步32条样本与ordinal集合、分配交织顺序已核对；不宣称不同world后续浮点计算轨迹逐位相同。

迁移不放宽核心checkpoint resume检查。新worker仍执行完整hash、模型/优化器/配置/源码身份及RNG校验。`MIGRATION.json`记录源文件哈希、world变化及源码差异；源码差异严格限制为启动脚本，模型/loss/优化器源码不变。

全量284,218条缓存准入已重新通过；94项CPU回归通过。训练源码和配置归档为training_source.tar.gz，映射在SOURCE_SNAPSHOT.json。

进度：status.json；逐步指标：metrics.jsonl；控制状态：CONTROLLER_STATE.json；控制日志：controller.log；worker日志：launch/node11.log。后台控制进程见CONTROLLER.json。

正常停止时创建本目录STOP，当前更新结束后会保存退出。接续时使用本目录config.json、cluster.json及本目录完整checkpoint；不再使用旧单卡控制器。旧单卡已停止且不会自动重启。
