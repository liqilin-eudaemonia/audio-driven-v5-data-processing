> **该次尝试在575步发生CPU解码worker停滞，已停止。最近完整检查点仍为500；501–575将重算。下文速度只反映停滞前的短窗口，不能作为稳定吞吐结论。后续使用独立可超时重启的解码进程，见新 isolated 运行。**

# 三机24卡：新筛选数据规划LoRA加速接续

按用户2026-11-02最新要求，在原四机训练完整`checkpoint-500`后，仅使用node17（172.29.15.91）、node12（172.29.252.37）、node4（172.29.225.51），各8卡。node8已退出本训练。

**实际接续已验收：** adapter逐值一致、504份optimizer状态与scheduler从500恢复、24个rank RNG精确恢复、24卡真实梯度检查通过。最新连续窗口见[switch_verified.json](switch_verified.json)；28项CPU回归通过。

## 数据、训练和供数

- train 690,337条／1,155.44h，speaker音频＋listener最终裁框首帧，原回应文本SFT。
- 全局batch保持256：24卡中8张每步10条、16张每步11条。配置中的micro_batch=8保留原时长排序的分组粒度；实际局部batch由FixedGlobalBatchShard分配。
- 全局2,697步，已完成500步；从501接续。每500步保存，最近3个周期检查点，结束另存final_adapter；关闭dev评估。
- 原模型、LoRA参数、学习率、optimizer、scheduler、监督mask和全局样本分组保持不变；32→24改变浮点归约与局部padding，不能承诺未来权重与继续32卡逐位一致。
- 复用有完整来源/像素校验的旧首帧PNG；其余AUTO4线程解码、nearest-PTS等价选帧。每卡4个持久worker，预取2→4，无新WAV/视频文件。

## 接续与验证

[原检查点receipt](../p2_planning_sft_filtered_3b_32gpu_20261102/performance/checkpoint-500-receipt.json)记录全部文件SHA256。`resume_origin/checkpoint-500`将权重、优化器、scheduler与trainer状态硬链接到原文件；RNG按保留物理节点映射：新rank0–7←旧0–7，新8–15←旧16–23，新16–23←旧24–31。旧32份RNG全部保留，映射仅创建24份新路径。

[全量分组检查](../p2_planning_sft_filtered_3b_32gpu_20261102/performance/batch_partition24.json)确认所有2,697个全局batch（含末尾原有padding）与原32卡一致；前500步128,000条不重新读取训练。CPU回归另验证实际DataLoader/跳过已训练batch行为，以及24卡不等局部batch的全局token loss梯度。

[50条输入等价检查](../p2_planning_sft_filtered_3b_32gpu_20261102/performance/input_equivalence50.json)：RGB、PTS、speaker波形、目标、processor全部张量相同；输入读取125.1→13.9秒。这不是三机端到端吞吐数字。

实际恢复检查写入`performance_load.json`、`rng_restore/rank-*.json`和`gradient_audit.json`；运行中的真实步时在`step_metrics.jsonl`。只有`COMPLETE.json`代表全训练完成。首轮三机启动的固定节点数错误发生在加载前，保留在`launcher_failed_node_count.log`，未动检查点，已修复并通过三机命令预览。

## 入口

- [训练配置](../../configs/p2_planning_sft_filtered_3b_24gpu_inputfast_20261102.json)
- [三机配置](../../configs/p2_planning_cluster_filtered_24gpu_20261102.json)
- [优化交接](../../related/P2_Qwen规划LoRA供数加速与无损接续_20261102.md)
- 主调度日志`launcher.log`，每机`launch_logs/`，TensorBoard `tensorboard/`。

继续本次训练时必须用该三机配置和本目录自己的完整`checkpoint-*`执行`--resume`，不能切回默认旧四机配置。初次迁移使用`performance_resume`记录，不是从原基座重训。

## 首个连续吞吐窗口

原32卡401–500步平均4.486秒/步；新24卡511–575步平均1.378秒/步，分别约57.1和185.8条/秒，观察到约3.26倍吞吐。两个窗口样本不同，不是严格同批AB；保存和启动开销另计。node8快照8卡均1MiB，已释放。[资源快照](resources_after_switch.json)。
