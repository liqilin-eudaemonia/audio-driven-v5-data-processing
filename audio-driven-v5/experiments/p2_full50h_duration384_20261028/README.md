# 完整50h：三机24卡时长对齐的八组训练

> **2026-10-28 用户最新安排：** 当前 AV＋Event＋Feature 冻结 Qwen 阶段完成第 1,249 步并保存最终权重后停止。后续 Qwen LoRA GPU 对照测试及训练已禁用；须等待用户再次要求恢复。三机已部署最终检查点停止守护，记录见 [停止安排](../p2_full50h_duration384_20261028/controllers/STOP_AFTER_CURRENT.json)。下文接续方案保留供后续恢复参考。

**后续LoRA接续已更新：** 四组S1保持当前24卡串行；全部完成后由[新接续控制器](../p2_full50h_omni_accel_20261028/README.md)测试并选择LoRA并行/串行方案。本目录继续记录S1进度，之后LoRA进度请看新入口。

2026-10-28已通过24卡ABBA实测并自动启动本版本。数据为完整29,975条、50小时零0.532秒，512×512，global batch=24，seed=20261023，每阶段1,249步。数据、静音后音频、标签、Gemma、冻结Qwen hidden及teacher缓存沿用已验收的完整50h版本。

## 当前入口

- [实时训练状态与输出目录](training_queue/state.json)、[八组配置](configs/matrix.json)、[启动依据](TRAINING_READY.json)。`step>0`表示已经实际更新；`running`但`step=0`可能仍在加载权重。
- [三机控制器](controllers/processes.json)，共享锁只允许一个控制器调度，另两台备用。每个实验占满24卡，先依次训练AV、AV＋Event、AV＋Feature、AV＋Event＋Feature的冻结Qwen阶段，再依次运行各自配对的Qwen LoRA阶段。
- 权重每100步和结束时保存，严格恢复模型/优化器/RNG；TensorBoard位于状态文件列出的各组输出目录下的`tensorboard/`。

后续接续已做[八组加速策略核验](PENDING_ACCELERATION_AUDIT.json)：实际队列入口的CPU dry-run确认八组生成的命令均调用`p2_duration_train.py`、每机8个rank；阶段切换保留同一份384窗口采样计划，LoRA阶段使用对应本版本S1父权重。逐一核验采样计划哈希、窗口边界、末步分配和preflight源码身份；当前运行的计划也一致。该检查未启动额外GPU任务或修改训练源码，不表示后续七组已完成实测。

## 实测结果

[原始结果](../p2_full50h_hard_keep_20261028/performance/duration384/RESULT.json)：同样384个样本位置，A/B/B/A各16步，全部使用三机24卡。原随机顺序两轮平均7.620秒/步；时长对齐两轮平均5.255秒/步，耗时降低31.0%，吞吐约提高45.0%。这是冻结Qwen的AV分支短测，不把该比例直接当作全部八组的全程收益。

第一轮逐卡分析中，前向＋反向时间基本不变，梯度汇总处含等待的平均时间从3.535秒降到1.079秒。第二轮对齐在第8步保存、恢复至第16步，最终模型和优化器与连续运行的对齐版本逐位一致。采样及既有调度相关29项CPU测试通过。

## 采样与可比性

每384个原随机序列位置内按完整时长分组，每24条一个batch，再随机打乱batch顺序。各rank共享计划，不丢样本、不裁短视频、不额外按桶补齐；总计29,976个位置和原方案完全相同，每条位置保留原ordinal对应的噪声seed。不同顺序会改变梯度更新轨迹，因此八组统一使用该计划，并从对应的共同初始化开始。

旧随机顺序训练已保存第53步，只作历史记录，不作为本版本任一组的父权重。[旧权重保存记录](../p2_full50h_hard_keep_20261028/performance/duration384/ORIGINAL_SAVED.json)。本版本各LoRA阶段只从本版本对应冻结Qwen阶段的完整最终权重开始。

## 恢复、依赖和源码

入口为[scripts/p2_duration_queue.py](../../scripts/p2_duration_queue.py)，训练入口为[scripts/p2_duration_train.py](../../scripts/p2_duration_train.py)，采样和严格身份校验为[scripts/p2_duration_execution.py](../../scripts/p2_duration_execution.py)。新源码、配置和计划均进入checkpoint身份；原生模型和缓存源码未修改。

手工恢复时，在没有另一个活动控制器的前提下，移除本目录的`STOP`后执行：

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python \
  /private/liqi_workspace/project_code/audio-driven-v5/scripts/p2_duration_queue.py \
  --root /private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_full50h_duration384_20261028
```

队列会选择最近完整检查点继续。不要移除旧`p2_full50h_hard_keep_20261028/STOP`，否则旧随机顺序队列会争抢GPU。

必须保留旧完整50h目录：本版本的成员view、Gemma/Qwen缓存、采样计划和原始验收仍引用该目录，旧训练前缀被归档不等于整个旧目录可删除。[完整恢复与缓存说明](../p2_full50h_hard_keep_20261028/README.md)、[加速实现说明](../../related/P2_训练加速策略总览与自动接续.md)、[加速源码本地快照](../p2_full50h_hard_keep_20261028/performance/duration384/code_backup/MANIFEST.json)。
