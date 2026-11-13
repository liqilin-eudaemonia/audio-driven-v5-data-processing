# 预检查修复、三机验证与 E0 重启

**验收已通过。2026-10-26 用户已授权停止第三台机器占卡的推理任务，两个推理进程及其启动器已终止；11:07 UTC 正式启动三机 24 卡 E0，已进入实际训练更新，启动验收至少完成 15 步。24 个 rank 正常，43 项 TensorBoard 指标已写入，所有已检查 loss/梯度指标为有限数值；详见 [启动验收](training_start_validation.json)。实际训练步数以 [实时状态](status.json) 为准。**

用户要求先停下 E0，完成代码调试后再训练。旧 E0 在全量预检查阶段停止，尚未产生训练更新；E1 已完成 1,268 步，原源码、配置、检查点均保留。旧控制器的 STOP 保留，不会重复启动。

本次修改位于独立 [release](../../releases/p2_preflight_v1/)，训练入口为 [新版 p2_listen.py](../../releases/p2_preflight_v1/scripts/p2_listen.py)。原 `scripts/p2_listen.py` 仍用于旧 E1 检查点的生成及恢复。新版 E0 的生成、恢复使用新版入口，避免源码身份不匹配。

## 改动范围

旧版每个 rank 都完整读取 30,415 段的媒体与训练缓存，24 卡重复 24 次。新版只由全局 rank 0 执行一次完整审计，最多 8 个 I/O 线程并发读取；结果按原始样本顺序交给原有类别统计函数，浮点累加顺序不变。

首次完整审计保存配置、源码、模型资产、清单、缓存视图与文件依赖绑定的校验凭证。再次启动或恢复时，只有 rank 0 核对凭证与文件状态；其余 rank 核对启动身份并共享统计结果。文件状态包含访问路径、实际解析路径、大小、mtime、ctime、inode。当前文件系统时间戳精度较粗，因此 JSON 与不超过 64 KiB 的文件另外计算 SHA256，大文件首次审计前要求至少 2 秒无修改。审计前后依赖变化会拒绝发布成功凭证。

这不是完全取消数据检查：缓存被训练实际读取时，原 SHA256 校验仍然执行；坏凭证直接失败，文件变化触发重新审计。原始媒体应保存在受控、不并发修改的目录。大文件状态复核不是防恶意伪造全部文件系统元数据的密码学证明。

完整审计的成功记录有文件锁及内容哈希；分布式 request/result/ack 通过共享存储交换，在 CUDA/NCCL 初始化之前完成。每次 torchrun 必须给三台机器设置同一个全新的 `P2_PREFLIGHT_SESSION`，重复会话或重复 rank 会被拒绝，错误会传播，并设有等待超时。

关闭 feature 的 E0/E1 只读取 input、VAE、events、text。没有新增 teacher、decoder、损失或优化器参数组。模型、优化、采样顺序、seed、保存周期和 TensorBoard 逻辑保持原版。

## 验证证据

- [CPU 回归](regression.log)：167 项测试，18 个子测试通过；另有 [14 项配对及资源保护测试](control_tests.log) 通过。覆盖凭证损坏、依赖变化、软链接切换、同大小且恢复 mtime 的修改、审计中变动、会话重复、错误传播、超时、并发仅一次审计和训练读取时的 SHA 校验。
- [源码范围证明](source_change_verification.json)：AST 核对训练函数仅替换预检查段，配置函数仅改实际源码身份记录；原 E1 的 70 个源码文件哈希保持不变。
- [真实 GPU 对照](gpu_comparison.json)：三机各 2 张 H200，原版与新版 E0/E1 各 5 步，并比较连续 5 步与第 3 步保存后恢复到第 5 步。核对模型、优化器、每 rank RNG、样本序列、类别统计和 loss；最终验收状态见 [verification.json](verification.json)。
- [全量审计结果](production/audit_summary.json)、[三机 24 rank 审计日志](audit_jobs/)：全量 30,415 段，与原 E1 标签签名和类别统计逐项核对。实测首次三机启动 638.25 秒，复用启动 116.86 秒；其中 rank 0 完整审计 596.50 秒，复用复核 75.30 秒。24 个进程首次完整审计总次数为 1，复用为 0。

原版和新版的源码身份是分别如实记录的。跨版本配对仅允许 arm、输出路径，以及明确列出的预检查覆盖文件不同；没有忽略整份 source 校验。真实模型数值对照另外验证该 I/O 改动不改变优化结果。

## 正式 E0

实际运行状态见 [status.json](status.json)，跨版本配对见 [pair_validation.json](pair_validation.json)。在所有验证通过后，由 [controller.py](controller.py) 检查三机 24 卡空闲并启动持久 supervisor；若有其他任务占卡则记录 `waiting_resources` 并等待，不终止其他任务。控制器和节点进程不依赖 SSH 会话存活。

- 数据：与原 E1 相同的 30,415 段及现有标签、缓存，均通过软链接引用。
- 配置：[production/config.json](production/config.json)，与 E1 仅 output 不同。
- seed=20261023，global batch=24，1,268 步；三机各 GPU 0–7。
- Connector 与 text-attention LoRA 仅由 AV loss 更新；事件头作为 detach 探针单独训练。feature decoder 与 teacher 关闭。
- 每 100 步及最终步保存，逐步 TensorBoard，完整断点恢复。
- 输出：`production/runs/20261023/E0/`；TensorBoard 位于其 `tensorboard/`。
- 节点状态：本目录 `node0.json`、`node1.json`、`node2.json`；日志为对应 `.log`，worker 日志位于 `worker_logs/`。

本目录创建 `STOP` 会请求原生训练在步边界保存后停止。节点异常或配对不一致时，控制器会核验 PID 身份，只停止自己拥有的 E0 进程。恢复时先确认进程退出，移走控制器与 E0 输出目录的 STOP，再执行：

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python \
  /private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_preflight_fix_20261026/controller.py \
  --detach --resume /绝对路径/production/runs/20261023/E0/step_XXXXXX.pt
```

控制器为每次恢复生成新的共享会话，不需要重新跑原版逐 rank 全量预检查。不要启动旧 `p2_e0_after_e1_train50h_20261026/controller.py`。
