# E1：50 小时子集，三机 24 卡，一轮训练

**E1 已完成全部 1,268 步（2026-10-26 09:40 UTC 前），三节点 returncode=0、stopped=false，最终检查点 SHA256 已核对。** [完成状态](runs/20261023/E1/status.json)、[最终权重](runs/20261023/E1/step_001268.pt)、[检查点完整标记](runs/20261023/E1/step_001268.complete.json)。E0 已自动接续进入三机全量预检查，见[切换验收](../p2_e0_after_e1_train50h_20261026/transition_validation.json)。

**E0 后续对照已由用户授权并排队：** [接续控制器与实验说明](../p2_e0_after_e1_train50h_20261026/README.md)。E1 完整结束且资源释放后，复用本目录同一 config 和 pair.json，从相同初始化训练 E0；其结果写入 `runs/20261023/E0/`。不改变当前 E1 作业。

用户已确认使用全部三机 24 卡，global batch=24，训练 1,268 个 optimizer steps。30,415 个唯一片段，共 50 小时零 4 秒；共使用 30,432 个样本位置，先完整遍历一次，再从下一轮取 17 个位置。训练 seed 为 20261023。

当前状态以 [各节点状态](node0.json)、[训练日志](runs/20261023/E1/metrics.jsonl) 和最终 [status.json](runs/20261023/E1/status.json) 为准。准备完成标记为 [READY.json](READY.json)；启动前仅有数据准备产物，不能据此认为已开始训练。训练完成也不等于已经验证生成效果或事件监督收益。

**启动验收已通过（2026-10-26 07:47 UTC）：** 三机 24 卡已实际完成超过 100 步并继续训练；全量预检查通过，七类全部启用。第 100 步自动保存 [step_000100.pt](runs/20261023/E1/step_000100.pt)（643,425,063 字节），模型与优化器张量均有限、优化器 step 均为 100，含全部 24 rank RNG，身份与 run.json 一致，文件 SHA256 通过。43 项 TensorBoard 指标均有第 100 步记录。见 [检查点验证结果](checkpoint_000100_validation.json)、[只读复核脚本](verify_first_checkpoint.py)、[启动配置/统计核对](startup_training_validation.json)。此次检查没有暂停或恢复作业；既有恢复逻辑沿用已验证版本。

前 100 步的常见耗时约 5.5–6.3 秒/步；10 秒三机采样平均 GPU-util 为 94.99%，见 [GPU 采样汇总](gpu_sample_summary.json)。GPU-util 包含通信等待，应结合实际吞吐判断速度。实际张量峰值约 114.45 GiB，allocator reserved 峰值约 138.28 GiB。启动时现有训练器让每个 rank 重复全量检查，CPU/I/O 预检查约 28 分钟；该时间不计入训练 step 的吞吐。

## 冻结设置

| 项目 | 设置 |
| --- | --- |
| 数据 | [原始 50 小时子集](../p2_train50h_backchannel_face_20261026/README.md)，全部为 train，保留独立 dev/test |
| 配置 | [config.json](config.json) |
| 机器 | node12 `172.29.252.11`、node17 `172.29.15.91`、node4 `172.29.225.51`，每机 GPU 0–7 |
| 训练 | E1，AV loss + 0.05 × Event loss；六类视觉＋一类音频 |
| 可训练模块 | Connector、事件头、LTX text-attention LoRA（rank=8、alpha=8） |
| Feature loss | 关闭；不构建 feature decoder、不读取 teacher、不建立对应优化器参数组 |
| 学习率 | Connector 3e-5、事件头 1e-4、LoRA 1e-5；warmup 50 步，随后恒定 |
| 性能 | alternating gradient checkpointing，32 MiB 梯度桶；沿用已实测的 24 卡 RoCE 配置 |
| 检查点 | 每 100 步；最后另存 step_001268.pt。完整 checkpoint 同时有 .complete.json SHA256 标记 |
| TensorBoard | 每步记录，30 秒自动 flush，保存检查点时 flush |

训练实现来自已备份版本 [p2-av-event-text-lora-h200-tensorboard-20261026](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/tree/p2-av-event-text-lora-h200-tensorboard-20261026/p2_training/av_event_lora_tensorboard_20261026)。本目录新增数据准入视图和启动文件，未修改训练算法。之后 E0 对照应使用相同数据、seed、24 卡、batch=24、1,268 步和其他训练参数，不能与旧 batch=8 的实验直接当作严格配对结果。

## 数据准入与缓存

原始标签保留了特征抽取阶段的 `formal_training_admitted=false` / `feature_preparation_only=true`。根据用户明确要求开展本次 E1，在独立目录中记录 [training_release.json](training_release.json)。[准备脚本](prepare_training_view.py) 逐样本检查源标签哈希、原生标签规则、非 debug 身份、完成状态、目标媒体绑定和 input/VAE/events 缓存身份。

派生标签只改变上述两个阶段标志，并增加可追溯的训练准入记录；原事件、描述、时间、coverage、unknown、training_admitted 及弱背景假设全部不变。仍为弱标签，没有宣称人工验证。原始子集、原标签和共享缓存不改写。

- `manifests/` 指向原始固定清单，训练范围严格限定为 30,415 段。
- `cache/input`、`cache/vae`、`cache/text` 复用既有缓存。
- `cache/events` 使用包含新标签 provenance 的新身份记录；payload 指向经过 SHA256 验证的原事件张量，字节不变。当前 rasterize 不读取 provenance，且所有影响 rasterize 的字段逐条保持一致。
- 本目录没有 teacher 链接。缓存引用可以包含全量数据，不能通过遍历缓存扩大训练集。
- [release_samples.jsonl](release_samples.jsonl) 记录每条源/派生标签哈希及新事件缓存身份。[READY.json](READY.json) 保存七类实际正负监督统计。
- 原生训练入口仍在每个 rank 启动时完整读取并校验所有训练样本、媒体及 input/VAE/events 载荷；该阶段占用 CPU/I/O，完成后才加载 GPU 模型。

## 启动、日志和恢复

三台机器各执行一次，node-rank 分别是 0、1、2：

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python \
  /private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_e1_train50h_lora_tb_20261026/run_node.py \
  --node-rank 0 --detach
```

启动器检查 READY、配置和源码哈希及本机 8 卡空闲状态；将作业脱离 SSH/终端，保留节点 supervisor / torchrun PID 和退出状态。不使用 `--s0`，也没有基准测试的短超时。

- 节点状态：`node0.json`、`node1.json`、`node2.json`；节点日志为对应 `.log`。
- 每个 worker 的 stdout/stderr：`worker_logs/`。
- 训练目录：`runs/20261023/E1/`；每 rank 明细 `train_rank*.jsonl`，全局聚合 `metrics.jsonl`。
- 检查点：`runs/20261023/E1/step_000100.pt` 等；预算末尾为 `step_001268.pt`。
- TensorBoard：`runs/20261023/E1/tensorboard/`。

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/tensorboard \
  --logdir /private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_e1_train50h_lora_tb_20261026/runs/20261023/E1/tensorboard \
  --host 127.0.0.1 --port 6006
```

需要正常暂停时，在训练目录创建 `STOP` 文件，作业会在 step 边界保存并退出。恢复时确认三节点旧进程均已退出，移走 `STOP`，三节点使用同一个完整检查点给启动命令增加 `--resume /绝对路径/step_XXXXXX.pt`。不能改变 seed、world、配置、源码、数据、缓存和权重身份；启动器若仍标记 running，先核实 PID，禁止重叠启动。断点恢复包含优化器、每 rank RNG 和 TensorBoard 清理逻辑。该启动器不自动无限重试失败作业。
