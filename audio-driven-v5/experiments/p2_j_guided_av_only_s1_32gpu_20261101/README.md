# 双通路文本指导，仅 AV loss 对照

2026-11-01：代码、四机32卡真实测试和性能对照已完成；正式训练尚未启动。对应当前 `p2_j_guided_av_event_s1_32gpu_resume200_20261101` 实验。

## 对照设置

| 项目 | 仅 AV 组 | 当前 AV＋Event 组 |
| --- | --- | --- |
| 生成目标 | AV flow-matching loss | AV＋0.05×整段 Event BCE |
| Event MLP | 不执行前向、不计算 BCE、冻结参数 | 19类整段多标签监督 |
| 数据 | 相同285,638条train、同一份J缓存 | 相同 |
| 文本条件 | 相同自然语言标注、双通路指导 | 相同 |
| Qwen | S1冻结hidden缓存；不加载规划SFT LoRA | 相同 |
| Connector | 1024维、4层、16头 | 相同 |
| 可训练生成参数 | Connector＋音视频text-attention LoRA | 相同 |
| LR | Connector 3e-5；text LoRA 1e-5；warmup 50步 | 相同 |
| 预算 | 从第0步初始化，1轮8,927步，global batch=32 | 相同总预算 |
| 采样 | 复用原顺序；前100步随机，后续512窗口时长分组 | 相同 |
| 保存与监控 | 每200步及结束时保存；每步TensorBoard | 当前接续配置相同 |

不使用AV＋Event已经训练过的检查点来初始化新对照。相同种子 `20261023` 初始化全部模块；事件头仍按原顺序构造以保留随机数消耗和检查点键名，随后冻结。因此它不会影响初始Connector或text LoRA参数，也不会获得梯度或Adam状态。优化器保留空的events参数组，以复用现有训练循环和检查点格式。

新全量1,095,039条train＋dev文本筛选属于规划SFT准备；这次AV对照继续使用已验收的AV缓存，不切换数据。

## 代码与配置

- 入口：`scripts/p2_j_av_only.py`，在自身进程安装仅AV模型和目标函数，并复用现有训练循环、分布式通信、时长采样、TensorBoard与严格恢复。
- 训练配置：`configs/p2_j_av_only_20261101.json`。
- 四机配置：`configs/p2_j_av_only_32gpu_20261101.json`，node17／node8／node12／node4各8卡，独立端口29831。
- 未修改当前AV＋Event所依赖的源码，其检查点源文件指纹不变。

必须使用本入口。原有 `loss=av` 实现仍训练detach后的Event探针；本入口额外要求 `event_head_training=false`，完全跳过事件头和事件损失。配置保留通用 `loss_weights.event=0.05` 字段以兼容共享配置校验，但本入口目标函数不读取它。实际 `weighted_total=av`；`event`、`per_event/*/loss` 和 `grad_norm/events` 的0是关闭占位值，不是事件预测得分。原标签数据库仍由共享数据读取器核对，不参与本组监督。

## 启动

在开发机项目目录运行。`commands` 只打印四机命令；`launch` 才会启动训练。已按后续指令加入[AV＋Event之后的自动队列](../p2_j_av_event_then_av_20261101/README.md)，无需手动执行launch；以下命令保留为独立启动说明。

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
../Omni-LTX-stage2/.venv/bin/python scripts/p2_j_av_only.py commands

# 等四机可用后，由用户安排启动；日志目录必须是新目录。
nohup ../Omni-LTX-stage2/.venv/bin/python -u scripts/p2_j_av_only.py launch \
  --logs experiments/p2_j_guided_av_only_s1_32gpu_20261101/transport_first \
  > experiments/p2_j_guided_av_only_s1_32gpu_20261101/launch.log 2>&1 &
```

采用BF16训练、实测选出的隔层block梯度检查点、32MiB梯度通信桶、134GiB单进程显存上限（32卡短测为128GiB，正式续训长样本验证后调高）、NCCL／IB网络配置及512窗口时长对齐。AV＋Event已按用户要求在4,200步保存后正常停止，退出时保存到4,201步；文本筛选使用另一台机器，未受影响。

从本组自己的检查点恢复示例：

```bash
../Omni-LTX-stage2/.venv/bin/python -u scripts/p2_j_av_only.py launch \
  --resume experiments/p2_j_guided_av_only_s1_32gpu_20261101/step_000200.pt \
  --logs experiments/p2_j_guided_av_only_s1_32gpu_20261101/transport_resume200
```

严格恢复检查实验、源码、数据和检查点哈希，不接受其他实验的检查点。正常停止沿用本输出目录的 `STOP` 文件。推理时可将本入口 `install_backend()` 返回值传给 `p2_matrix16.inference.restore(..., backend=backend)`，保持源码身份检查和相同权重格式。

## 验证

`preflight.json` 已生成：285,638条，数据身份与当前AV＋Event一致、共享源码指纹一致、完整确定性采样计划一致。复用现有完整缓存验收结果，并记录其路径及SHA；本次没有重新遍历全部媒体。

6项CPU测试通过，覆盖：AV反传、事件头不执行且不更新、相同初始化及RNG、分布式损失归一化、四机启动参数、错误模式拒绝，以及检查点恢复后的下一次更新一致性。随后完成112次真实32卡测试更新；最长11.95秒样本通过，重算策略切换与严格恢复均得到逐位相同的模型、优化器和32卡RNG。配对步时从4.03秒降到3.64秒，默认已改为隔层重算。详见[32卡验证与加速记录](../p2_j_av_only_benchmark_20261101/README.md)。
