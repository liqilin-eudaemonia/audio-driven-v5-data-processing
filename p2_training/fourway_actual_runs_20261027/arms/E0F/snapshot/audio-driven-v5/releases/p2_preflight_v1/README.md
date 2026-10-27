# P2 共享预检查启动器 v1

实际运行与验证见 [实验说明](../../experiments/p2_preflight_fix_20261026/README.md)。入口：

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python \
  releases/p2_preflight_v1/scripts/p2_listen.py train \
  --config /绝对路径/config.json --arm E0 --train-seed 20261023
```

分布式启动时，所有机器必须设置同一个全新的 `P2_PREFLIGHT_SESSION`。三机实际启动与恢复使用实验目录的 `controller.py`，它自动生成会话并保留节点 supervisor。

这是依赖原项目源码的覆盖层，不是独立可搬走的源码包。覆盖 training.py 的预检查段与 config.py 的源码身份记录，新增 preflight.py；原缓存、模型、损失、优化、保存、TensorBoard 等实现从原项目导入。已有 E1 使用原启动器；此版本训练的检查点使用本启动器生成或恢复。

首次完整审计、后续凭证复用、每 batch 原有 SHA256 检查组成完整流程。不要直接删除预检查或伪造旧源码身份。训练期间保持原始媒体、缓存、模型权重与源码不变。
