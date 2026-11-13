# P2 AV＋Event＋text-attention LoRA＋TensorBoard 固定版本备份

日期：2026-10-26。固定标签：`p2-av-event-text-lora-h200-tensorboard-20261026`。

这是当前训练版本的独立源码快照，来源于 `/private/liqi_workspace/project_code/audio-driven-v5`。与旧标签 `p2-av-event-no-feature-20261026` 相比，新增原生 text-attention LoRA、H200 执行优化和 TensorBoard 监控；旧标签和旧备份目录保留，不覆盖。

## 1. 本版行为与入口

- **训练对照：** E0 用 AV loss 更新 Connector＋LoRA，事件头读取 detach(U)；E1 的 Event loss 进一步反馈 Connector。事件损失不直接反馈下游 LoRA。
- **LoRA：** 48 个原生 LTX block 的视频/音频 text-attention q/k/v/out，共 384 个投影，rank=8、alpha=8；原生基座冻结。
- **无 feature 分支：** 不构建 feature decoder，不读取 teacher，不计算 feature loss；读取 input、VAE、text、events。
- **H200 配置：** 隔层梯度检查点＋32 MiB 梯度通信桶，保持 FP64 累加、全局 SUM 和确定性策略；正式 global batch=8、1000 步。
- **保存和恢复：** 默认每 100 个 optimizer step 保存，正常结束或受控停止补存末步；显式 `--resume` 恢复模型、优化器、步数和各 rank RNG。没有自动重启或自动选择最新检查点。
- **TensorBoard：** 当前 LoRA/H200 配置默认开启，rank 0 记录全局 AV、视频/音频分项、Event、各组学习率、梯度、吞吐和显存；恢复清理超出检查点的旧曲线及 JSONL 尾部。

| 用途 | 文件 |
|---|---|
| 推荐 H200 配置 | [p2_listen_av_event_text_lora_h200_v1.json](snapshot/audio-driven-v5/configs/p2_listen_av_event_text_lora_h200_v1.json) |
| 两 seed E0/E1 协议 | [H200 pair](snapshot/audio-driven-v5/configs/p2_listen_av_event_text_lora_h200_pair_v1.json) |
| 常规全层检查点 LoRA 配置 | [LoRA config](snapshot/audio-driven-v5/configs/p2_listen_av_event_text_lora_v1.json) |
| 训练、缓存、生成入口 | [p2_listen.py](snapshot/audio-driven-v5/scripts/p2_listen.py) |
| 训练与 checkpoint | [training.py](snapshot/audio-driven-v5/src/p2_listen/training.py) |
| LoRA 构建、共享参数归属 | [model.py](snapshot/audio-driven-v5/src/p2_listen/model.py)、[原生投影 LoRA](snapshot/audio-driven-v4/src/omni_av_p21/lora.py) |
| H200 检查点/通信优化 | [performance.py](snapshot/audio-driven-v5/src/p2_listen/performance.py) |
| TensorBoard、全局聚合、恢复清理 | [monitoring.py](snapshot/audio-driven-v5/src/p2_listen/monitoring.py) |
| TensorBoard 使用与指标口径 | [监控说明](snapshot/audio-driven-v5/related/P2_TensorBoard训练监控与恢复.md) |
| 24 卡实测和资源分配 | [GPU 报告](snapshot/audio-driven-v5/related/P2_24卡联合训练GPU验证与提速.md)、[跨机环境](snapshot/audio-driven-v5/configs/p2_listen_h200_multinode.env) |
| 原始实验目的与评测计划 | [实验计划](snapshot/audio-driven-v5/related/P2_倾听事件监督实验计划.md) |
| 实现细节 | [LoRA 说明](snapshot/audio-driven-v5/related/P2_TextAttention_LoRA实现与验证.md) |

## 2. 保存范围

`snapshot/` 保留三个同级目录：

```text
snapshot/
  audio-driven-v5/       P2 训练、相关 P3 工具、配置、测试、说明、示例、轻量验证记录
  audio-driven-v4/       P2 复用源码，包含 omni_av_p21 的 LoRA
  Omni-LTX-stage2/       实际使用的 LTX core/trainer/pipelines 源码及系统 prompt 文本
```

LTX 目录保存相关 Python 包源码、元数据、顶层配置、LICENSE 和上游 uv.lock，不是整个上游仓库镜像。上游 commit 和未提交修改列表见 [external_sources.json](validation/external_sources.json)；快照保存的是实际工作树内容，包含本机修改，不能只按上游 commit 替代。

[BACKUP_MANIFEST.json](BACKUP_MANIFEST.json) 对每个 snapshot 文件记录原路径、大小和 SHA256。`validation/environment.json` 保存实际 Python/包版本及导入路径；上游 uv.lock 只是上游记录，不保证等于本次实际安装环境，不应直接将 `uv sync` 当成完整 GPU 环境恢复。

未上传模型权重、训练检查点、媒体、特征缓存、TensorBoard event 二进制、虚拟环境或凭据。原数据、缓存、GPU 检查点与完整事件文件仍在源机器。快照中的历史文档、验证脚本及总索引可能引用未纳入备份的大型实验目录或原机器绝对路径。

## 3. 已完成的验证及边界

**当前含 TensorBoard 版本：** 原机器 146 项 CPU 测试通过；真实 H200 单卡 E0/E1、双卡 E1 各 5 步，E1 从第 3 步恢复到第 5 步。日志开关前后模型、优化器、RNG 逐位一致；单/双卡模型和优化器逐位一致；主动插入的旧第 4 步曲线被正确清理，实际核对 42 条指标。

**此前 LoRA＋H200 执行版本：** 加入 TensorBoard 前已完成 3 机 24 卡真实 5 步 S0、单/双卡 7 步及恢复、生成检查和性能压测。24 卡、batch=24 的短测约 3.60 秒/步；最长 11.948 秒样本约 114.2 GiB。这些是此前执行实现的性能记录，不冒充本次含 TensorBoard 版本重新跑过的 24 卡结果。

**本次备份副本：** 在副本中重跑 146 项 CPU 测试及两进程 CPU/Gloo 训练、TensorBoard 全局聚合/单 writer 检查；P2/v4/LTX 导入来源均核实为本快照。没有加载预训练模型或重新启动 GPU 训练。副本与原验证 run 中的 70 个源码指纹匹配（按原文件内容验证）。

- [备份验收汇总](BACKUP_VALIDATION.json)
- [备份副本 CPU 测试](validation/restored/unit_tests.json)
- [备份副本训练 DDP](validation/restored/training_ddp.json)
- [备份恢复脚本执行日志](validation/restore_cpu_check.log)
- [当前版本原 GPU 验收](snapshot/audio-driven-v5/experiments/p2_tensorboard_20261026/gpu_verification.json)
- [此前 24 卡证据](snapshot/audio-driven-v5/experiments/p2_lora_gpu24_20261026/summary.json)

尚未开始正式数据效果实验。上述验证不构成“事件监督改善实际生成回应”的效果结论。

## 4. 完整性检查与源码恢复

从本备份目录运行，不需要 GPU 或 ML 依赖：

```bash
python verify_backup.py
```

使用已有训练依赖环境，运行完整的 CPU 恢复检查；脚本隐藏 CUDA、指定本快照的 PYTHONPATH，避免误用原工作目录源码：

```bash
P2_PYTHON=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python \
  bash restore_cpu_check.sh
```

在其他机器上将 `P2_PYTHON` 改为已安装匹配依赖的 Python。默认新证据写入 `validation/recheck/`，可用 `P2_VALIDATION_DIR` 改路径。当前环境包含 TensorBoard 2.21.0，补充依赖见 [requirements-p2-monitoring.txt](snapshot/audio-driven-v5/requirements-p2-monitoring.txt)。恢复脚本不安装或升级环境。

需要在其他位置工作时，复制**整个备份目录**到新的空目录，再运行检查。保持 snapshot 下的三个同级目录；不要只拷贝 audio-driven-v5，因为 `reuse_root=../audio-driven-v4`，且需使用本版 LTX 包。实际运行训练时也应让 PYTHONPATH 指向恢复位置的三个 LTX `src` 目录，或在独立环境中安装这些本地包。

## 5. 恢复源码不等于可直接续训旧 checkpoint

- CPU 恢复检查不需要预训练权重、真实数据或缓存；真实训练需要原运行要求的 Omni/LTX/Gemma 权重和已准入数据。
- 配置中的模型、output、媒体和缓存路径是原机器路径。新实验必须选择自己的 output 并完成数据准入与缓存身份校验，不能直接对着历史实验目录启动。
- 已有 checkpoint 的恢复严格校验源码内容及路径、配置、数据、基础权重、训练 seed、rank 数等。复制到新目录后，绝对路径身份可能变化；不能删除校验来强行继续。
- 继续**本版本已有运行**时，应恢复原绝对路径布局和对应配置、数据/缓存、依赖及权重，在同一运行目录显式指定该运行的完整 checkpoint（`.pt` 加 `.complete.json`）。checkpoint 本身不在 GitHub，需从源机器单独保全。
- 旧无 LoRA 标签、加入 TensorBoard 之前的源码和本标签是不同版本。跨版本不能直接严格续训；需要对应版本源码，或另行设计并验证迁移。
- TensorBoard 读取 `<output>/runs`；保存周期仍由 `training.save_every=100` 控制，监控周期由 `training.tensorboard` 控制。

完整 GitHub 固定入口：
[本标签目录](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/tree/p2-av-event-text-lora-h200-tensorboard-20261026/p2_training/av_event_lora_tensorboard_20261026)。
