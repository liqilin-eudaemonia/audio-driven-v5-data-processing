# P2 弱背景监督 GPU 验收：2026-10-25

状态：第一轮训练、断点续跑和独立生成成功，但数值验收未全部通过，不能作为最终通过结论。恢复状态精确，重复 forward 的事件 loss 漂移约 8e-6；单/双卡最大参数差 4.43e-5，超过 2e-5。记录保留，FP32 修复后从头复测见 [新目录](../p2_listen_weak_background_gpu_fp32_20261025/README.md)。用户授权在 `172.29.104.220` 后四张 GPU 上叠加测试；实际使用 GPU 4/5/6，GPU 7 初始显存余量不足以稳妥容纳已有 S0 配方，本轮未使用。不停止其他任务。

数据：上一轮 CPU 接入验证得到的 50 条真实样本弱标签。来源为 `../p2_listen_weak_background_validation_20261025/derived/`，策略为 `event-interval-weak-background-v1`。六类有有效正/背景支持，摇头因无正例禁用。全批仍为 debug，不能据此开始正式训练或声称生成质量提升。

## 验收范围

- E0 与 E1 相同初始化、50 条样本、有效 batch 8、训练 seed 20261027，各 7 步；覆盖全 50 条。
- E1 在第 5 步停止，恢复至第 7 步；另外从第 6 步重放第 7 步，按已有 GPU 数值容差比较模型/优化器状态、RNG 与 forward loss。
- GPU 4/6 双卡 NCCL E1，各 rank 4 条有效 microbatch，总 batch 8，共 7 步；与单卡 E1 对照参数、样本序号、全局有效 bin 数。
- 新 E1 checkpoint 独立生成一段 512×512 音视频；文件访问审计禁止读取 listener 目标视频/音频、训练标签、teacher/VAE/events 缓存；随后检查实际解码、帧率、时长及有限音频值。
- 所有训练均不带 `--debug-positive-events`。事件覆盖和背景来自新派生标签，没有使用旧 positive-only 特例。

## 文件与复现

- `config.json`、`ddp/config.json`：单/双卡配置。除输出位置外一致；`save_every=6` 方便保存恢复与相邻步重放。
- `run_gpu_checks.py`：调度脚本，检查显存余量，顺序衔接对应卡上的任务；每 10 秒更新 `job_status.json`。已有 run 不能直接重复追加，复现需新目录。
- `validate_results.py`：逐项验收和汇总脚本；生成 `summary.json`、`ddp/comparison.json`、`generated_source_only/media_validation.json`。
- `preflight.json`：50 条样本缓存身份、源码 SHA256、远端环境。
- `E0.log`、`E1_first5.log`、`E1_resume.log`、`E1_ddp.log`、`replay.log`、`generate.log`：实际命令日志。
- `runs/s0_20261027/`：新的单卡 E0/E1 权重和训练日志；`ddp/runs/s0_20261027/E1/` 为双卡结果。
- `replay.json`：恢复重放数值验证；`generated_source_only/` 为生成音视频与目标读取隔离记录。
- `protected_before.json`：原标签、旧 S0 汇总及本轮新标签/事件缓存的保护哈希。

input/teacher/vae/text 缓存通过身份验证后读取既有真实 GPU 缓存；标签及 events 缓存独立复制，历史调试 run 保持原样。没有重新执行标注模型或重新提取不变的特征。

本轮是工程 S0，不能代替正式 E0/E1、speaker audio_swap 与生成适当性评价。
