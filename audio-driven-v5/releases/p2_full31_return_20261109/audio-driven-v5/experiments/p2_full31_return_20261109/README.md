**已按用户要求立即停在6428：** 完整保存、SHA及模型／Adam／31RNG核验通过，31个训练worker退出；6500预约取消，小时观察器停止，无后续训练启动。后续恢复应使用 `../p2_ltx_all_lora_full31_resumed_20261109/step_006428.pt`，不得盲目执行原6208迁移启动命令。

# 当前node11完整组返回四机31卡

用户澄清：续训的是正在node11运行的LTX全LoRA＋Whisper＋Feature＋Event完整组，沿用冻结Qwen hidden条件；不是在线Qwen联合微调。最新完整保存为step6208。正式接续验收以 `ACCEPTED.json` 为准。

父输出：`../p2_ltx_all_lora_full8_after6000_20261109/`；新输出：`../p2_ltx_all_lora_full31_resumed_20261109/`。迁移保留全部模型和Adam张量、step6208；原8份RNG保留在父checkpoint，新31rank采用父rank0启动RNG，后续每条样本按原ordinal独立设种子。全9169步的sample/noise pairs逐条核对保持一致。

配置：node12/node8/node4各GPU0–7，node11 GPU1–7，共31卡，node11 GPU0预留。既有vLLM保留，node11进程上限108GiB，其余134GiB。沿用已验证四机单网卡mlx5_0；三机八网卡实测不外推到曾超时的四机拓扑。

仍为global batch31，每卡1条；原全attention/FFN rank8 LoRA、Whisper两路全参数、AV＋整段Event＋双路Feature、原学习率、all重计算、最终500h及最新caption、每500步完整保存均保持。去掉旧8卡多微批CPU梯度累积路径，恢复GPU streaming FP64 SUM。各rank先做两次最长11.95秒真实更新，再严格恢复6208模型/Adam/RNG才开始正式6209步。

`runtime31.py`、`training.py`、`canary.py`、`worker.py` 是独立派生代码，原训练源码不改。`control.py prepare-migrate`创建并逐值验证迁移；`control.py launch`用于本次6208启动，不应在已有后续进度时盲目重跑。后续恢复应使用新输出的最新完整checkpoint并保持当前来源身份。

正常观察使用 `watch_hourly.py`，每3600秒；只读，不终止训练或其他进程。完整启动验收后替换旧双组观察器。

无Event batch24实验已按用户要求中止：仅完成同744样本测速和660状态迁移，没有正式batch24更新。两份无Event660完整checkpoint保留，其测试结论不影响本组训练配置。

## 本次验收

已从6208实际推进到6217，首9步平均15.66秒；31个训练worker和保留的vLLM核对通过。两次最长样本更新峰值94.6367GiB，随后严格恢复6208才进入正式更新。小时观察器已替换并完成首次检查；代码、配置及证明归档于 `../../releases/p2_full31_return_20261109/`。
