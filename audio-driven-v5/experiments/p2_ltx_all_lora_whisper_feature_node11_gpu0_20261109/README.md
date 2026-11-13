**GPU正式启动验收通过：** 最长11.95秒样本两次完整更新及初始状态恢复通过；首个31条累积更新完成，四组梯度、损失公式及Event关闭检查通过，峰值116.20GiB。首步244.70秒，输入为31条平均10.02秒长片段，不能直接当作稳定平均步时。见[启动验收](LAUNCH_ACCEPTED.json)。

# node11 GPU0：LTX全LoRA＋Whisper＋Feature，无Event

用户明确授权2026-11-09在node11（172.29.189.119）物理GPU0启动。本目录独立于原31卡配置／输出；其余GPU及已有任务不修改。

## 模型、数据、损失

原生48层attention／FFN均为rank8／alpha8 LoRA，1632处、84221952个LoRA参数；两路新增speaker Whisper attention全参数；Connector、Whisper adapter、双路Feature decoder可训练。Qwen、Gemma、Whisper encoder、VAE及teacher冻结。

`loss = AV + 0.025 video Feature + 0.025 audio Feature`，Event关闭。最终train＋dev284218段／500.001259h，最新完整caption和原始未SFT Qwen H，五项缓存直接复用；不重新打标或抽缓存。

## 单卡执行

直接使用已验收原生入口`scripts/p2_ltx_all_lora_noevent_train.py train`，WORLD_SIZE=1，CUDA_VISIBLE_DEVICES=0。有效global batch31保持，每个optimizer step串行累积31条；样本和noise ordinal与31卡计划相同，但单卡累加／分布式归约的数值轨迹不承诺完全一致。

9169个optimizer step一轮，每500步完整保存模型／Adam／单rank RNG。保留duration496、CPU预取2、GPU Adam、BF16前后向、FP32训练参数、全重计算、134GiB进程显存上限。累积31条时原生执行器使用FP64梯度累积缓冲；没有多卡NCCL通信。单卡step会明显长于31卡；一个step仍表示31条样本。

315个既有绑定源码文件保持不变。本目录config仅更换独立输出路径；原生validate与32条真实缓存preflight通过。第1／2／500／9169步的单卡31条计划与原31卡逐rank计划逐项相同。开始时执行最长样本两次完整反向／Adam检查并恢复初始权重、优化器与RNG，然后保存step0并进入正式训练。

## 状态、保存、停止与恢复

- PID及物理卡：`CONTROLLER.json`；完整启动命令／环境：`COMMAND.json`；日志：`train.log`。
- 运行身份：`run.json`；最长样本验收：`LONGEST_SAMPLE_CHECK.json`；正式步数：`status.json`；loss／梯度／耗时：`metrics.jsonl`；TensorBoard日志：`tensorboard/`。
- 权重在本目录`step_XXXXXX.pt`，同名`.complete.json`发布后才可用于恢复。单卡只有1份rank RNG，不能直接当31卡检查点恢复。
- 写入本目录STOP后，原生训练在下一次更新边界保存退出；一个更新包含31条，因此可能需要等当前累积完成。不要kill其他任务。
- 恢复使用COMMAND.json中相同环境与原生train命令，追加`--resume <本目录完整step检查点>`。运行身份、世界大小及模型／Adam／RNG严格校验；不能从其他消融或旧1500权重续训。

启动配置备份在`launch_snapshot/`，原绑定源码摘要在`SOURCE_IDENTITY.json`。GPU实际验收以`LONGEST_SAMPLE_CHECK.json`和正式更新记录为准。
