> 2026-11-09更新：本页的31＋1自动接续安排已被用户提出的24＋8布局取代，旧接续控制器已撤下；本页其余内容保留为历史准备记录。新布局尚未启动。

# step6000后交换完整组与无Event组的GPU资源

用户授权：完整组到step6000保存后，无Event组从node11 GPU0迁移到四机31卡；完整组从step6000迁移到node11 GPU0继续训练。两组均续训各自已有权重和Adam，不重新初始化，不互换模型。

## 安排与入口

- 原完整组：`../p2_ltx_all_attention_ffn_lora31_20261108/`；到6000精确保存退出。
- 原无Event组：`../p2_ltx_all_lora_whisper_feature_node11_gpu0_20261109/`。完整组接近5900时先请求它在一次31条累积更新完成后保存，提前准备迁移，减少31卡等待。
- 新无Event31卡：`../p2_ltx_all_lora_whisper_feature31_resume_20261109/`。node12/node8/node4 GPU0–7，node11 GPU1–7；从原单卡实际保存的step继续。
- 新完整组单卡：`../p2_ltx_all_attention_ffn_lora_gpu0_after6000_20261109/`。node11 GPU0，从6000继续。

原生LTX全部attention／FFN的rank8 LoRA、双路Whisper attention全参数、数据、最新caption、原始Qwen H、损失权重、学习率、预取、全重计算和有效batch31均保持。单卡每次串行累积31条，因此一个optimizer step会明显长于31卡。沿用每500步保存和原9169步预算。

## 迁移保护

`gate.py`独立轮询完整组；看到6000后用独占创建写入原生STOP，不覆盖已有STOP。原生6000检查点完整保存及旧31rank退出后，无Event31卡立即启动；完整组的CPU检查点迁移与它的启动重叠，随后启动GPU0完整组。

两个目标使用独立输出目录。严格检查父运行身份、配置、源码、数据和统计；完整核验检查点SHA及模型／Adam有限性；迁移后逐Tensor比较模型与Adam完全相等，step不变。只改变输出路径、world及启动RNG布局。31→1保留父rank0启动RNG，1→31将单rank启动RNG复制到目标ranks；原生训练每条样本按不变ordinal重新设置全局与生成器种子。所有9169步的样本／noise计划已逐项验证world1与world31一致，数值轨迹不承诺逐bit一致。

worker在导入原生模型前明确绑定CUDA设备。无Event31卡沿用稳定单网卡、512MiB FP64 SUM和GPU Adam；node11 GPU1–7保持108GiB上限，其余134GiB。完整组GPU0使用134GiB，单卡累积使用原生FP64缓冲。不会终止其他GPU任务，GPU不足或观测不完整时等待。

## 验收与状态

7项CPU检查通过，包括两种世界大小的真实原生小模型／Adam检查点迁移及恢复、禁止覆盖原检查点、batch31的累积／分区SUM及下一次Adam更新对照、6000停止门和外部STOP保护。两组各32条真实缓存回读、两组全部9169步计划与原282／315源码身份均通过。

GPU验收使用两组接续后的真实训练更新：无Event组至少3步、完整组单卡至少2步，核验有效梯度、损失公式、Event关闭／开启及实际GPU对应，再发布`SWAP_ACCEPTED.json`。在该文件发布前不能将排队成功当作切换已完成。

- 原定配置和授权：`PROTOCOL.json`；CPU准备：`CODE_READY.json`／`tests.xml`。
- 控制器：`CONTROLLER.json`／`STATE.json`／`controller.log`。
- 独立6000门：`GATE_CONTROLLER.json`／`GATE_STATUS.json`／`GATE_STOP_REQUESTED.json`。
- 权重迁移：目标目录的`MIGRATION.json`及`run.json`。
- 当前阶段检查：`ARMED_ACCEPTED.json`；实际切换验收：`SWAP_ACCEPTED.json`。
- 本目录STOP取消后续交换；两个原父目录STOP保留，不自动清除用户或其他所有者的STOP。错误写入`ERROR_<pid>.json`，已经健康运行的另一组不会被连带停止。

不修改正在运行的282／315绑定源码。代码、CPU证据及父配置备份在`../../releases/p2_ltx_resource_swap6000_20261109/`。
