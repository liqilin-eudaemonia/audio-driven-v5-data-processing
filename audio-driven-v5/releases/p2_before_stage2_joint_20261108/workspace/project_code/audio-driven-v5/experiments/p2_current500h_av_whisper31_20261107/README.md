**九小时专项巡检已完成，训练继续运行（2026-11-07 23:48 UTC）：** 最终实查1609步、四机31rank均正常，最近200步平均5.99秒／步；Whisper500／1000／1500完整模型＋Adam＋31RNG均验证通过。全量实际loss／梯度、源码和最终500h／最新caption数据身份复核通过，无训练故障或持续降速。九小时observer按期退出，常规训练监控和TensorBoard继续运行；node11 GPU7预留，每500步保存。[逐项最终验收](../p2_current_ablation_guard9h_20261107/FINAL_ACCEPTED.json) · [实时步数](status.json) · [TensorBoard](http://172.29.252.37:6009)。

**已实际启动并通过31卡验收（2026-11-07 21:08 UTC）：** AV＋Feature精确停在6000，完整模型／Adam／31份RNG与SHA复核通过，旧31rank均已退出；本AV＋Whisper消融从原始预训练初始化启动。31rank最长11.95秒样本反传和超过20步真实更新通过，纯AV loss及Connector／text LoRA／Whisper三组有效梯度通过。前30步约5.7秒／步，峰值94.50GiB；持续监控。node11 GPU7保留；每500步保存。[父训练完整保存](PARENT_SAVED.json) · [运行验收](RUNNING_VERIFIED.json) · [实际切换验收](../p2_current_ablation_guard9h_20261107/HANDOFF_ACCEPTED.json)。请勿重复手动launch。

**再次代码复核通过（2026-11-07）：** 36项CPU测试通过，新增双进程真实梯度归约的参数／Adam误差均为0；speaker有效内容影响两路输出、masked token无影响、冷CLI入口通过。源码／数据身份未变，未修改训练逻辑。GPU实测仍待启动。[复核说明](RECHECK.md) · [验收](RECHECK_ACCEPTED.json)。

# 最终500h：AV loss＋speaker Whisper 消融

已在AV＋Feature第6000步保存退出后实际启动。沿用原Qwen＋Whisper双路条件AV pipeline，替换为最终500h及对应最新完整caption。Whisper是条件分支，训练仍只有视频与音频AV loss。

- 数据：284,218段／500.001259126小时，train 281,060＋dev 3,158，test 0；与当前AV-only、AV＋Feature使用相同最终selection、样本顺序、Qwen/Gemma/AV缓存。
- 文本：最后冻结的完整caption原文，不做截断、正则重写或旧描述回退。原始未做规划SFT的Qwen2.5-Omni-3B hidden保持冻结；Gemma与Qwen都绑定这批caption。
- speaker条件：冻结Whisper-large-v3 encoder缓存，FP32／1280维；双路独立残差注意力保留时间RoPE，分别注入LTX视频和音频分支。使用配对speaker音频，非listener目标音频；静音不裁剪，保持共同时间原点。
- 目标：`L = L_video_AV + L_audio_AV`。Event、视频Feature、音频Feature系数全部显式为0；不读取Event标签、listener teacher，不构建Feature Decoder。
- 训练参数：Connector、LTX text-attention LoRA、Whisper adapter及注意力残差分支。Qwen与Whisper encoder冻结，LTX主干冻结。原始初始化及seed20261023，不继承其他消融训练权重。
- 执行：四机31卡（node12／node8／node4各8卡，node11仅0–6卡，GPU7留推理）；global batch31、duration496、完整一轮9169步，尾部补齐21条；真实样本/noise ordinal与其他31卡消融一致。每500步及最终步保存完整模型、Adam、31份RNG。
- 提速：沿用稳定all全重计算、CPU预取2、512MiB streaming FP64梯度SUM归约及GPU Adam，进程上限134GiB。独立rendezvous端口29857。
- 缓存：277,676条历史Whisper复用＋6,542条已完成增量，不重新提取。

## 代码与验收

原实现：`src/p2_whisper_av/model.py::QwenWhisperAV`；本实验直接复用其构造、forward、prepare、参数所有权及优化器分组。`native_view`仅为兼容旧校验器提供历史保留的未激活权重字段；正式实验spec中所有辅助损失权重均为0，forward只返回AV。

最新Dataset只打开Qwen/Gemma/AV及混合speaker Whisper读取器。旧data_config保留以锁定原架构，实际extractor配置来自当前AV protocol，训练成员来自当前cache index与最终sample_list；旧train-only标记不筛除最终选定的dev。

- `tests.xml`：新实验及原Whisper pipeline共33项CPU测试，通过原生小尺寸LTX多步更新、完整重计算、三组梯度、原初始化与前向逐值一致、模型/Adam/RNG精确恢复、禁止辅助loss及31卡停止参数传递。
- `preflight.json`：32条真实缓存回读，覆盖train/dev、新旧Whisper，检查完整文本、音视频目标与speaker时间/维度/掩码。
- `WHISPER_AUDIT_FULL.json`：全量核对284,218条speaker身份、区间、复用观测合同／新增合同及payload存在；运行时仍逐payload检查SHA。
- `FULL_CAPTION_AUDIT_REUSE.json`：绑定相同selection及三套缓存协议和完整验收收据，复用已完成的284,218条caption实际文件／Qwen与Gemma合同全量审计；本轮未重复重读全部caption。
- `PREFETCH_VERIFIED.json`：实际CPU预取线程读取train/dev和Whisper，独立reader不共享SQLite连接。
- `CURRENT_TRAINING_UNCHANGED.json`：当前AV＋Feature训练所冻结源码未变。
- `PREPARATION_ACCEPTED.json`：本轮最终准备验收。

本实验GPU最长样本反传、显存、31卡联合更新和长期稳定性仍待分配卡后实测。正式入口先进行所有rank的最长样本反传，检查完整损失及梯度有限，再开始更新；本次不占用训练卡。

## 入口

使用 `/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python`，`PYTHONPATH=src:scripts`。配置为config.json，四机命令为COMMANDS.json。

```bash
python scripts/p2_current500h_av_whisper.py preflight --output experiments/p2_current500h_av_whisper31_20261107
python scripts/p2_current500h_av_whisper.py commands --output experiments/p2_current500h_av_whisper31_20261107
```

明确分配资源后才运行：

```bash
python scripts/p2_current500h_av_whisper.py launch --output experiments/p2_current500h_av_whisper31_20261107 --logs experiments/p2_current500h_av_whisper31_20261107/rank_logs
```

31卡短测可加`--stop-after 2`，会传递给全部rank。STOP用于当前更新完成后完整保存退出；只接受本实验身份一致且带完整标记的检查点恢复。
