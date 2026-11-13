# 规划SFT新H接入AV＋Event训练

本次完成独立训练backend、配置、来源校验及CPU读取/全量preflight。已按用户要求挂上自动接续，等待现有AV＋Feature完整结束后使用四机32卡启动。最终状态见运行目录 `READY.json`。

## 数据与模型

- 仍使用当前500h、285,638条train；最终listener修正版裁框、512×512视频、官方时间并集净化listener音频均由原AV缓存复用。
- Qwen为已完成2,697步规划SFT的Qwen2.5-Omni-3B，冻结后提取的第8/18/28/35层H，维度2048。LoRA文件和SHA绑定在新H协议及训练data_identity中。
- 新H位于 `experiments/p2_j_train500h_sft_hidden_20261102/cache/input_plan/`。Qwen输入仍是配对speaker音频、listener最终裁框首帧和原始打标描述；不重新生成规划文本。
- Gemma文本、视频/净化listener音频latent、support布局及clip Event标签复用原始500h缓存。
- 本组为冻结已SFT的Qwen H训练LTX/Connector，配置仍是stage1；不是在线继续微调Qwen的stage2。

## 训练规格

配置：`configs/p2_j_sft_h_av_event_20261102.json`。

`L = L_AV + 0.05 * L_Event`。Event保持现有clip多标签方案：16类视觉＋3类音频；不是早期七类，也不是逐帧事件时间监督。Feature关闭，不读取teacher特征payload。Event更新Connector及Event heads；AV更新Connector和LTX文本attention LoRA。

Connector 1024维、4层、16头；global batch32、8,927步、相同seed及学习率。duration512从第1步启用，每500步保存并保存最终步；three_quarters梯度重算、BF16、32MB通信桶沿用现有实现。本组应从原始LTX和新初始化的Connector/Event head开始，不能严格续训旧H的AV＋Event权重。

## 独立入口与安全边界

- `src/p2_j_sft_event_backend.py`：独立backend，验证新H协议、最终完成记录、LoRA SHA、原J数据来源及成员；每条读取核对Event来源、H身份和原文一致性。
- `scripts/p2_j_sft_event_train.py`：调用原Matrix训练循环，安装原duration采样器。使用它，不要使用仍读旧H的 `p2_j_duration_train.py` 或 `p2_clip_events.py`。
- 原 `p2_clip_events`、`p2_matrix16`、缓存提取代码均未改动。当前AV＋Feature训练source_identity检查仍完全一致。
- `plan_cache`和`sft_hidden_cache`都指新H目录，`j_cache`仍指原AV缓存。训练身份额外绑定新H protocol/COMPLETE和adapter文件哈希，避免静默读回旧H。
- 全量preflight检查新H及复用AV/text_guided的285,638条来源标记和文件存在；不重复读取全部2.66TB payload或重提取特征。此前独立缓存审计已完成SHA及回执验收，实际读取仍执行load_part完整校验。

## 运行和验证

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
$PY scripts/p2_j_sft_event_train.py preflight \
  --config configs/p2_j_sft_h_av_event_20261102.json --processes 16
# 分布式torchrun应调用：
# scripts/p2_j_sft_event_train.py train --config <上述配置绝对路径>
```

运行目录 `experiments/p2_j_sft_h_av_event_ready_20261102/` 保存 `cpu_tests.xml`、`reader_check.json`、`preflight.json`、`launch_commands.json`。四机训练命令尚未执行，`executed=false`；后台接续控制器已在等待，会在启动时更新此记录。

35项CPU回归通过。真实首/中/末3条：H与新缓存逐值相同、与旧H确实不同；布局、原文Gemma及AV/support与原包逐值相同；保留19类Event标签，不读Feature teacher；核对最终裁框。完整GPU反传、实际峰值显存及步速仍需开跑前做短步验收，当前不能宣称这部分已通过。

## 最终CPU接入验收

`experiments/p2_j_sft_h_av_event_ready_20261102/READY.json`已写入：35项测试、3条真实读取、285638条全量preflight全部通过（约93秒）；配置/源码/数据身份完全一致，Event Dataset顺序与duration采样顺序一致。独立代码、配置、测试和本文快照在运行目录 `source_snapshot/`，附SHA256清单。GPU短步完整反传尚未执行，将在前一组完整结束后自动执行，通过后正式续跑。

## 已挂上自动接续

控制器为 `scripts/p2_j_sft_event_queue.py`，集群配置 `configs/p2_j_sft_h_av_event_32gpu_20261102.json`。控制器PID、命令、日志和状态均位于本实验 `queue/`，读取 `queue/state.json` 即可确认。

1. 每15秒跟随当前AV＋Feature目录的 `CURRENT_RUN.json`，等待正式预算完成、controller complete及四机退出码全0；不以达到步数或GPU利用率低作为结束。
2. 检查8927步完整检查点及SHA，再确认四机8卡均无compute进程、每卡空余显存至少135GiB。不终止其他任务。
3. 再核对新实验preflight的来源、数据身份及duration计划。使用新初始化开始训练，前3步属于8927步预算。
4. 检查3步完整断点、AV＋0.05 Event算式、Connector/Event/LTX LoRA梯度及Feature确实关闭；通过后从第3步严格续跑至8927步。
5. 保存间隔500步，最终步另存。短步验证会额外保留0和3步完整检查点，便于验收与恢复；其余为500、1000……及8927。

完成条件和算式门控新增2项CPU测试通过（`queue_tests.xml`），此前35项接入测试仍保留。控制器定期检查指标进度；启动600秒无指标、或训练中180秒无新指标，会仅结束自己启动的进程并写STALL/failed，保留完整检查点，不伪报完成、不跳样本。故障不会自动绕过检查强行启动下一阶段。

取消排队可在新实验根目录创建 `STOP`；停止前一组或前一组失败不会启动后组。本次没有恢复此前已停止的20分钟聊天汇报。
