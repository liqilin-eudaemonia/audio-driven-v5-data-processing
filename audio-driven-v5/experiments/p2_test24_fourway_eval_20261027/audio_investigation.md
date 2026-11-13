# 生成近静音与训练音频监督核查

确认：四组训练均监督listener视频和独立listener音频。实际AV目标为视频flow-matching MSE＋音频flow-matching MSE，两项系数均为1；Feature两路0.1为另外的辅助目标，未替代音频AV loss。

## 实际训练证据

四组metrics.jsonl均有1268步非零loss/av_audio。抽查原训练随机24条，四组VAE缓存合同中的target_audio与listener目标清单一致，与speaker输入不同；audio_latents非空，音频support每条至少63个有效token。抽查目标RMS中位数0.0266866（约−31.47 dBFS）。这些是24条缓存抽查，不宣称重新审计了30415条全部目标。

| 组别 | 第1268步视频AV loss | 第1268步音频AV loss |
|---|---:|---:|
| E0 | 0.362817 | 0.634797 |
| E1 | 0.362725 | 0.635132 |
| E0F | 0.362655 | 0.632957 |
| E1F | 0.363058 | 0.632884 |

## 文件与播放

抽查四组MP4均包含48kHz双声道AAC音轨；独立listener.wav为直接vocoder输出裁剪所得。低电平在独立WAV内已经存在，因此不能归因于仅未封装音频。原HTML单个video有声音控制；“四组同步播放（静音）”按钮将四个video.muted设为true，用户之后单独播放该video时需手动取消静音。此次按用户要求未改HTML。

| 集合 | 组别 | 输出数 | RMS中位电平(dBFS) | 低于−80dBFS数量 |
|---|---|---:|---:|---:|
| 原train24 | E0 | 48 | -90.92 | 43 |
| 原train24 | E1 | 48 | -87.85 | 39 |
| 原train24 | E0F | 48 | -78.21 | 23 |
| 原train24 | E1F | 48 | -89.62 | 44 |
| 独立test24 | E0 | 48 | -90.93 | 43 |
| 独立test24 | E1 | 48 | -88.39 | 40 |
| 独立test24 | E0F | 48 | -79.27 | 23 |
| 独立test24 | E1F | 48 | -90.57 | 46 |

−80dBFS只用于描述波形电平，不是语音检测阈值。未对生成结果增益归一化或替换文件以掩盖近静音现象。

## 目标缓存解码对照

选取抽查中目标音频电平最高的一条（原RMS0.159517）。CPU float32重建RMS0.157475；GPU float32重建0.157448；GPU实际bfloat16路径重建0.157306。所测目标在缓存与GPU解码过程中没有塌成静音。CPU bfloat16测试因vocoder float输入/bfloat16权重不匹配报错，已原样保留；它不是实际GPU推理路径，不能据此宣布GPU解码有错。

## 同样本、同seed生成控制

下列单样本使用原30步、视频CFG3、音频CFG7、固定Gemma prompt、首帧及seed137。text_only为原生LTX、无训练Connector/LoRA且无speaker条件；remove_u保留E0F已训练LoRA但移除U；四组normal输出为已完成train24评测。

| 路径 | RMS | dBFS |
|---|---:|---:|
| text_only | 0.00022646 | -72.90 |
| remove_u | 0.00034677 | -69.20 |
| E0 | 0.00004454 | -87.03 |
| E1 | 0.00005072 | -85.90 |
| E0F | 0.00042163 | -67.50 |
| E1F | 0.00003556 | -88.98 |

原生权重text_only也出现低电平，因此近静音并非仅在训练后出现。该对照仍走本项目采样链路，尚未等价验证官方LTX pipeline，不能据此排除自定义采样问题。这不能单凭一条样本判定根因；固定prompt、CFG/采样与条件强度等仍需受控区分。尚未证明“训练不足”、Event loss或某一模块是原因。

## 对评测结论的影响

现有指标实现仅将RMS<1e-5判为静音不适用；本批很多近静音输出高于该门槛，仍会产出UTMOS/SyncNet数值。数值成功计算不代表产生了可听辨的回应，不能用这些分数宣称音频回应质量或时机得到改善。训练集报告中此前的UTMOS下降信号也仅是这些输出上的指标差值，不足以判定有效发声的音质变差。

下一步应先做固定输入与seed下的prompt/音频CFG及原生对照，确认可听辨回应，再讨论规模训练或把音频分数作为质量结论。本次没有修改训练代码、权重、生成协议或已评分输出。

证据：[训练缓存与日志核查](training_audio_supervision_audit.json)、[全部384条生成WAV电平](generated_audio_audit.json)、[GPU重建](audio_decode_gpu_audit.json)、[生成控制](audio_generation_controls.json)。
