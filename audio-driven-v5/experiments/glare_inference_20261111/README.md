# GLARE 自训练检查点推理

**常驻批处理优化已完成。** 新入口支持模型常驻、输入缓存、每批8帧解码及导出重叠。短片热推理生成约0.45秒，三条视频含导出共3.52秒，冷启动另计。[批处理命令、数值差异与验收](../glare_resident_inference_20261111/README.md)。


入口是 `scripts/infer_glare_checkpoint.py`，启动封装是 `scripts/run_glare_inference.sh`。使用当前自训练的 best.pt 或 latest.pt，合并同一套冻结 LIA、Wav2Vec2 和 SER 资源。模型参数名称、形状、冻结资产 SHA 和已加载训练参数的数值均会核对。读取正在训练的检查点时先创建临时快照，输出 JSON 保存实际检查点步数和 SHA。

在 audio-driven-v5 根目录执行以下命令。reference.png 应是与训练构图一致的 listener 人脸裁剪，speaker.wav 为 speaker 音频。

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/run_glare_inference.sh \
  --checkpoint experiments/glare_paper_settings_node11_gpu7_20261111/best.pt \
  --reference /path/to/reference.png \
  --audio /path/to/speaker.wav \
  --output /path/to/output.mp4 \
  --seed 137 --nfe 10 --audio-guidance 2
```

输出包含带 speaker 音轨的 MP4，以及同名 JSON 记录输入、权重 SHA、步数、韵律结果、采样配置和生成耗时。脚本要求显式选择已分配的 GPU。示例使用当前授权的 node11 GPU7。

直接复用官方 `FLOAT.inference_from_latent`、`sample`、CFG/ODE 和 `decode_latent_into_image`，默认 Euler、10 个时间点。参考图用官方 LIA 编码一次后供给原生 latent 推理入口。动作统计采用官方训练得到的 null 条件。输入依赖仅为参考图和 speaker 音频。

韵律复用官方离线 `Qwen2AudioAnalyzer`、原 prompt、greedy/max64、normalize_result，以及官方 IntensityEncoder 的逐帧转换。音频读取和响度归一化也调用官方离线函数。按完整2.4秒块处理，官方 NULL 和尾部默认零强度。独立 WAV 从其文件起点分块。需要复用按原长音轨映射的 speaker 强度时，可传 `--intensity-npz /path/to/condition.npz`，其中 intensity 为与当前 WAV 帧网格一致的 `[N,1]` 数组，N 为音频在25fps下向上取整的帧数。

本次必要适配包含以下内容。

- 当前训练保存可训练参数，加载时与冻结预训练资源合并。
- 当前训练采用首帧参考、无历史，因此将标准化整段音频去除前640采样点生成 N-1 帧，视频第0帧保留输入参考图。整段一次生成，按实际长度设置位置编码和 mask，与已训练窗口契约对应。
- 官方统计条件 CFG 的主输入扩成5路，prev_x/prev_wa仅扩成4路。对本模型长度为0的历史张量补齐batch维度后，调用原官方 forward。补丁仅作用于推理实例，官方文件和训练入口均未编辑。
- 官方音频处理器旧参数 audio 与已安装版本 audios 之间做兼容转换，生成设置沿用官方。
- 使用 PyAV 和 ffmpeg 导出视频与音轨，显式按帧数确定时长。

已用第12000步权重、真实2.6秒speaker音频、listener第0帧完成默认端到端验收。输出65帧、512×512、25fps，H.264视频和AAC音轨均为2.6秒。真实韵律输出为官方 NULL，生成张量有限，视频帧已解码检查。空历史CFG回归检查和官方音频读取对照通过。单条短片验收覆盖运行链路，长音频和生成质量尚需后续评估。

- 示例视频为 [smoke_final.mp4](smoke_final.mp4)，三帧预览为 [preview.jpg](preview.jpg)。
- 验收记录为 [ACCEPTED.json](ACCEPTED.json)，生成身份为 [smoke_final.json](smoke_final.json)。
- 官方原始流程见 [generate.py](../../related/glare/generate.py) 和 [README](../../related/glare/README.md)。
- 当前训练及恢复见 [训练文档](../glare_paper_settings_node11_gpu7_20261111/README.md)。
