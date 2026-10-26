# 虚拟音频与增量缓存：固定100条验收

2026-10-26。按用户要求，先完整运行固定100条并逐条验收。全量处理和训练均未启动。

- [实现与命令](../../related/P2_虚拟静音音轨与音频缓存增量更新.md)
- [独立100条配置](view100/config.json)
- [本轮逐条验收结果](acceptance100.json)、[验收日志](acceptance100.log)、[验收脚本](check_all100.py)
- [缓存运行记录](view100/audio_postprocess/worker-all-0.json)、[耗时记录](cache100_timing.json)
- [100条波形数值等价](waveform_equivalence.json)
- [前一轮4条真实编码验证](real_encoder_check.json)
- [26项回归测试记录](unit_tests.log)
- [原冻结抽取文件哈希](frozen_code_before.json)

## 结果

100/100条音频缓存及事件缓存已就绪，实际训练读取与断点复用全部通过。98条更新音频缓存，2条处理波形未变化，精确复用原音频缓存。

100条内存处理波形与已试听实体WAV逐采样点一致。98条更新后的音频latent与实体WAV重新编码逐位一致；另外2条验证原缓存原样复用，并独立验证虚拟波形与实体WAV在同一CPU编码器上的结果逐位一致。跨设备重编码不作为原缓存必须逐位相同的条件。

保留63个音频事件：38条有发声标注、62条无发声标注；53条波形全静音。没有发声标注不等于整段音频全静音，仍按已确认的官方词／句／VAD保留区间处理。音频标签原文和有效区间与已复核版本完全一致；六类视觉事件及训练目标不变。

新增音频payload共4,133,972字节（约3.94 MiB），未新增WAV，未复制或重算视频缓存。input/text复用原目录，视频latent从原VAE包读取。原媒体、原canonical标签和原缓存均通过哈希检查，没有改写。

本轮补齐96条缓存并检查100条的命令耗时约72.8秒；随后100条物理WAV对照、训练读取与续跑验收约77.3秒。仅为此批CPU实测，不作为全量吞吐承诺。全程未初始化CUDA，未占用其他训练任务GPU。

## 复验

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
P2_PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
"$P2_PY" scripts/p2_audio_postprocess.py cache \
  --config experiments/listen_audio_virtual_validation_20261026/view100/config.json \
  --part all --device cpu --cpu-threads 2
"$P2_PY" experiments/listen_audio_virtual_validation_20261026/check_all100.py
```

缓存命令会跳过已完成的音频编码；验收脚本会重新编码实体试听WAV作对照。`source_before/`保留接入功能前的训练读取与P2 CLI源码。本轮只验证缓存和训练数据读取，未执行训练前向、反向或生成评估。
