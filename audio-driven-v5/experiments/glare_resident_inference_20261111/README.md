# GLARE 常驻批处理推理与实测

已实现并验收模型常驻、多输入处理、输入缓存、每批8帧解码，以及视频导出重叠。沿用官方模型、ODE/CFG、LIA编码器和解码器，默认NFE为10。批量解码通过重复参考特征后调用原 LIA decoder 实现。

在 audio-driven-v5 根目录执行以下命令。

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/run_glare_batch_inference.sh \
  --checkpoint experiments/glare_paper_settings_node11_gpu7_20261111/best.pt \
  --jobs experiments/glare_resident_inference_20261111/jobs.jsonl \
  --cache-dir experiments/glare_resident_inference_20261111/cache \
  --decode-batch 8
```

示例 jobs.jsonl 中的输出已存在，实际使用请新建清单并指定新的输出路径。每行为一个JSON对象，包含 reference、audio、output，可指定 seed 和 nfe。

```json
{"reference":"/path/listener_crop.png","audio":"/path/speaker.wav","output":"/path/output.mp4","seed":137,"nfe":10}
```

代码入口是 `scripts/infer_glare_batch.py`，运行实现是 `scripts/glare_resident_inference.py`。GLARE和冻结编码器在整个进程中常驻，Qwen按需加载一次。进程固定同一个检查点，并保存实际步数和SHA。更换检查点需要启动新的实例。

韵律条件以音频内容SHA、提取合同和实现指纹缓存到磁盘，可跨进程复用。进程内最多缓存8个音频输入、16组音频编码结果和2个参考图编码。参考图、投影后的Wav2Vec2特征和SER结果通过原模型计算后复用，缓存绑定当前实例的固定权重。

默认每批解码8帧，可用 `--decode-batch 1` 调用官方逐帧路径。解码与前向保持FP32。导出使用一个CPU线程，最多保留一个待完成的导出任务，与下一条推理重叠。使用 `--serial-export` 可顺序导出。视频完成后才写入对应JSON，批处理结束生成清单同名的 `.result.json`，其中包含模型加载时间、总时间及各条缓存命中情况。

## 实测

实测基于第14000步权重和2.6秒真实输入，在node11 GPU7与训练共卡运行。

- 模型加载约16–19秒，首次新音频还包含Qwen初始化与韵律提取。冷启动仍有明显开销。
- 模型与输入缓存就绪后，逐帧解码的生成时间约0.84–0.95秒，每批8帧的稳定时间约0.45–0.46秒，约2倍。该计时额外包含数值对照所需的CPU张量拷贝。
- 相对官方逐帧输出，批量解码最大浮点差约1.96e-5，平均差约1.5e-7。相同原版重复执行也有约1.2e-5的最大差异。批量解码采用了数值容差验证。
- 三条缓存就绪视频，顺序生成及导出共4.23秒，重叠后3.52秒，约减少17%耗时，平均约1.17秒/条。冷启动时间另计。
- 正式命令行入口完成两个人物、两段真实音频及重复输入，生成帧数分别65/55/65，均有音轨。新进程再次读取已缓存音频，成功跳过Qwen提取。

[RESULT.json](RESULT.json) 记录解码与导出对照，[jobs.result.json](jobs.result.json) 记录三条CLI输入，[jobs_disk.result.json](jobs_disk.result.json) 记录跨进程缓存验证，[ACCEPTED.json](ACCEPTED.json) 记录最终验收。

示例视频为 [cli_a.mp4](cli_a.mp4) 和 [cli_b.mp4](cli_b.mp4)。目前覆盖短片运行、缓存和数值一致性检查，长音频及更大规模吞吐需要按实际输入继续测量。

单条推理命令和必要的空历史CFG兼容说明见 [原推理文档](../glare_inference_20261111/README.md)。
