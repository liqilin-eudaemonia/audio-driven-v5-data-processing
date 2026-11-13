# 六条 listener 杂音处理试验（2026-10-26）

用户指定四条有杂音样本、两条干净对照。仅生成试听副本，未修改训练输入、标注、缓存或正在执行的任务。

## 查看

服务端页面：`http://127.0.0.1:8876/listen_benchmark_review/audio_cleanup_6/`。
通过 Cursor 的 Ports 面板转发 **8876**，用当前本地地址打开该路径；不要沿用过期的 63034 端口。

[试听页面](../../data_previews/listen_benchmark_review/audio_cleanup_6/index.html)、[全部结果元数据](../../data_previews/listen_benchmark_review/audio_cleanup_6/review.json)。

每条可比较原音、温和降噪、完整降噪和视听提取，支持同一时间切换与“移除部分”试听。音频为 FLOAT WAV；不逐版本归一化音量。原始 speaker 音轨另列，仅供辨别干扰来源。反馈按样本和方法保存在浏览器，可导出 JSON。

## 实际运行的方法

1. **实验性线性参考消除**：读取同源 speaker/listener，使用前后各最多 8 秒上下文。16 kHz 相关估计 ±500 ms 延迟，48 kHz STFT 4096/hop1024，speaker 活跃且 listener/reference 能量比低的帧校准，交替 0.5 秒块做拟合/验证，逐频相关门控。6/6 未过可靠性门槛，均返回原音；不能当作有效消除，也不能据此排除串音。不是专业 AEC 的替代品。
2. **DeepFilterNet3**：官方预训练权重，48 kHz、上下文输入、`pad=True` 补偿延迟、关闭 post filter。温和档 `atten_lim_db=12`，完整档 `None`。前者将原信号部分混回，仍不保证保留弱回应。
3. **AV_MossFormer2_TSE_16K**：ClearVoice 0.1.2 官方 HF 权重，1,308 个模型张量全部与 checkpoint 匹配。输入 listener 训练裁剪视频＋原始 listener 音轨（不混入 speaker），25 fps，S3FD 人脸框、13 帧中值平滑、cropScale=.4、224 灰度中心 112、均值 .4161 / 标准差 .1688，官方 3 秒滑窗解码。依照官方峰值归一化输入，再精确还原该输入增益；输出 16 kHz 重采样到 48 kHz 仅供对齐，不恢复高频。检测短缺口线性插值，主框覆盖要求≥90%。S0193 的头发出现一处微小误检，已检查图像；最大框需≥次框4倍才允许选择，否则失败。官方人脸检测权重从其 Google Drive 链接下载；不使用最初不兼容的 face-alignment 权重。

这是一轮试听试验。只完成数值、媒体与网页检查，**尚无人类试听结论，不声称任何方案已经可靠保留所有 backchannel**。通用降噪可能保留干扰说话声，同时抑制 listener 的轻声或笑声；AV 方法也不能仅凭模型类型认定有效。

## 数值结果

整体 RMS 改变不是 SNR 改善，也不是保真度评分。没有干净真值，不能用这些数值给方法排序。

|样本|用户反馈|温和降噪 RMS 变化|完整降噪 RMS 变化|视听提取 RMS 变化|
|---|---|---:|---:|---:|
|S1855|有杂音|-0.97 dB|-1.31 dB|-0.08 dB|
|S0300|有杂音|-3.87 dB|-4.86 dB|-1.78 dB|
|S0193|有杂音|-4.76 dB|-6.88 dB|+0.02 dB|
|S1252|有杂音|-3.40 dB|-4.87 dB|+0.02 dB|
|S0866|干净对照|-0.23 dB|-0.31 dB|+0.02 dB|
|S1921|干净对照|-3.55 dB|-5.14 dB|+0.02 dB|

S1921 是用户确认的干净对照，但完整降噪改变整体能量约 −5.14 dB，需要特别检查回应是否被削弱。视听提取在多数样本上的整体能量几乎不变，也不能据此证明已去掉干扰。

“移除部分”定义为 **原音 − 输出**，可能包含相位变化、重采样带宽差异和失真，并非纯噪声。原始事件只作定位参考，未重新打标或修正类别。

## 复现与文件

- 准备/线性参考：[listen_audio_cleanup_trial.py](../../scripts/listen_audio_cleanup_trial.py)
- 预训练模型处理：[listen_audio_cleanup_models.py](../../scripts/listen_audio_cleanup_models.py)
- 页面生成：[build_audio_cleanup_review.py](../../scripts/build_audio_cleanup_review.py)、[模板](../../scripts/audio_cleanup_review.html)
- `cases/<sample_id>/` 保存每个方法 WAV、参数、上下文与原始标注。
- `deps/` 为隔离依赖；未升级共享环境。DF 只加了已移除的 torchaudio 元数据类型兼容层，音频 I/O 使用 soundfile。
- `model_manifest.json` 保存官方来源和权重 SHA256。
- `verification.json`：原始文件及源路径 SHA256 一致、48 kHz、样本数匹配、数值有限。
- `av_checkpoint_verification.json`：全部参数加载匹配。
- `browser_check.json`：24 个主试听版本可解码播放、音画同步、同时间切换、残差播放、反馈导出、手机无横向溢出、Range 206。

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
python scripts/listen_audio_cleanup_trial.py prepare --run experiments/listen_audio_cleanup_6_20261026
python scripts/listen_audio_cleanup_trial.py reference --run experiments/listen_audio_cleanup_6_20261026
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/experiments/listen_audio_cleanup_6_20261026/deps" /private/software/conda/envs/qwenomni/bin/python scripts/listen_audio_cleanup_models.py df --run experiments/listen_audio_cleanup_6_20261026
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD/experiments/listen_audio_cleanup_6_20261026/deps" /private/software/conda/envs/qwenomni/bin/python scripts/listen_audio_cleanup_models.py av --run experiments/listen_audio_cleanup_6_20261026
python scripts/build_audio_cleanup_review.py
```

AV 命令保留已有成功输出，失败样本可重跑。要改参数重测，应另建运行目录，以免混用不同实验结果。

## 原始音轨来源实测核验

已对这六条样本的双方共 12 条音轨追溯并重算，全部通过（`source_provenance_check.json`）。读取的是下载 tar 内同参与者 ID 的独立 `.wav`，不是 MP4 内嵌音轨。记录中的 `contract.source.member` 虽保留 `.mp4` 参与者索引名，但 `p3_listen.media.uri(source, ".wav")` 会定位同包 `.wav` 成员。

从独立 WAV 重新执行记录中的整轨响度增益、限幅和 48 kHz 输出，与已有 normalized_audio 解码样本逐点一致；按原区间裁切并复现 PCM16 量化后，与本轮原始 listener/speaker 片段逐点一致。文件哈希也匹配。官方 README 将独立 WAV 标为 speaker-bleed denoised audio，但这不保证每段没有残留串音。

这六条属于 Seamless，不能套用 RealTalk 混音分离的来源描述。HTML 的“原始 listener”是进入本轮降噪试验前的规范化裁切版，并非未经任何幅度处理的官方整轨。
