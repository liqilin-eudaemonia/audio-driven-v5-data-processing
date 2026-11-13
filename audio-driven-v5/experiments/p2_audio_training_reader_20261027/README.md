# 后处理音频训练读取验证

实现入口：[新训练版本](../../releases/p2_audio_postprocess_v1/README.md)。保留原冻结训练/推理版本和全部音频缓存。

- `real_validation.json`：64条，50h/全量各16条changed＋16条reuse；video/audio latent、mask、新事件载荷与实际缓存一致，无teacher和波形重处理。
- 两个按数据源命名的JSON：分支覆盖与原未准入样本拒绝记录。
- `preflight_validation.json`：两种视图各32条真实预检查、第二次复用，无重复全量读取；统计与新event tensors一致。
- `gpu_read_validation.json`：8条CUDA读取（两来源、两音频分支各2条），latent和events与CPU逐元素一致。
- QA视图只使用已有缓存与媒体引用；未启动任何新训练，未写新WAV或重算latent。
