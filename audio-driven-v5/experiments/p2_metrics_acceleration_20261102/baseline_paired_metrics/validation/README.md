# 验收记录（2026-11-01）

此页为首版历史验收；2026-11-02 独立复核发现并修复 FVD 缩放及 DeepFace 归一化顺序差异，最新情况见 [复核报告](AUDIT_20261102.md)。下方分数及 contract 对应当时版本，不覆盖、不冒充新版结果。

这是一组实现正确性的 smoke test，不是生成模型质量评测，也不能作为论文的两样本 FID/FVD 排名。

11 项 CPU 回归测试通过：

```bash
/private/software/conda/envs/AnyTalker/bin/python \
  -m unittest evaluation.video.paired_metrics.test_core -v
```

真实模型输入为两段已裁剪的 listener 视频，各取开头 32 帧，保持 25 fps，缩至 224×224，编码 mp4v。来源为实际训练检查页样本：

- `improvised_V01_S0227_I00001226_local_2262c9e23394`
- `improvised_V03_S0209_I00000582_local_5ffe6778447f`

源目录：`/private/liqi_workspace/project_code/audio-driven-v5/data_previews/listen_benchmark_review/j_face224_training_20261101/media/<sample_id>/listener.mp4`。

原始运行文件和测试媒体：`/private/liqi_workspace/.cache/paired_video_metrics/smoke_20261101/`。本目录保存每次的输入／代码／权重 SHA256、逐段结果和汇总，测试媒体不加入 Git。

| 指标 | 同视频自对比 | 亮度＋左右抖动扰动 |
|---|---:|---:|
| PSNR | inf | 18.0432 |
| SSIM | 1 | 0.46237 |
| LPIPS | 0 | 0.06804 |
| FID | 0 | 18.99779 |
| FVD | <1e-8 | 308.13470 |
| Emo-Acc | 1 | 0.96875 |
| Emo-Div generated / GT | 1 / 1 | 1.5 / 1 |
| ΔResp | 0 | 3.46330 |

扰动是在编码后的 GT 帧上，每帧交替平移 −4/+4 像素（反射边界），亮度加 25 后裁到 uint8，再编码。两组的共同有效表情帧均为 64/64，运动有效相邻帧为 62/62。Emo-Div 在扰动后增大不代表更好，恰好说明它应对照 GT 而不是单向最大化。

1. [self_v2/summary.json](self_v2/summary.json)：全部真实权重自对比；退出码 0。
2. [perturbed_v1/summary.json](perturbed_v1/summary.json)：扰动敏感性；退出码 0。
3. [no_face_v1/summary.json](no_face_v1/summary.json)：生成端替换为全黑，表情与 ΔResp 全部为 null，0 个共同有效帧，partial，退出码 2；未把检脸失败算作 0 运动或 neutral。

在 CPU 上运行，Torch、OpenCV、ONNX Runtime 和 TensorFlow 的线程数均限制为 2；没有启动全量评测或占用训练 GPU。检查到 InsightFace 0.7.3 的顶层 get_model 丢弃 sess_options，已改为直接调用 ModelRouter 传递线程设置，最终三次实测未出现线程亲和性错误。

`perturbed_v1` 在最终补充版本记录字段前执行，其 contract.json 保留实际运行时代码哈希；self_v2 与 no_face_v1 使用最终代码版本。三次使用相同权重和指标公式。
