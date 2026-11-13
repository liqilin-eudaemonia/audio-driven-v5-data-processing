# 全attention／FFN LoRA6000：ViCo正式test43

**已全部完成：43条推理、86个分辨率副本的完整解码检查、官方5项、运动补测和DIM十二项均完成；16,238个DIM共同有效帧，原代码16项数值对照误差0。相关评测进程已退出，其他GPU7任务保持运行。** [完整结果](RESULTS.md) · [视频与音轨](VIDEOS.md) · [交付复核](FINAL_USER_REVIEW.json)。

训练权重：`../p2_ltx_all_attention_ffn_lora31_20261108/step_006000.pt`。本轮已取得用户对三组指标的确认，仅node11（172.29.189.119）物理GPU7，不停止其他任务。

原完整权重SHA256 `3c2c78e8b1c81e3cf84efdc565b16a28565c0e923cacf8e902c3adcd6232eea6` 已重新核对；使用前次逐张量一致、去掉Adam的12.6GB推理导出，SHA256 `594670fb897a4d8ebc6d4ad1e3477084150e9390e3b4d53d9e894e4bc06af82e`。282个训练来源文件相符，真实GPU严格加载1632处LoRA目标已通过。完整训练权重保持不变。

## 固定设置

- 正式test43，保留独立listener UUID，不含OOD；单次seed137。
- 复用此前ViCo SFT2752的规划文本、H、Gemma、speaker Whisper及listener首帧缓存，不重新规划，不读取未来listener音视频或GT描述作为生成条件。
- Euler30、video CFG3／audio CFG7；原生512×512／25fps，float RGB抗混叠bicubic缩小到256×256，最近时间戳转换为30fps评分。
- 原生生成和VAE解码分进程，避免DiT与解码器同时驻留；4个固定、互斥样本分片按显存准入并行解码，CPU CPBD与解码重叠。保留16GiB资源余量，不使用其他卡。
- 首次探测GPU7另有约47GiB任务；本轮生成预算约77.27GiB，真实模型加载成功。显存准入随当时资源变化，已完成输出按提交标记保留。

## 指标范围

1. 官方画质／身份5项：PSNR、SSIM、FID、CSIM、CPBD，沿官方脚本和原汇总方式。
2. CustomListener风格运动补测：原BFM提取器，FD、RTLCC、RWTLCC、FID_delta_fm、SND、V-D；原lag±2秒、完整4秒窗口／1秒步长和短片段N/A规则不变。
3. DIM表情／姿态12项：指定EMOCA_v2_lr_mse_20；FD、P-FD、MSE、SID、Var、rPCC分别计算Exp／Pose。主Var采用用户已指定的逐系数总体方差平均，发布代码的拉平方差另存。单次采样，不做GT择优；同覆盖重算旧SFT2752参考。

BFM和EMOCA复用已核验的GT／speaker特征，只重提取43条新生成视频。公开定义／公开代码复现与作者私有协议仍有差别，不将本轮写成论文原表的严格复现。用户未选择AF SPECTRE，不接入该组或七类事件pilot。

## 入口与产物

- 推理与官方评分：`scripts/p2_vico43_all_lora6000_gpu7_20261111.py prepare|run`。
- 运动与DIM评分：`scripts/p2_vico43_all_lora6000_metrics_20261111.py run`，由主控制器自动接续。
- [CPU准备验收](CPU_ACCEPTANCE.json) · [真实模型加载](MODEL_ACCEPTANCE.json) · [冻结协议](protocol.json) · [启动回执](LAUNCH.json)。
- [控制器阶段](status.json)；latent阶段该文件是阶段快照，实际完成数以`latents/all_lora6000_sft2752/vico/*/complete.json`及[render进度](progress/render/00.json)为准。
- 原生视频：`generated/all_lora6000_sft2752/vico/<id>/generated.mp4`，配套`listener.wav`。
- 评分副本：`scoring/all_lora6000_sft2752/<id>.listener.mp4`，逐帧转换记录在`conversion/`。
- 评分结束后：[总结果](RESULTS.md)、[完整JSON](ALL_RESULTS.json)、`IMAGE_RESULTS.json`、`bfm/RESULTS.json`、`DIM_RESULTS.json`、[最终验收](FINAL_AUDIT.json)、[完成标记](COMPLETE.json)。没有完成文件时不得将启动或部分产出写成已完成。

本轮不创建或恢复旧Table4复现Goal。人工巡检沿用户偏好约5分钟一次，仅在完成、错误或需要处理的变化时汇报。
