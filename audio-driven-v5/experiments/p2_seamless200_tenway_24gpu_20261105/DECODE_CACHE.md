# Seamless200：无损GT解码缓存提速（2026-11-05）

本轮原先约7.8秒/条的decode计时包含GT和生成视频两部分；其中GT在不同权重间重复解码。本次只缓存固定200条GT的完整uint8 RGB与原始解码审计，生成视频逐系统独立解码，原指标、采样帧、FPS、最终裁框、GT净化音轨与覆盖率协议均不变。

## 实测和一致性

两条短/长片段（51/297帧）完整GT重新解码3.82/16.00秒，缓存读取0.036/0.190秒。缓存读取包含完整RGB SHA256校验。全部像素、FPS、裁框与PTS审计完全一致，非平凡PSNR/SSIM测试也完全一致，见`DECODE_CACHE_PARITY.json`。

在实际GPU评分器上重放相同两条：GT＋生成视频decode总耗时从14.901/3.175秒降至0.452/0.115秒；PSNR、SSIM、LPIPS、表情指标、可计算的ΔResp，以及GT/生成侧全部FID/FVD特征逐值一致（最大差0）。原来不可计算的动作幅度仍按原规则报告缺覆盖，没有填0或删除。见`DECODE_CACHE_GPU_PARITY.json`。这是两条实测，不能把该倍数当作整轮评分或推理倍数。

## 实现与启用

- `src/p2_testsets_eval/decoded_gt.py`：缓存、校验与执行上下文。
- `scripts/p2_cached_video_metrics.py`：调用原指标CLI的可选缓存入口。
- `scripts/p2_decoded_gt_benchmark.py`：固定池CPU对照及受控预热。
- `src/p2_testsets_eval/metrics.py`：仅在Seamless协议显式配置`metrics_decoded_gt_cache`时采用包装入口。

原OmniResponse reader/scorer文件没有改动。包装器在每份评分contract的`execution.decoded_gt_cache`及独立`decoded_gt_cache.json`中记录自身SHA、来源、命中和像素摘要；原模型/公式代码身份保持一致，已有GT特征缓存仍可复用。既有评分和正在运行的进程保留，后续新worker自动接入。

缓存位于本目录`decoded_gt_cache/`，仅限这200条测试GT，不扫描全量数据。按冻结行、最终裁框、原媒体位置/时间、reader/包装代码、库版本及源archive大小/mtime/ctime绑定。每次命中仍重新验证冻结行、crop配置和现有媒体解析契约；RGB完整SHA256每次读取均核对。源档变化或像素损坏拒绝旧结果。源档变化前后的两次stat检查避免边读边改。构建使用互斥锁与临时目录，完整后原子发布，重复请求只解码一次，命中不持排他锁。

缓存为只读mmap的无损`.npy`，保留原始RGB，无视频再编码；仅用于评分，不作为训练输入。不新增WAV。200条预计约24GB，可按`DECODE_CACHE_PREWARMED.json`核对实际字节数；删除后可重建。CPU预热4线程，不占GPU。

17项相关CPU回归通过：缓存命中、数据变更失效、跨线程单次构建、失败不发布、损坏像素拒绝、包装卸载，以及原调度/汇总检查。真实GPU回归已通过，再启用协议选项；`DECODE_CACHE_ENABLED.json`记录生效时间。全200条预热及后续生产命中仍以各状态文件为准。

## 首批正式生产验证

已在正式评分worker中观测到无损缓存读入，不只是在对照脚本中运行。命中/未命中的逐条耗时见`DECODE_CACHE_HIT_TIMINGS.json`；预热期间未命中仍走原解码并发布缓存，因此全窗口平均包含冷构建成本，不能等同全命中耗时。
