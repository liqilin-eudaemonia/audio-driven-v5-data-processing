# ViCo test43 运动补测结果

已完成联合 AV＋Event＋Feature＋speaker Whisper step8000 的补测，独立验收通过。完成时间：2026-11-06 07:10:16 UTC。正式提取／评分／审计总耗时 8.87 分钟；此前的小批核验和开发耗时未包含在内。

这是**公开指标定义复现＋用户确认的明确参数**，不宣称严格复现 CustomListener 表1。原生成512×512／25FPS，评分副本256×256／30FPS；最近时间戳转换产生的重复帧会影响运动差分和相关性。

| 指标 | exp | angle | trans | pose | 覆盖 |
|---|---:|---:|---:|---:|---:|
| FD ↓，×100 | 13.112942 | 10.219366 | 6.493092 | — | 43/43 |
| RTLCC ↓ | 0.225741 | — | — | 0.269646 | 42/43 |
| RWTLCC ↓ | 0.292209 | — | — | 0.332713 | 41/43 |
| FID_delta_fm ↓，×100 | 3.714983 | — | — | 0.084429 | 43/43 |
| SND ↓ | 2.754344 | — | — | 0.120995 | 43/43 |
| V-D ↑ | 0.006784 | — | — | 0.003811 | 43/43 |
| GT 时间方差参考 | 0.021512 | — | — | 0.008256 | 43/43 |

pose = angle3＋trans3；exp64。FD与FID_delta_fm展示乘100，SND仍为原始尺度的FID_fm＋FID_delta_fm，不与已乘100的展示数直接相加。完整原始值、展示倍率和逐片段结果在 [RESULTS.json](RESULTS.json)，表格CSV在 [summary.csv](summary.csv)。

所有指标逐视频计算后等权汇总；Pearson TLCC延迟−60…60帧逐帧扫描，正延迟为listener滞后；RWTLCC使用完整120帧窗口、步长30帧，窗口等权后再视频等权。无系数z-score、无GT拟合、无时间平移。短片段只在不适用的指标标N/A，其他指标保留：

| ID | 长度 | RTLCC | RWTLCC |
|---|---:|---|---|
| YFhkBtIUgBM_000106_000107.0 | 30帧／1秒 | N/A | N/A |
| UCA1A5GqCdQ_001058_001061.0 | 90帧／3秒 | 有效 | N/A |

三路共48,864帧：speaker16,288、GT16,288、prediction16,288。speaker与GT漏检0帧，prediction漏检6帧，全部发生在 `UCA1A5GqCdQ_000353_000361.0`，按官方策略使用上一个有效关键点；未丢弃视频或帧。616帧speaker检出多个候选，按官方策略选择首个，抽查选中前景说话人；GT及prediction无多脸帧。关键点检查图见 [DETECTION_REVIEW.png](DETECTION_REVIEW.png)。

[FINAL_AUDIT.json](FINAL_AUDIT.json) 验证输入／权重／原始源码快照和评分修订SHA、三路帧数、系数切片、逐片段汇总、N/A覆盖、倍率、GT自比较和父结果未改变；独立SciPy协方差积sqrtm距离误差最高 3.47e-08，NumPy corrcoef误差最高 3.77e-15。7项数值回归含解析Fréchet、已知时延、秩亏矩阵、常量、短片段等。

使用node8物理GPU7，4个独立进程。提取阶段GPU利用率平均 84.38%、中位数 98%，显存峰值 26.43 GiB；结束后已释放GPU7。[资源记录](GPU_USAGE.json) · [完成标记](COMPLETE.json) · [协议与复跑方法](README.md)。

原画质与身份五项结果保留：PSNR19.6683、SSIM0.66245、图像FID30.1163、CSIM0.59684、CPBD0.13913；图像FID与本页FID_delta_fm含义不同。[原五项结果](../p2_vico256_combined8000_20261106/RESULTS.json)。
