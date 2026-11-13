**执行已结束并独立验收（2026-11-06）：** DyStream在固定Seamless200条上生成、评分及全量FID/FVD汇总完成；30,100评分帧，只用node11 GPU6，总耗时28分01秒。画质五项均200/200：PSNR15.4589、SSIM0.66241、LPIPS0.38292、FID23.2626、FVD243.2780。情绪与ΔResp覆盖199/200：同一条片段与历史模型具有完全相同的有效支持不足记录，保留N/A，因此结果状态为partial。全部200条GT RGB摘要与历史评分相同，GT无损缓存200次命中，未裁短评分帧，全部原生及评分视频已验收。Learning2Listen未运行。

[完整结果](dystream/RESULTS.md) · [逐项覆盖](dystream/RESULTS.json) · [独立验收](dystream/FINAL_ACCEPTANCE.json)。

以下为运行记录。

# DyStream：Seamless固定200条

仅使用node11（172.29.189.119）物理GPU6；其余卡不使用。Learning2Listen按用户要求暂缓。

冻结清单：`../seamless_quick200_latest_20261105/prepared_200/references.jsonl`，SHA256 `d4bda633eefebfd0b0a1b055684a749d86a6b28ec7c9d1a032e4a151b0468b07`；200段／1199.3653秒（19.99分钟）。独立test子集，已冻结最终512裁框、25fps、speaker音轨和净化GT配方。推理不使用GT caption、listener未来画面或listener音轨。

真实调用原有Seamless conditioning reader，逐条重新取得speaker WAV和listener首帧，首帧PNG及完整输入审计与历史模型缓存完全一致；自身音频分支为严格全零。权重、seed137、官方5步、CFG及FP32原逐帧渲染与已验收DyStream流程相同，关闭额外裁脸。首帧人脸检测失败只跳过不参与模型输入的诊断遮罩，沿用真实图像与LIA编码逐值一致的备用路径。

12路共享领取生成，8路视频评分，仅同一张GPU6。生成512×512／25fps；按精确GT帧数导出512／25评分副本，官方少帧只在尾部延展，原生输出和每条映射均保留。无listener生成音频，音频指标不适用。

评分使用原Seamless PSNR、SSIM、LPIPS、FID、FVD、Emo-Acc、Emo-Div、ΔResp，face-padding0、batch64、原默认FID/FVD取帧；复用已验收GT特征缓存及逐次验证RGB SHA的无损解码缓存。FID/FVD从全200条特征统一重算，覆盖不足与N/A保留，不删除样本。

- 运行入口：`compare/deployment/seamless_dystream.py`
- 后台PID：`dystream/PIPELINE.json`；进度：`dystream/status.json`、`prepare_status.json`
- 日志：`dystream/pipeline.log`、`dystream/logs/`；资源：`dystream/gpu6.jsonl`
- 原生输出：`dystream/generated/<id>/generated.mp4`；评分副本：`dystream/scoring/dystream/<id>.listener.mp4`
- 最终结果：`dystream/RESULTS.json`、`RESULTS.md`、`FINAL_AUDIT.json`、`COMPLETE.json`；未生成这些文件时仍未完成。

不要并行启动第二份相同队列；pipeline.lock及领取锁避免重复任务。异常时保留已完成产物，先核实旧进程再恢复未完成claim。
