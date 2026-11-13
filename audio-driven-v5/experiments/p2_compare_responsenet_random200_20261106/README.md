**已全部完成并独立验收（2026-11-06）：** DyStream固定ResponseNet200/200条生成及全部8项视频指标完成，44,046评分帧，视频／指标执行错误0。node11仅使用0–6卡，7卡未使用；接续耗时393.64秒，0–6卡已释放。node8的139条原输出与评分副本全部保持原SHA，node11补齐61条；两条无脸检测仅跳过未参与推理的诊断遮罩，干净首帧及LIA路径与正常实现逐值一致。原ResponseNet评分代码、资产及关键参数完全一致，无帧数裁短。Learning2Listen按用户要求仍暂缓。

[结果表](dystream/RESULTS.md) · [完整数值及覆盖](dystream/RESULTS.json) · [最终独立验收](dystream/node11_extension/FINAL_ACCEPTANCE.json)。PSNR16.4107、SSIM0.64069、LPIPS0.34168、FID17.5310、FVD184.4808、Emo-Acc55.52%、Emo-Div3.080（GT3.345）、ΔResp0.23861。

以下为部署和运行过程记录。

**已转接node11的0–6卡（2026-11-06）：** 用户要求保留7卡；使用7张H200、每卡4路生成，共28进程，后续每卡2路ResponseNet评分。保留node8的139条有效输出，仅续跑剩余61条；旧控制器因两条MediaPipe首帧检测失败退出，历史错误与未完成产物已归档在`dystream/node11_extension/previous/`。备用路径跳过不参与模型输入的诊断遮罩，仍只用同一干净首帧及同一LIA编码器；真实图像／latent逐值一致，跨机试跑MP4及评分副本SHA256与node8相同。权重、seed、采样、512／原FPS评分不变。资源授权与实时状态见`dystream/node11_extension/RESOURCE_EXTENSION.json`、`LIVE_ACCEPTANCE.json`和`dystream/pipeline_status.json`。

**本轮提速已实测（2026-11-06）：** 已从4路增加到12路，仅node8 GPU7；显存约113/144GB。生产窗口吞吐：4路17.46、8路24.31、12路31.18原生帧/秒（12路窗口120秒，较4路约1.79倍）。片段时长随队列变化，属于生产观察，不宣称同样本严格加速比。新增进程共享原子领取队列，不重复生成；权重、seed、精度、原始逐帧渲染及评分协议不变，自动评分接续保留。批量渲染虽更快但有1灰阶取整差异、CUDA Graph捕获失败，两者均未用于正式结果。新增进程在剩余24条未领取任务时停止领取并完成手中片段，原4路收尾并交接评分，避免破坏已有控制器的完成检查。[资源与吞吐验收](dystream/CAPACITY_ACCEPTANCE.json)。

# DyStream：ResponseNet分层随机200条

本轮仅运行DyStream；Learning2Listen按用户最新要求暂缓。

输入是`p2_responsenet_random200_twelveway_20261105/references/references.jsonl`中的固定200条，1504.605秒（25.08分钟）。references SHA256：`8279171c39501a644c75c36b2e5fe110036270a28d9f38f0e097503ba2c61146`。CPU输入验收200/200：首帧与历史模型完全相同，speaker波形与已有conditioning一致，listener自身输入音频严格全零。

资源限定node8（172.29.104.237）物理GPU7；不占0–6卡。后台pipeline先真实试跑，成功后4个模型常驻进程共享领取长片优先队列；生成结束自动4路评分、全200特征汇总与审计。

原生生成512×512／25fps、seed137、官方5步及既有CFG；不二次裁脸。仅speaker音频和listener首帧，不读取listener未来画面、音轨或GT描述作为生成条件。评分副本保持512，按最近时间戳映射到各GT的原始帧率（23.98／24／25／29.97／30）和精确帧数，尾部官方少一帧及音视频终点差单独记入每条complete.json，不改变播放速度。

评分调用原ResponseNet `evaluation.video.paired_metrics.run`，保留GT PIL bicubic512、face-padding64、batch64、GPU人脸与表情、既有GT特征缓存及默认FID/FVD取帧策略。覆盖PSNR、SSIM、LPIPS、FID、FVD、Emo-Acc、Emo-Div、ΔResp。FID/FVD从全200条特征重算，不平均分片。该方法没有生成listener音轨，音频指标不适用，原生MP4中的speaker背景音不冒充listener生成音频。

- 入口：`compare/deployment/responsenet_dystream.py`
- 后台PID：`dystream/PIPELINE.json`；阶段：`dystream/pipeline_status.json`
- 日志：`dystream/pipeline.log`、`dystream/worker_00.log`至`worker_03.log`
- 生成进度：`dystream/status.json`；资源：`dystream/gpu7.jsonl`
- 视频：`dystream/generated/<id>/generated.mp4`（原生），`dystream/scoring/dystream/<id>.listener.mp4`（评分）
- 评分进度：`dystream/pipeline_status.json`；分片：`dystream/metric_shards/`
- 最终结果：`dystream/RESULTS.json`、`RESULTS.md`、`FINAL_AUDIT.json`、`COMPLETE.json`。这些文件尚未出现时不代表已完成。

恢复时必须先核实后台是否仍活着；pipeline.lock防止重复调度。失败应查看PIPELINE_FAILURE.json与对应日志；不要启动第二份相同队列。
