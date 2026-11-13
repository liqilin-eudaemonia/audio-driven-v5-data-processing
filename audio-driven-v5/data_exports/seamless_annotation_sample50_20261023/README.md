# Seamless 标注代码测试包：随机 50 条

从训练主池音质通过候选 1,200,212 段中均匀无放回抽取 50 段，固定种子 2026102301。没有按画面、人脸、裁框质量二次挑选；本包不包含专门抽取的拒绝样本，不能用于估计误拒率。

入口为 `items.json`（数组）或 `manifest.jsonl`（每行一条）；所有媒体路径均相对于本 README 所在目录，整包复制即可使用。`samples.csv` 为精简表格，`labels_template.jsonl` 为待填标签。

每条有 speaker/listener 各一份完整画幅片段 MP4、归一化单声道 PCM_24 WAV，以及边界前后各最多 3 秒的完整画幅 context MP4。MP4 内音频为对应人的归一化音频 AAC；精确音频测试请使用 WAV。没有叠加框或文字，也没有执行空间裁剪，适合打标程序自行绘框。

`roles.listener.clip_video` / `roles.speaker.clip_video`：切分片段，从本地 0 秒播放。`roles.*.audio_wav`：同区间音频。`roles.*.context_video`：前后上下文，`segment_in_context_s` 标出真正切分区间。`source_start_s/source_end_s` 是原始视频时间，不能直接用作本地片段时间。截取边界因视频帧离散最多约一帧误差；音频按 round(原时间×采样率) 截取，精确采样边界在 media_checks 内。

`crop_proposals.listener` / `speaker` 为本片段固定参考框；x、y 为对应完整画幅左上角原生像素坐标，size 为正方形边长。视频保持原生编码宽高，禁用自动旋转和缩放，因此坐标可直接绘制。缩放显示时应同步缩放坐标。参考框只针对正式片段生成，不保证覆盖额外上下文。status=proposal_only 不代表最终验收，unmeasured 时无可用坐标。fits_estimated_head 是几何估计，valid_fraction 是有效帧占比；尚未做人脸裁框准入。旧 listener_crop=null 字段不用。

每条 `original_record.json` 保存完整原清单字段（含原 TAR、成员名、偏移、归一化整轨音频路径、转写、角色、边界标记、人脸统计等）；`quality_evidence.json` 保存对应片段 QC3/QC4 证据。上述原始绝对路径仅供溯源，移机测试使用 items.json 中的相对媒体路径。转写是源数据文本，未经新增人工校正。

`sampling.json` 记录随机种子和来源行号；`validation.json` 记录媒体解码、尺寸、时长检查；`SHA256SUMS` 可用于复制完整性校验。`export_sample.py` 是本次导出脚本存档，重新导出需要原服务器依赖。
