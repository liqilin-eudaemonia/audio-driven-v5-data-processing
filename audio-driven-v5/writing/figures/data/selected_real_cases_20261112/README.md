# 两个真实 case 的论文配图素材

## 第一个片段：输入条件

`naturalistic_V03_S1371_I00000205_local_08ad393f3e19`，6.306 秒。

- `case1/speaker_waveform.{svg,pdf,png}`：真实配对 speaker 音轨，有时间轴。
- `case1/speaker_waveform_transparent.{svg,pdf,png}`：透明背景、无文字版本，方便拼图。
- `case1/listener_first_frame.png`：最终修正版裁框后的 listener 首帧，512×512，和已有训练审核导出的首帧一致。
- `case1/speaker.wav`：对应 speaker 独立音轨；不是 listener 或混音。

## 第二个片段：完整数据处理流程示例

`naturalistic_V03_S0557_I00000077_local_c4e0cb6f423c`，6.470 秒；源区间 233.290–239.760 秒；原始人脸短边 p10=414.04 像素。

- `data_pipeline_real_case.{svg,pdf,png}`：沿用用户草图三部分布局的真实素材流程图。SVG文字、箭头、波形、表格可编辑；PDF为矢量排版，PNG为预览。
- `case2/original_frame.png`：原始完整单人视角帧，2160×3840。原始数据是双方各自的视频，未拼造双人镜头。
- `case2/original_frame_with_final_crop.png`：原始画面及实际最终裁框；x=0,y=134,width=height=1181；不改变原始图像比例。
- `case2/listener_first_frame.png` 与 `keyframe_01.png`–`keyframe_05.png`：最终裁框后的真实原生帧，512×512；具体源PTS与相对时间在 provenance.json。不是从压缩预览重新截取。
- `case2/keyframe_strip.{svg,pdf,png}`：连续关键帧素材。
- `case2/listener_before_after.{svg,pdf,png}`：原 listener／净化 listener 同尺度波形，附官方词、句、VAD区间。
- `case2/cue_timeline.{svg,pdf,png}`、`cues_exact.json`：两段点头候选、抬眉、Right?词线索，全部沿用现有结果。
- `case2/caption_verbatim.txt`、`caption_original.json`：完整原文；流程图保留完整原始段落，没有重写或删减动作。
- `case2/event_labels.{svg,pdf,png}`：实际训练Event标签节选；`event_targets_all19.csv`和`event_targets.json`为完整19维记录。
- `case2/listener_before.wav`、`listener_clean.wav`、`speaker.wav`：无损浮点PCM音轨；`listener_clean_preview.mp4`为有声预览，非训练缓存。

## 图中的口径

1. 两条样本均来自最后500h，且同时在Qwen SFT train中；来源由原有冻结20条清单复用，没有重新挑选或打标。
2. 首帧指训练输入的listener首帧。speaker条件是独立speaker音轨；没有把listener误当speaker。
3. 官方词与句均为3.710–3.970秒，VAD另保留4.296–4.900秒。并集后扩边得到3.610–4.070和4.196–5.000秒；10ms淡入淡出，区间外置零，时长不变。gate_exact.json记录精确样本索引。
4. 波形直接由真实采样的有符号最小/最大包络绘制。Before/After共享幅度刻度，未分别放大。第二条门控外原始能量约2.596%，去除部分并不等于已证实的纯杂音，不夸大为强降噪案例。
5. 点头候选为3.440–3.760和4.960–5.720秒；不能据此声称恰好点头两次。抬眉3.480–3.960秒；词线索Right?为3.710–3.970秒。检测线索不是人工时间真值；帧下标签只写实际时间，不为静态图补造动作判断。
6. Event是整段presence监督：本例5个正类为抬眉、点头、头倾斜、微笑和speech；frown为0.1权重弱负例；gaze shift为mask=0，不是确定负例。所有值来自已存训练缓存。mouth closing全局禁用，仅在完整19维表中保留。
7. 一次/反复动作共享一个整段Event标签。图中类别与证据节选保留原文含义，完整caption和完整标签可追溯。
8. 原始图、音频、caption、缓存及用户草图未改；只新增配图素材。来源和SHA256见AUDIT.json与各case/provenance.json。

复现：`/private/software/conda/bin/python scripts/build_data_pipeline_selected_cases_20261112.py`

## S0557 附近一分钟（2026-11-13）

已新增 `case2_context60s/`：源视频206.525–266.525秒，原选pair对应本分钟26.765–33.235秒。提供双方原尺寸视频、并排预览、独立WAV和波形SVG/PDF/PNG（高亮、无高亮和透明背景）。一分钟上下文沿用门控前的双方独立音轨；原case2净化结果不变。页面底部可下载独立ZIP，详见该目录README。
