# GLARE 点头候选：此前复核的 15 段

2026-10-29。最初计划使用现有 50 段；用户要求先提供刚复核的十几段，因此本轮仅处理其中 15 段（9 个关键分歧、6 个随机对照），没有继续扩展其余 35 段。目录名沿用初始 50 段试验名。

[打开页面](http://localhost:8876/listen_benchmark_review/prompt_versions_20261029/glare_nodding/)，在 Cursor 转发远端 **8876**。原 prompt 对照及 fps 复核页均已加入入口。左侧视频和同步音频，右侧三组候选、原生视频 8／16 fps 的完整原文与窗口结果；可切慢速、点击时间、筛选与导出复核。

## 运行与结果

| 参数 | 有候选的片段 | 候选区间 | 区间总时长 |
|---|---:|---:|---:|
| Python 默认 | 0/15 | 0 | 0 秒 |
| README 平衡 | 8/15 | 8 | 3.72 秒 |
| README 高召回 | 13/15 | 16 | 10.92 秒 |

15/15 成功，CPU 两个并发任务，OpenCV/BLAS 单线程；本轮计算墙钟约 15.4 秒（不含环境安装、页面构建、验收）。MediaPipe CPU 后端自行管理内部线程；未占用 GPU。14 段逐帧检出率 100%；S0284 为 96.15%。这是头部运动候选统计，**不是准确率、召回率或独立点头次数**。默认无检出说明这组参数未能提供候选，不能证明没有点头；放宽后更多候选也不能证明更准确。

优先复核：

- S0333：仅高召回 3.08–3.52 秒，判断短运动是否为点头。
- S2128：平衡 1.04–1.40 秒，高召回扩大到 0.44–1.88 秒，判断是否将姿态变化过度扩展。
- S0343：高召回两段 0.92–1.48、1.80–2.20 秒，核对往复运动与连续姿态变化的区别；不沿用旧 VLM 的周期数判断。
- S1668、S0339：三组均无候选，核对是否仍漏检。

以上是复核问题，不是已经观察确认的判断。尚未形成可替换生产标签的准确性结论。

## 输入与算法

样本来自 `../listen_visual_input50_20261029/native_fps_decision/feedback/human_review_20261029T103511Z.json` 的 15 个唯一 ID。反馈只选择样本，不填写检测结果或调整单段参数。

视频复用 `../p2_features_full_crop384_20261025/media/<sample_id>/video.mp4`，逐条检查 ID、裁框、时长、512×512 尺寸、25 fps 和帧数。不扫描原始语料或解包源视频。三组共享同一份逐帧姿态，不是分别以 8／16 fps 检测；页面中的 8／16 fps 指已有 VLM 结果。

直接调用本地 GLARE 标注仓库：

- `related/visual_reaction_annotation/nod_headshake/face_detection/detect_face_multithread.py`：MediaPipe FaceLandmarker，逐帧 pitch/yaw/roll、关键点和人脸检出状态。
- 同目录 `face_landmarker_v2_with_blendshapes.task`：已有模型资产。
- `related/visual_reaction_annotation/reaction_detector/nodding/run_nodding_intensity_mt_v2.py`：去趋势俯仰、速度/加速度、偏航抑制、滞回阈值与运动约束。
- 三组完整参数、源代码和模型 SHA256 见 `protocol.json`。平衡／高召回取自原仓库 `README_nodding_params.md`；默认取 Python 参数，未使用有参数传递问题的 shell 启动器。

原算法已门控后的正分数连续帧组成半开候选区间；时间为帧号÷实际 fps，末尾裁到源片段时长。原始帧级结果不裁改。强度按各片段归一化，不是概率；速度单位为度/帧。默认要求至少两次方向转折与一个算法定义的周期，可能排除短促、单次运动。降低门槛可能误收低头、抬头、姿态调整；三组候选并不保证严格嵌套。缺失姿态由原算法插值，页面保留检出率并提示含缺失帧的区间。

**不使用六类互斥的 `combine_reactions.py`**，避免抹掉同时发生的微笑和点头。原 VLM 原文、时间、音频后处理配方、生产标注均不改动。本轮不新增训练标签或声称 ±0.2 秒边界已验证。

## 代码和复现

```bash
# 独立环境，避免修改训练环境；本轮具体版本见 requirements.freeze.txt
/private/software/conda/envs/qwenomni/bin/python -m venv /tmp/glare-nod-pilot-env
/tmp/glare-nod-pilot-env/bin/pip install mediapipe==0.10.21 opencv-contrib-python==4.11.0.86 tqdm

cd /private/liqi_workspace/project_code/audio-driven-v5
/tmp/glare-nod-pilot-env/bin/python scripts/run_glare_nodding_pilot.py --workers 2 --reviewed-only
python scripts/build_glare_nodding_review.py
FONTCONFIG_FILE=/tmp/native-fps-fonts.conf /tmp/native-fps-review-qa/bin/python scripts/check_glare_nodding_review.py
```

每段成功结果由协议与视频哈希检查后可续用；失败单列，不计为无点头。`samples.json` 固定本轮名单，`results/` 包含元数据、姿态、三组帧级分数和候选；`landmarks/` 保存原始逐帧 PKL；`scores/` 保存原算法 CSV；`summary.json`、`run.log` 为运行记录。

`scripts/glare_nodding_review.html`、`scripts/glare_nodding_review.js` 是页面源文件；复用 `listen_visual_prompt_review_audio.js` 保持原始 listener／处理后 listener／双人参考音轨同步，不新增 WAV。

`audit.json` 已通过：候选重建与时间界限、缺帧标识、CSV 长度、输入视频哈希、VLM 原文完整一致；15 段视频和双方音轨 HTTP206；实际 Chromium 原始／处理后／双人音频播放、慢放与跳转同步；反馈保存与导出、手机宽度、无 JS 错误。自动化反馈仅位于临时浏览器上下文，不作为人工反馈。
