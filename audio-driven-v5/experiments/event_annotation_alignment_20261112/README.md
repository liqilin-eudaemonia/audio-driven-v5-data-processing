# 评测检测与数据标注流程对齐（2026-11-12）

用户确认以复用数据标注pipeline为目标，不针对这批小样本人评另调阈值，也不要求继续补齐人工标注。

## 当前执行口径

- 视觉：同一最终裁框；25fps MediaPipe pose/blendshape；复用标注中的 nod recall 参数、摇头方向门控、FER＋expression_gate_v3；8fps视频送入同一Qwen3-Omni模型。
- 本轮补齐标注流程的抬眉候选，复用原 surprised detector 与 assess_surprise，不添加新阈值。该线索只作为“brows raised”辅助观察，评测输出仍是原视觉4类＋音频3类。
- 候选观察名称直接使用生产 `listen_j_corrected_reaction.OBSERVATION`：head shaking、brow lowering、brows raised；点头为 head nodding。不把眉部运动直接解释成内部情绪。
- 发声候选继续复用生产 `normalize_asr_cue_for_english_prompt`；只传规范化提示，原ASR保留审计。微笑曲线继续传给Qwen，不加二值微笑阈值。
- 移除评测独有的“人脸检测覆盖率低于100%就将所有absent强改unknown”后处理；覆盖率仍作为观察证据，最终present/absent/unknown由实际媒体判断。媒体技术失败仍记录error。
- 整段类别存在性、七类定义和Macro-P/R/F1统计不变；不使用训练Event头或已有caption代替生成媒体检测。

## 用途不同的接口适配

标注输出有序自然语言描述；评测输出七类结构化状态。评测的视觉／音频仍分别观察，以免从画面微笑推断笑声。参考使用已处理listener音轨，生成使用自己的完整波形，不按GT区间净化或静音；生成素材没有官方转写，ASR来自自身波形。这些是评测接口适配，不声称两个流程逐字节相同。

## 版本与验证

默认配置 `configs/p2_event_presence_v1.json` 的detector为v3；prompt为 `annotation-pipeline-presence-visual4-audio3-20261112-v3`，候选为 `annotation-visual-candidates-v3`。新增surprised源码进入冻结身份。

23项事件／提示回归通过。另用一条生产标注中含抬眉线索的真实转正素材做CPU候选检测，并回读候选到提示，结果见 `VALIDATION.json`。本轮不重跑GPU大模型或全量指标，不声称检测准确率已提高；已有v2结果及人工标签保留，不覆盖成v3。

关键代码：`src/p2_event_presence/candidates.py`、`hints.py`、`prompts.py`、`inference.py`、`media.py`。修改前备份见 `before/`。
