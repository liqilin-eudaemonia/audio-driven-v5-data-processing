# 整段事件类别一致性

协议于2026-11-08由用户确认。本模块对应[事件级评测框架](../../writing/事件级评测框架_当前确认版.md)和[完整设计](../../docs/superpowers/specs/2026-11-08-event-presence-evaluation-design.md)，复用当前标注pipeline检测最终GT／生成媒体。它是现有视频、GLARE时间匹配、声音活动与音质评分之外的独立阶段。

**当前实现用于pilot，尚不能宣称正式benchmark检测器已经校准。** 代码、CPU边界检查、原始模型响应与真实执行证据分别记录，不把自对照或程序成功当成人工可靠性结果。正式执行状态与验证证据见本页末尾。

## 固定类别

| 模态 | ID | 定义 |
|---|---|---|
| 视觉 | nodding | 点头；不凭静止歪头或单向低头推断点头，不推断赞同 |
| 视觉 | head_shaking | 摇头；不把单次侧向转头当作摇头，不推断反对 |
| 视觉 | smiling | 微笑／笑态；不要求有声，不能只因张嘴就判笑容 |
| 视觉 | frowning | 可观察到的皱眉／眉部收紧降低，不等同于心理悲伤或愤怒 |
| 音频 | audible_laughter | 可听笑声／轻笑；不从视觉笑容或ASR单独推断 |
| 音频 | brief_verbal_response | 独立简短词语／短语，例如yeah、okay、right、no、wow |
| 音频 | nonlexical_response | 非词汇回应，例如mhm、uh-huh、hmm、uh、ah、oh；排除笑声、普通呼吸、咳嗽、噪声 |

允许同一片段出现多种类别，但一个uh-huh不能因ASR拼成两个英文词就同时算短词和非词汇。持续完整长句另记long_speech辅助标记，不自动计入短词类；确有独立短回应才计入，难判为unknown，不凭空设秒数／字数阈值。抬眉、眨眼、视线变化不纳入本次主类别。

## 指标

每片段／每draw／每类只判一次“存在”。同类出现一次或五次、发生时间与参考不重叠，存在性计数相同。不按次数、持续时间、GT事件IoU加权。

对每一类别，在完整测试清单上累积TP、FP、FN，再计算：

```text
P_c  = TP_c / (TP_c + FP_c)
R_c  = TP_c / (TP_c + FN_c)
F1_c = 2 TP_c / (2 TP_c + FP_c + FN_c)
C_m  = 该模态冻结参考标签中有至少一个正例的类别
Macro-P / Macro-R / Macro-F1 = 分别在 C_m 上取算术平均
```

视觉和音频分别汇总，另列全部七类的参考支持、预测正例、计数和覆盖；可附Micro（同一主类别集合内计数），不以Micro代替Macro-F1。没有参考正例的类不进主宏平均，**仍必须报告其FP和误报率**。因此主Macro-F1不能独自惩罚所有额外类别，须结合逐类误报及语境评价解释。

- 支持类完全不预测时P/F1为0；完全不可评时N/A。全部参考无事件时主Macro为N/A，保留误报，不填1。
- 双空不产生TP，不通过大量安静样本抬高F1。
- 标签状态是present、absent、unknown或error。遗漏／未提及不等于absent，检测失败不等于安静或无动作。
- 有未决单元时完整集headline为null；可评子集另列并报告缺项。多方法按**每类**取所有方法共同可评ID交集，固定原参考正例主类集合；交集丢失某个主类的正例时主Macro为N/A，不静默删除该类。
- 多draw分别计算全测试集指标，再等权报告均值与标准差；保留每draw值，不跨draw合并事件，不按GT取best。样本、参考或draw错配拒绝比较。
- 不支持音频输出的方法在音频组为N/A；支持音频但缺少文件是错误。真实全零波形是可测的静音，不借缺文件伪造静音。

它衡量**与单一观察参考的类别一致性**，不等于反馈的语境正确率。语境适当性／参与感和生成内部视听事件协调性仍独立评价；本模块不凭自动类别结果生成这两项人工分数。

## 复用标注pipeline的范围与适配

原链路见[项目README](../../README.md)、[全量打标说明](../../related/P2_四机32卡检测打标重叠运行_20261103.md)、[v4自然语言标注](../../experiments/listen_native_caption_nod_vocal24_20261103/README.md)。本模块复用：

1. MediaPipe25fps姿态和blendshape，原点头recall规则、摇头方向门控、皱眉expression_gate_v3／FER候选；保留独立类别线索，不使用六类互斥argmax结果当多标签真值。微笑由实际视频判别，blendshape曲线仅作证据，不新造语义阈值。
2. 冻结Qwen3-Omni-Instruct（本地30B MoE权重）和原8fps媒体接口、可观察事实原则、ASR提示规范化规则。
3. 同一冻结Qwen3-Omni先在每侧实际波形上产生ASR候选，复用原规范化函数，再用独立一次音频判别请求重新听辨；这里没有引入独立ASR模型。候选不等于检测真值。视图分为无声视频和纯音频，避免从笑脸猜笑声、从声音猜动作。

评测适配使用新版本提示及严格七类状态JSON，并非逐字复用原自然段prompt。GT与生成均重新检测，不读旧caption、SFT规划输出或训练Event head。生成使用自己的**完整音轨**，不套用GT时间窗／转写；GT使用既有封存listener波形配方。两侧音频保持增益，不归一化、不按GT事件剪生成声音。

长片段以不依赖GT事件的固定12秒窗口／0.5秒重叠完整覆盖，窗口结果按类OR聚合；未见正例但有窗口未知/失败时保留unknown/error，不能宣称全段不存在。25fps线索和8fps观察输入可能漏掉快速动作，需人工核验；face覆盖不足时不据此输出缺失类别的确定负例。

阈值沿用标注规则并注明来源，尚未据独立人工子集完成benchmark校准；模型／提示、库版本、媒体与代码身份在contract中保存。当前模型身份为config/index哈希及各权重分片大小/mtime，**不宣称已逐分片完成密码学哈希核验**。复用同一标注器存在共享误差风险，独立人工核验仍必要。

## 接入当前框架

直接接受现有`metric_manifests/.../*.jsonl`中的绑定pair记录：`dataset/id/generated`，`gt`或`gt_reference`，`pair_duration_s`，以及音频引用。GT懒解码、最终裁脸和净化波形读取复用OmniResponse的原adapter。新阶段独立输出，不改写旧summary或让在运行worker自动切换指标。

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5

python scripts/p2_event_presence.py bind \
  --manifest /path/to/bound_pairs.jsonl \
  --method model_name --draw 0 \
  --work /path/to/new_event_scores

# 自动调度原媒体环境、CPU候选环境及冻结标注模型环境
python scripts/p2_event_presence.py run --work /path/to/new_event_scores

# 仅重新聚合已完成检测，不重新加载模型
python scripts/p2_event_presence.py score --work /path/to/new_event_scores

# 清单内有多个method时，同时输出逐类共同子集对照
python scripts/p2_event_presence.py compare --work /path/to/new_event_scores
```

多清单可重复`--manifest`；每条可显式提供`method`、字符串`draw_id`和`capabilities: {"visual":true,"audio":true}`。视觉方法可`--visual-only`。自对照需要显式`--self-check`和`self_check_only`标记，报告不将它列为模型结果。

默认[配置](../../configs/p2_event_presence_v1.json)只使用node11 GPU7，模型不训练，单卡需要足够空闲显存及16GiB余量；可使用CPU offload保持相同权重，不量化替换模型。资源不足明确退出，不停止其他作业。动态库和PyAV沿用现有vLLM／paired-video环境，不修改训练环境。恢复只复用身份一致的完整媒体和检测；若代码、提示、模型身份、媒体或标签发生变化，使用新输出目录。

## 输出及人工核验接口

```text
contract.json / frozen_pairs.jsonl       冻结清单、配置和代码身份
prepared_pairs.jsonl / media/            各侧独立媒体、音轨、观察窗及候选
detector_contract.json                  冻结模型／提示／运行版本和适配差异
media/*/*_raw.json / *_prompt.txt        原始模型输出、重试及实际提示
media/*/detection.json                  各类显式状态、证据和失败
event_pairs.jsonl                       评分输入标签和参考身份
summary.json / per_class.csv / REPORT.md 完整集／条件子集／共同子集／逐draw统计
blind_review_media.jsonl                隐去方法及GT/生成角色的核验素材索引
human_labels.template.csv               空白人工七类标签表
HUMAN_REVIEW_STATUS.json                人工核验与校准状态
```

人工标注填写`present/absent/unknown`、证据和`rater_id`；空白保持未标注，不自动变负例。同一单元多位标注者不一致时先保留unknown，不擅自多数表决。导入后单独计算自动检测相对人工标签的逐类结果；它不自动改变主表参考标签，也不自动宣称阈值已经校准。

```bash
python scripts/p2_event_presence.py score \
  --work /path/to/new_event_scores \
  --human-labels /path/to/completed_blind_labels.csv
```

真正的语境适当性／参与感及视听协调性量表仍待另行确认，此表只核验事件检测。

## 验证状态

CPU不变性和边界检查：`python -m unittest discover -s tests -p test_event_presence.py -v`。包含全集计数、重复／时间平移不变、双空、无参考正例误报、未知／失败、逐类共同集合、多draw、模态N/A、参考冲突、漏记录及人工分歧。

实际小规模执行已完成：[选择、结果与验收说明](../../experiments/p2_event_presence_pilot_20261108/README.md)。ResponseNet／Seamless各两对真实GT／生成，加四对GT自对照，去重后八个媒体；候选和模型检测技术失败0，56个媒体类别单元中1个unknown。Seamless完整音频主表因此为N/A，保留可评子集与具体缺项。28张逐类计数表独立重算通过；缺生成文件时GT仍可读的真实控制也通过。本次模型GPU进程已退出。

[盲化视频／音频核验页](http://localhost:8876/listen_benchmark_review/event_presence_pilot_20261108/)的八个视频／八段音频播放及手机布局已验证。独立人工标签与阈值校准仍pending；上述结果证明端到端执行可用，不证明检测器已具备正式benchmark可靠性。
