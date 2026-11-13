# 第一版描述方式＋精简点头与静音说明：50 条对照试验

2026-10-30，用户要求回到第一版的描述组织方式、减少附加指令后重新试验。本轮是独立小样本对照，不自动替换已冻结的 v1，不启动全量或训练。

[三版有声对照 HTML](http://localhost:8876/listen_benchmark_review/native_simple/natural_caption_first_evidence/) · [本轮完整 prompt](prompt.txt) · [逐条实际 prompt](prompts/) · [结果与输入证据](results/) · [共同口径文本筛查](comparison.json)

## 实际改动

第一版自然语言描述的正文保留，仅将“只使用所给视频和同步音频”一句扩大为允许补充证据。追加两项简短说明：

1. 点头条目的 id 是标识，start/end 是相对 pair 起点的秒数，observation 是动作；自然融入有证据支持的点头，区间数不能当动作次数，空候选不证明动作不存在，不解释这些元数据。
2. listener_audio_is_silent=true 表示真实输入波形全部为零，不能声称可听的说话或发声；否则以实际听到的内容为准。

没有提供任何旧段落或转写。第一版与上一版结果只由 HTML 构建脚本在推理完成后读取，供用户比较。

本轮说明约286个英文空白分词，上一版约294个；主要改变是恢复第一版的段落组织要求并减少点头专项指令，不是大幅缩短 token。未改变原有解码策略。

## 输入与运行

- 同一批50条、listener 512×512原生视频8 fps、同一净化音轨；40条全静音、10条非全静音。
- GLARE recall 候选与上一版一致：38条共50个区间，12条空列表；没有重新跑检测器。
- 模型：`/private/ckpts/Qwen3-Omni/Instruct`。node4八卡各完成6/6/6/7/6/6/6/7条；0–3卡显存预算0.55，4–7卡0.85，其他任务保留。
- temperature=0、4096 token、repetition_penalty=1.05；生成异常最多重试一次，保留每次原文。本轮50条均首次正常完成，重试0、截断0。
- 音频按配方在内存置零，不保存新WAV，不改变时间轴。所有输出原文保留，没有人工补写或事后删句。

## 结果与仍存问题

改善的具体案例：

- S0699：上一版只有一次Okay，本轮恢复两次，并写入点头；点头和发声先后仍需看画面复核。
- S0805：保留两次Right、五次Yeah，分别写出前段和片尾点头；发声集中在段末描述，视听同时发生关系仍不够清楚。
- S1337：上一版全静音却声称说话，本轮明确没有发声；仍有engaged解释性措辞。

明确保留的问题：

- S2043：仍未明确写入点头，while still laughing可能延长笑声至遮嘴阶段。
- S5193：仍有contemplative，早期明显表情变化未完全恢复。
- S0173：上一版写三次点头，本轮写两次；仅凭两份文字不能判断哪份正确。
- S5259：将空候选解释为没有点头，并写出provided supplementary information，违反本轮指令。

有候选且正文含nod的片段从上一版31/38增至35/38。这是文字检索，不是准确率；全文中另有否定或元数据用法，不能把所有nod词命中当真实检出。共同规则检出的数字时间表述为25→23条、指定推演词为3→5条，改善并不全面。`comparison.json`明确标注这些只是文本筛查，不是人工真值。

## 代码与验证

- `src/listen_benchmark/native_caption_nod.py`：新增可选`first_plus_evidence`风格，默认v1保持原样；已逐条核对50条旧v1的render输出完全不变。
- `scripts/prepare_native_caption_nod_direct.py --prompt-style first_plus_evidence`：冻结新风格及相同样本输入。
- `scripts/run_listen_native_caption_pilot.py`：按protocol中的prompt_style渲染，不引入旧caption。
- `audit.json`：50条输入身份、四字段、静音标记、实际prompt、原始attempt、八卡完成检查通过；semantic_quality_verified=false。
- `cpu_validation.json`：第一版正文保留、v1渲染不变、新请求无旧描述的检查。
- `browser_validation.json`：有声播放、区间跳转、两版历史描述展开、50条音频PCM哈希、移动端布局与JS错误检查。
- `code/`与`code_hashes.json`：实际运行时的源码；`delivery_code/`：最终准备、审计与三版展示代码的本地快照。

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
python scripts/audit_native_caption_nod_direct.py --run experiments/listen_native_caption_first_evidence50_20261030
python scripts/build_listen_native_caption_review.py --run experiments/listen_native_caption_first_evidence50_20261030 --output-name natural_caption_first_evidence --compare-run experiments/listen_native_caption_clean50_20261030/text8 --compare-run experiments/listen_native_caption_nod_final50_20261030
```

页面左侧播放相同媒体，右侧展示本轮原文，并可分别展开第一版和上一版。重点样本有具体复核提示。
