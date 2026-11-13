# 回应边界与重复次数复核（2026-10-25）

用户已同意先按“原始观测→独立事件候选→上层回应组”的方案处理并人工复核。当前仅验证旧版发生合并的 5 组；50 段完整原始观测保留并可切换查看。未改正在运行的全量打标。

页面：远端 http://127.0.0.1:8876/listen_benchmark_review/response_review/ ，本机通过编辑器转发 8876。已通过编辑器发起打开。

## 已完成的处理

- 旧组只保留 candidate_span，明确不是连续活动区间，不用于精确边界或次数监督。
- 从原始 ASR 文件恢复完整词序，包括零时长、越窗记录；不能用过滤后的有效词数代表实际重复次数。
- 不同转写之间，相同文字且两端相差不超过0.2秒、有实际重叠、双向唯一匹配，并与已有组内全部记录满足条件，才连接为候选重复。该容差是候选匹配参数，不代表时间精度已经验证。
- 同一次转写的不同词永不合并，不做传递式区间合并，不让偏宽记录串起多次回应。重复候选依然保留所有原始成员。
- 整段 ASR 只作参考。5组实际发生次数为null，confirmed_occurrences为空，待人工听音后填写。视觉候选描述和原始观测不变。

## 发现

S0805 整段为5个yeah，其中2个词零时长；窗口的词数/边界互相冲突，不能声称是一声持续yeah或已确定5声。S2043的3个“嘿”和S3358的2个“嗯”来自各自同一次转写，原先的相接合并掩盖了词序。S0835的Ed整段边界延伸至10秒，窗口还出现Yeah，内容与时间待核对。S5193的两条发声描述只重叠0.02秒，不能据此确定同一次回应。

5段共53条原始词记录，15条时间无效（零时长或越窗）；12个跨窗口候选重复组。后者是观测去重候选数，不是实际发声次数。本次未重新运行媒体模型，也未声称完成听觉人工判定。

## 人工复核

逐段播放独立 listener 音频，选择“一次发声／多次发声／不是回应／无法确定”。确定时填写真实区间与描述；不能确认时留空区间并说明。有效整段词区间可以载入编辑，但不会自动补造零时长词的边界。保存本组后导出JSON；本地浏览器保存可导入恢复，不自动写入训练标签。

数据：[完整证据](review.json)、[候选与监督状态](review_candidates.jsonl)、[原始观测](observations.jsonl)、[验证](validation.json)、[源文件哈希](manifest.json)。

## 代码与重放

src/listen_benchmark/response_review.py，scripts/build_listen_response_review.py，scripts/listen_response_review.html，tests/test_listen_response_review.py，scripts/check_listen_response_review.py。

使用具备numpy/soundfile/fontTools的Python运行build_listen_response_review.py，参数：--source experiments/listen_response_episodes_20261024T120000Z --alignment experiments/listen_split_validation_20261024T074733Z/word_alignment --output <新目录> --preview data_previews/listen_benchmark_review/<新目录>。可选--font指向NotoSansCJK-Regular.ttc（字体编号2），按当前页面文字生成woff，避免沿用旧版子集字体缺字。无--font使用系统字体。

6项去重/边界单元测试通过；浏览器检查播放同步、5组媒体Range206、无效复核拒绝、保存导出导入、50段切换、手机布局通过。浏览器自动化测试记录只在独立临时浏览器中产生，未当作人工结论。

## 2026-10-25 收到的人工反馈

已保存 [原始反馈](human_feedback/20261025T022241Z.original.json) 和 [逐字段解释](human_feedback/20261025T022241Z.interpreted.json)。备注纠正：S2043为“嘿嘿”式笑声；S0805确认Right×2、Yeah×5但区间未补齐；S0835为Yeah及尾音，不能沿用Ed的3.84–10秒；S3358确认两次嗯；S5193确认Uh huh及Yeah并提供大致时间。人工确认内容、次数、边界分别保存，未将表单提交等同于全部字段已确认。旧复核页面仍为当时的模型候选，未自动变成最终标签。
