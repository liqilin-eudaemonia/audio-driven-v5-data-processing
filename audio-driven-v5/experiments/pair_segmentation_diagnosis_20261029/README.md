# 七来源切分退步定位：保留先切候选的流程

2026-10-29。用户认可“先切完整候选，再筛音质、局部恢复”，本轮不将流程顺序认定为错误，也不回退到先排音质再切。目标是定位具体切分决策。未修改切分／音质代码，未重跑音质模型或全量。

## 结论及复现方法

在相同源数据、相同 `propose([])` 输入条件下，串行重算七条的 V1／F／G／L／H 初始切分。H 的所有起止与原文均与已发布结果一致。随后仅在诊断进程内，记录每次 `recover_region` 调用中贪心切分结果和短尾补回后的结果；对已有待复核父候选调用同样的局部恢复，不重新判音质。

### 1. 短尾补回会把已经完整的句末改成未完句

`listen_local_recovery_localbc.recover_region` 先由 `_greedy_recover_region` 按句末优先切分，随后在 315–317 行尝试把剩余词全部拼回。只要声学切分和时长可行，就直接替换；未检查是否从 `sentence_terminal` 降成 `recovered_word_boundary`。后面的断言只要求词全部被覆盖，不能保护语义完整性。这个步骤源于 F 版对短尾遗漏的修复，G/L/H 继承了它；不是最后 H 的句首补接造成。

因此，“多保留几个词”可能破坏更好的切点。记录到 19 次句末降级调用，其中 **13 处精确对应旧全量与最新版最终通过结果**：补尾前起止及原文等于旧全量，补尾后等于最新版。此计数仅是七条的诊断，不是全量发生率；标注句末也不是人工语义金标准，如 I mean... 仍需结合语音判断。

| 来源 | 旧全量／补尾前 | 最新版／补尾后 | 新结尾 |
|---|---|---|---|
| `naturalistic_V02_S3610_I00000088` | 140.652000–145.876000 | 140.652000–146.002000 |  morning. Brody wasn't ready. Drunk, passed out, didn't get none. He's mad. He's |
| `naturalistic_V02_S3610_I00000088` | 148.430000–153.400000 | 148.430000–154.183000 | And even though he get lucky every other time, but he ain't about it. He |
| `naturalistic_V02_S5228_I00000302` | 105.600000–109.592000 | 105.600000–109.913000 | I was sweating. I was like, dude, trembling. I was like, you |
| `naturalistic_V03_S2098_I00000195` | 119.006000–123.109000 | 119.006000–123.770000 | eryone thought they were super cool, they really weren't. They were just kind of |
| `naturalistic_V03_S2098_I00000195` | 378.337000–384.799000 | 378.337000–385.139000 | ith. If I didn't have a difficult person to work with, maybe I wasn't working. I |
| `improvised_V00_S0862_I00000794` | 104.370000–106.391000 | 104.370000–107.710000 | gonna be completely different. I think we can have |
| `improvised_V00_S0862_I00000794` | 141.680000–147.959000 | 141.680000–148.399000 | oing to be in a new area where people don't know the Il DiGiorno name. I mean... |
| `improvised_V00_S0862_I00000794` | 169.639000–174.422000 | 169.639000–175.101000 | ne in the valley, one in the city, maybe, I don't know, one Paso Robles. I don't |
| `improvised_V00_S0862_I00000794` | 184.328000–190.589000 | 184.328000–191.149000 | ow, I love your optimism. You're the yang to my yin. It's great. And now I think |
| `improvised_V00_S0862_I00000794` | 230.340000–234.855000 | 230.340000–234.995000 | I don't know, it's just startup costs. It's kind of crazy. And then the lease? I |
| `improvised_V00_S0862_I00000794` | 246.132500–249.101000 | 246.132500–250.381000 | seven years ago when I was just a little bit more loose. And I had a lot less to |
| `naturalistic_V03_S1909_I00000205` | 287.669000–290.010000 | 287.669000–290.230000 | Sluggish. Where's it going? Is it |
| `naturalistic_V03_S1909_I00000205` | 12.157000–16.925000 | 12.157000–17.007000 | I think going to college now really reinforces that I like to learn. I |

最明确的两个真实例子：

- 140.652–145.876：原本停在 “He's mad.”；补尾变为 140.652–146.002，末尾为 “He's mad. He's”。
- 148.430–153.400：原本停在 “he ain't about it.”；补尾变为 148.430–154.183，末尾追加孤立 “He”。

两例补尾前 `recovery_complete_sentence=True`，补尾后为 False，且 `semantic_end` 从 sentence_terminal 变为 recovered_word_boundary。无需改变任何音质判定即可直接观察这一步的负作用。不能据此简单删除整个补尾功能，因为已确认的短尾完整句遗漏修复仍依赖它；需要区分补完整语义单位与追加下一句碎片。

### 2. 5 秒阈值与局部贪心策略相互作用，拆散本可单独保留的长句

第一条 21.495–21.815 秒是 “Man,”，后一句 after 从 25.376 秒开始，间隔 3.561 秒。到 Theater. 结束 35.198 秒，从 Man 起算跨度 13.703 秒，超过 11.95 秒上限。

同样先切候选、完全不传旧音质阻断证据时：

- V1／F（3 秒）：25.160–35.498，完整保留 after that ... Mufasa movie theater ... Dolby Theater.
- G／L／H（5 秒）：21.340–31.350，到 Mufasa 结束；32.337–43.320 从 movie theater 接起，并与后面的句子拼在一起。此时尚未筛音质，长句已经拆开。

原因为短间隔被连入同一说话区间后，算法固定从当前最早词开始，候选时长不足 1.6 秒的不尝试；可选句末超过 11.95 秒时，退回词／停顿边界。它没有比较“当前怎样分，能让后续整句更完整”的整体方案。后续音质恢复又把第二段缩成 32.337–34.778 的 movie theater ... the Dolby，是另外一层破碎。

这不是说 5 秒规则应该撤回。5 秒可作为强制长停顿边界，但并不能替代段内的软切点与后续整句保护；也不能保证短语气词与长句都能在现有最短／最长时长约束下完整保留。

### 3. 音质恢复后的碎片没有独立的完整性保护

原本完整的候选出现模型音质疑点后，恢复函数从“安全区间”重新切片。新子段通过音质，就可归入 keep_candidate；语义边界标记仍是提示，不阻止子段成为最终音质通过片段。

真实例子：

| 来源 | 初始候选 | 恢复后最终音质通过片段 |
|---|---|---|
| 196 秒 | 54.775–61.240：Yeah, because we've known each other since kindergarten ... high school. | 54.775–56.814：Yeah, because we've known each other since |
| 170 秒 | 32.337–43.320：movie theater ... Dolby Theater. And I was looking ... right track. | 32.337–34.778：movie theater, whatever, the Dolby |
| 190 秒 | 134.119–146.069：when ... because the the trip was so crazy and mind-blowing and | 134.119–139.990：when ... because the；141.323–146.069：trip was ... |

三个子段都标注 `recovered_partial_sentence`，末尾是 recovered_word_boundary，却仍是 audio_final.pool=keep_candidate。这里不把模型疑点认定为真实噪声，也不建议为了不碎而直接丢更多数据；应把“完整候选存在”和“音质模型是否通过”与“恢复子段是否仍值得作为完整 pair”分清。

## 对此前结论的纠正

“新增词覆盖”和“切点只移动几百毫秒”不足以证明优化。多个看似小幅延长，实际从完整句末变为下一句的残片，确实能解释用户认为旧版更好的听感。旧版本在这些例子中更好，是保住了原有句末；不能据此推导先排音质再切是更好的总流程。

当前定位已证明上述具体机制，尚未把全部 39 组变化逐一做完语义优劣判定。保留已确认成功的完整短尾、San Diego、During 等修复，不自动推广当前实验结果。

## 证据和入口

- [七条五版本初始切分与恢复前后跟踪](probe.json)
- [19 次句末降级记录与其中 13 处旧新最终结果的精确匹配](terminal_downgrades.json)
- [验收与冻结代码指纹](validation.json)
- [第一条长句被拆开](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3610_I00000088&fixed=1&compare=production_diff1#fixedPairReview)
- [He's mad. 后面错误补入 He's](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3610_I00000088&fixed=1&compare=production_diff5#fixedPairReview)

原 HTML 的 13 处实际变化已补充原因标注，视频与切分结果未改。

运行：`python experiments/pair_segmentation_diagnosis_20261029/probe.py`。仅访问七条已知快照及结果，串行 CPU 诊断。
