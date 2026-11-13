# 两条来源的句末完整性试修 J

2026-10-29。用户授权先切两条原始视频供人工检查。本次选择 170 秒 `naturalistic_V02_S3610_I00000088` 和 190 秒 `naturalistic_V02_S3579_I00000175`；前者覆盖残尾与长句拆分，后者检查已有修复不退步。保持先切候选、后筛音质的流程，5 秒强制停顿边界、2–11.95 秒限制及原音质规则均保留。串行试验，未修改旧全量结果或启动全量。

## 先看 HTML

- [170 秒：长句恢复、He’s／He／What 句末保护和整段待复核](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3610_I00000088&fixed=1&compare=trial_long_sentence#fixedPairReview)
- [190 秒：During 整句及其他已有修复、完整候选待复核](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3579_I00000175&fixed=1&compare=trial_during#fixedPairReview)

三列是旧全量 P、修复前 H、试修 J。每组注明比较初始候选还是最终结果；三列均按实际边界播放，不加上下文。整段待复核／规则排除的候选不是自动通过。默认列出全部初始候选，版本下拉保留旧全量及 H/L/G/F；旧链接仍解析到相应版本。其余五条没有应用 J。

## 实现范围

这是一份独立试验模块 `scripts/listen_local_recovery_semantic_trial.py`，未修改冻结旧生产模块或 H/L 规则。

1. **句末保护。** 短尾补回若把已经选好的 sentence_terminal 改成非句末，保留原切点，记录未采用扩展。原本补完整句的短尾修复保留。
2. **有限的后续句子检查。** 只在同一安全区间开头存在不超过三个词、语音跨度小于 1.6 秒的短前缀，随后停顿 1–5 秒时考虑。若“前缀＋后续句末”超过 11.95 秒，而后续句子独立通过原声学切分，则在停顿处分开处理。1 秒、三词等为本轮试验限制，尚未验证全量；不是恢复全局 3 秒阈值。不跨另一人实质发言、保护区或已有候选。
3. **恢复完整性保护。** 对局部恢复的子段，要求句末为 sentence_terminal，句首有已知边界证据；不能确认时保留原候选及原音质结论，供人工复核，记录未采用的子段。没有把音质疑点自动改为通过。这是保守试验规则，不是语法模型，可能暂缓实际可独立使用的子句，需要用户试听。

## 实测结果

| 来源 | 初始候选 H→J | 初始起止不变 | 旧全量通过 | H 通过 | J 通过 | J 未自动通过的整段候选 |
|---|---:|---:|---:|---:|---:|---:|
| 170 秒 | 20→20 | 16 | 19 | 18 | 16 | 4（3 待复核、1 规则排除） |
| 190 秒 | 20→20 | 20 | 20 | 20 | 16 | 4（待复核） |

初始父候选与恢复子段不能相加当作覆盖。170 秒初始有 14 段音质通过、6 段未通过，其中两段恢复为完整子段通过；最终合计 16 段通过、4 段原候选未自动通过。190 秒的恢复提案被保守完整性检查暂缓，全部 20 个初始候选保留，16 段通过。

明确修复与保留：

| 内容 | H | J | 结论 |
|---|---|---|---|
| after that ... Mufasa movie theater ... Dolby Theater | 21.340–31.350；32.337–43.320 中的跨段内容 | 25.160–35.498 | 整句恢复，音质通过；与旧全量该段相同 |
| He's mad. | 140.652–146.002，追加 He's | 140.652–145.876 | 恢复原句末并通过 |
| ...he ain't about it. | 148.430–154.183，追加 He | 148.430–153.400 | 恢复原句末并通过 |
| ...Mufasa. | 50.715–58.098，追加 What | 50.715–57.618 | 不再以 What 残尾结束，音质通过 |
| San Diego／回应 | 5.743–11.700 | 同左 | 原修复保留且通过 |
| During...marry her | 149.903–161.853 | 同左 | 整句修复保留且通过，尾部零留白提示保留 |
| 末尾求职整句 | 174.058–176.179；176.179–186.780 | 同左 | 短尾整句修复保留且通过 |

相较 H 初始候选，170 秒有两个原覆盖词未进入 J pair：Man,（21.495–21.815）及 What（57.658000–57.798000）。它们没有从原数据删除，未成段原因和位置保留；本次不声称无词覆盖损失。不能为多保留一个短前缀或残尾而拆坏后面的长句／已有句末。

## 需要人工判断的整段候选

八段未自动通过，其中六段因恢复子段不能确认完整，保留原候选供判断：

### naturalistic_V02_S3610_I00000088

- J1-04：36.078000–46.380000，audio_review：And I was looking for some bitches. And my cousin, he was not on the right track. I don't know what he was doing in the back, talking to some guy in the back.
- J1-13：104.440000–110.908500，audio_review：Cake. Cake. I don't know if they all understand me. They probably understand like a motherfucker.
- J1-19：160.669000–167.534000，audio_reject：We over here enjoying our time, talking to the bitches, chilling with the bitches, chilling, getting the money, big

### naturalistic_V02_S3579_I00000175

- J2-03：19.437000–30.710000，audio_review：And then after that, man, I hit up Snoop Dogg and Wiz Khalifa. And, you know, we got to chilling, making music.
- J2-11：97.956000–107.721000，audio_review：So the walk back to Canada was crazy because that is such a long walk. And I was so thirsty and hungry.
- J2-15：134.119000–146.069000，audio_review：when I made it to my destination to Canada I needed to see a therapist because the the trip was so crazy and mind-blowing and

另有 170 秒的 83.425–86.146，以及 190 秒的 118.066–121.528（That's like maybe two or three days.），音质疑点本轮未自动放行。试修不是“最终通过越多越好”的结论，需人工看完整性与模型疑点。

## 验收与文件

- [三版本逐段起止、原文及状态 CSV](intervals.csv)，包含所有片段，不只变化样本。
- [候选与所有媒体](review/manifest.json)；[统计](comparison.json)。
- [数据验收](validation.json)：无新增词中切口、时长违规或重叠；38 个冻结文件与旧全量结果指纹不变。
- [五项回归](tests.log)：真实残尾保护、真实长句与未成段前缀记录、已有三处修复保留、恢复保留父候选且音质结论不变、时长与词边界约束。
- [浏览器验收](browser_validation.json)：两个页面、三列标签、全部时间和转写、50 次对照播放、Range 206、手机布局、旧版本链接均通过。
- QC3/QC4 仅对三个新边界片段重新计算，39 个精确匹配区间复用原音质证据；媒体仅新导出三段，39 条媒体记录复用。没有新增降噪或静音处理。

```bash
python experiments/pair_semantic_trial2_20261029/run.py
python -m unittest discover -s tests -p test_listen_semantic_trial.py -v
python experiments/pair_semantic_trial2_20261029/export.py
python experiments/pair_boundary_real_audit_20261029/publish_review.py
PYTHONPATH=/tmp/v8_browser_deps LD_LIBRARY_PATH=/tmp/v8_browser_libs/extracted/usr/lib/x86_64-linux-gnu python experiments/pair_semantic_trial2_20261029/check_browser.py
```

本轮仅是两条试修，不宣称整体优于旧全量或已适合全量。尚存初始候选的旧语义疑点、源视频结束时话未说完等情况，在页面保留提示。
