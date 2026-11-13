# 原抽样剩余五条：同一份 J 规则扩展检查

2026-10-29。用户检查前两条后，要求继续处理原抽样视频并比较差异。本轮处理原七条中剩余五条，没有新增抽样，没有重跑已检查的两条。沿用 `sentence-integrity-two-source-trial-6`；模块 SHA256 与已检查两条完全一致，没有按新结果修改规则。所有处理与媒体导出串行。

## HTML 入口与统计

旧全量统一取 `data_corpora/seamless_listen_local_full/results` 的实际结果；H 为此前最新版，J 为本轮沿用的试修版。不能混用候选数与自动通过数。旧全量 P、H、J 三列实际视频旁均注明阶段、起止时间、原文及状态；初始切分与最终音质通过分别比较。

| 来源 | HTML | 初始 H→J | 起止完全不变 | 旧全量／H／J 自动通过 | J 未自动通过终端候选 |
|---|---|---:|---:|---:|---:|
| `improvised_V01_S0122_I00000125`（196.000 秒） | [打开对照](http://127.0.0.1:8875/listen_full_source/?pair=improvised_V01_S0122_I00000125&fixed=1&compare=remaining_initial1#fixedPairReview) | 17→16 | 14 | 9／8／4 | 12 |
| `naturalistic_V02_S5228_I00000302`（176.000 秒） | [打开对照](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S5228_I00000302&fixed=1&compare=remaining_initial1#fixedPairReview) | 19→19 | 18 | 21／21／19 | 1 |
| `naturalistic_V03_S2098_I00000195`（397.998 秒） | [打开对照](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V03_S2098_I00000195&fixed=1&compare=remaining_initial1#fixedPairReview) | 44→44 | 39 | 45／45／42 | 2 |
| `improvised_V00_S0862_I00000794`（442.000 秒） | [打开对照](http://127.0.0.1:8875/listen_full_source/?pair=improvised_V00_S0862_I00000794&fixed=1&compare=remaining_initial3#fixedPairReview) | 55→54 | 46 | 53／55／53 | 1 |
| `naturalistic_V03_S1909_I00000205`（353.987 秒） | [打开对照](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V03_S1909_I00000205&fixed=1&compare=remaining_final1#fixedPairReview) | 42→42 | 40 | 38／39／39 | 3 |

初始切分合计 177→175 段，157 段起止完全相同，17 组变化。最终通过共 157 段，另 19 段终端候选未自动通过；其中 7 个父候选因恢复子段不能确认完整，保留整段供人工检查。音质未通过者未自动放行，保留候选不计入通过时长。

## 需要优先人工检查

**398 秒来源出现新的取舍：I like that. 未再成段。** 原 H 从 42.230 秒开始，包含完整短句 I like that.；J 的后续整句从 44.164 秒开始，到 55.194 秒结束，再接 55.214–61.939。触发短前缀后的长句切点调整：短句对齐 42.462–42.923 秒，后续间隔 1.441 秒。这个短句未能独立通过当前最小时长／声学切分条件，留在未成段诊断，不应宣称它是无用残尾。已放在本条首个重点对照。

**442 秒来源的 I mean... 和 Yeah, 需要试听判断。** 句末保护分别把 141.680–148.399 缩回 141.680–147.959、380.235–384.230 缩回 380.235–383.938。它们可能有独立回应意义，不能仅根据标点自动认定应该去掉。其余多处确实停止在原来的句末，但仍需人工核实。

相较 H，49 个已有对齐词未再进入 J 初始 pair，按来源和具体时间完整保存在比较 JSON。多数出现在被移出的短尾中，但其中包含完整短句和可能有效回应，因此本轮不宣称无覆盖损失或已达到全量条件。

此前 Okay／Yeah 的句首补接、Right 扩展失败后的完整回退，以及 354 秒来源 up／Oh, yeah 结尾补全，实际区间与音质通过状态仍保留。

## 初始切分的全部 17 组变化

### improvised_V01_S0122_I00000125

| H 起止及原文 | J 起止及原文 |
|---|---|
| 54.775000–61.240000：Yeah, because we've known each other since kindergarten and elementary school, middle school, high school.<br>61.250000–67.790000：And we're too close that we know each other so much. There's nothing that | 54.775000–65.540000：Yeah, because we've known each other since kindergarten and elementary school, middle school, high school. And we're too close that we know each other so much. |
| 94.650000–101.877000：Was that, like, one of the first big places y'all had gone to? Like, y'all didn't feel like going, like, oh, I want to go to New York. I want to go to California. You're like, nope, let's go to China. Oh, | 94.650000–101.716000：Was that, like, one of the first big places y'all had gone to? Like, y'all didn't feel like going, like, oh, I want to go to New York. I want to go to California. You're like, nope, let's go to China. |

### naturalistic_V02_S5228_I00000302

| H 起止及原文 | J 起止及原文 |
|---|---|
| 83.020000–92.182000：But I eventually was like, ah, I can play it, but I don't think I can be the one creating it. But I still think coding is really cool. I still think it's cool. I just don't think | 83.020000–91.162000：But I eventually was like, ah, I can play it, but I don't think I can be the one creating it. But I still think coding is really cool. I still think it's cool. |

### naturalistic_V03_S2098_I00000195

| H 起止及原文 | J 起止及原文 |
|---|---|
| 42.230000–52.711000：I like that. You know, sadly, it's a good thing that you learn that when you're because a lot of people much older than that, but<br>52.711000–61.939000：people my age, they still haven't learned that lesson. Like, you know, not everybody has the same life as you. I sing that song a lot. | 44.164000–55.194000：You know, sadly, it's a good thing that you learn that when you're because a lot of people much older than that, but people my age, they still haven't learned that lesson.<br>55.214000–61.939000：Like, you know, not everybody has the same life as you. I sing that song a lot. |
| 119.006000–123.770000：And the thing is, is the people that everyone thought they were super cool, they really weren't. They were just kind of | 119.006000–123.109000：And the thing is, is the people that everyone thought they were super cool, they really weren't. |
| 378.337000–385.139000：So I've had a lot of times where I had difficult people to work with. If I didn't have a difficult person to work with, maybe I wasn't working. I | 378.337000–384.799000：So I've had a lot of times where I had difficult people to work with. If I didn't have a difficult person to work with, maybe I wasn't working. |
| 389.100000–397.967000：my favorite movies is The Devil Wears Prada. And where Miranda Priestly is just like the boss from hell. And | 389.100000–397.147000：my favorite movies is The Devil Wears Prada. And where Miranda Priestly is just like the boss from hell. |

### improvised_V00_S0862_I00000794

| H 起止及原文 | J 起止及原文 |
|---|---|
| 90.919000–99.960000：feeling a little hesitant because I feel like we're almost like, you know, we've got our younger crowd, we're hip, we're city oriented. And then it's almost like we're gonna do a whole new concept<br>99.960000–103.165000：to appeal to a different demographic. So I just I'm a little | 90.919000–102.345000：feeling a little hesitant because I feel like we're almost like, you know, we've got our younger crowd, we're hip, we're city oriented. And then it's almost like we're gonna do a whole new concept to appeal to a different demographic. |
| 104.370000–107.710000：gonna be completely different. I think we can have | 104.370000–106.391000：gonna be completely different. |
| 141.680000–148.399000：if we have trouble, like, marketing? I mean, we're going to be in a new area where people don't know the Il DiGiorno name. I mean... | 141.680000–147.959000：if we have trouble, like, marketing? I mean, we're going to be in a new area where people don't know the Il DiGiorno name. |
| 169.639000–175.101000：After we get one in the valley, one in the city, maybe, I don't know, one Paso Robles. I don't | 169.639000–174.422000：After we get one in the valley, one in the city, maybe, I don't know, one Paso Robles. |
| 184.328000–191.149000：know, and I love, you know, I love your optimism. You're the yang to my yin. It's great. And now I think | 184.328000–190.589000：know, and I love, you know, I love your optimism. You're the yang to my yin. It's great. |
| 230.340000–234.995000：too. And it's just, I don't know, it's just startup costs. It's kind of crazy. And then the lease? I | 230.340000–234.855000：too. And it's just, I don't know, it's just startup costs. It's kind of crazy. And then the lease? |
| 246.132500–250.381000：seven years ago when I was just a little bit more loose. And I had a lot less to | 246.132500–249.101000：seven years ago when I was just a little bit more loose. |
| 380.235000–384.230000：did that, so I just, I don't know why we need to open up a second one, you know? Yeah, | 380.235000–383.938000：did that, so I just, I don't know why we need to open up a second one, you know? |

### naturalistic_V03_S1909_I00000205

| H 起止及原文 | J 起止及原文 |
|---|---|
| 283.330000–287.640000：That could be. The snail is sluggish. I | 283.330000–287.371000：That could be. The snail is sluggish. |
| 287.669000–290.230000：Sluggish. Where's it going? Is it | 287.669000–290.010000：Sluggish. Where's it going? |

## 全部最终通过差异与待复核内容

原 HTML 共 61 组对照，包含：H→J 初始切分变化、三版本最终通过范围的所有差异，以及所有未自动通过的终端候选。初始与最终为同一组三段视频时合并展示并注明，不要求重复试听。每条的完整初始／通过时间表也包括未变化片段。

- [三版本逐段 CSV](intervals.csv)：旧全量、H、J；初始候选／最终通过／全部终端结果分层，含原文、起止及状态。
- [完整比较 JSON](comparison.json)：对旧全量和 H 分别计算，保留未再覆盖的词、边界与角色信息。
- [视频、证据与三列比较清单](review/manifest.json)。

## 验收与资源

- [数据验收](validation.json)：与用户检查过的 J 模块同指纹；38 个冻结代码文件及五条旧全量结果指纹不变；无新增词中切口、时长违规或重叠；旧全量／H／J 的所有最终通过差异、H→J 初始变化及未通过候选均已展示。
- [浏览器验收](browser_validation.json)：五条页面、全部时间及原文、三列标签、194 次对照视频播放、Range 206、390px 手机布局及旧链接检查通过；原先两条 J 页面和入口保持可用。
- 原规则不变，不额外重跑既有单元测试；对这五条实际输出做完整数据验收和播放检查。
- 5 段新边界重新计算 QC3/QC4，177 条精确匹配记录复用已有音质证据；7 条媒体新导出，175 条精确复用。单任务串行，CPU 模型限制为 2 个计算线程；无全量扫描或 GPU 并发。
- J1/J2 沿用此前两条的编号；本轮按列表分配 J3–J7。原 P/H/L/G/F 历史版本均保留。

## 复现

```bash
python experiments/pair_semantic_remaining5_20261029/run.py
python experiments/pair_semantic_remaining5_20261029/export.py
python experiments/pair_boundary_real_audit_20261029/publish_review.py
PYTHONPATH=/tmp/v8_browser_deps LD_LIBRARY_PATH=/tmp/v8_browser_libs/extracted/usr/lib/x86_64-linux-gnu python experiments/pair_semantic_remaining5_20261029/check_browser.py
```

用户本轮要求的是扩展人工检查。原生产、训练数据未替换，未启动新版全量。
