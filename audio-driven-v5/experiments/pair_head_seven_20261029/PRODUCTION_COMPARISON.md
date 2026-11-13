# 旧全量版 → 最新版：七条实际结果对照

本报告是用户要求的主对照。旧基线取自 `data_corpora/seamless_listen_local_full/results/<pair>.json`，不是连续审核小批、L 版或 H 同版重跑。旧运行版本为 `seamless-full-source-no-scene-crop-statistics-2`，旧切分规则为 `local-listening-recovery-1`；新版为 `local-feedback-head-attachment-5`。

**结论：205→206 段最终通过，159 段起止完全不变，共 39 组变化。存在内容减少的来源为用户列表第 1、2、6 条；未满足“只是明确优化”的条件，未启动全量。**

旧结果直接读取全量完成时的产物；新版沿用本轮已完成的七条 H 结果。没有再次修改切分规则。每条主对照左右均只列最终通过片段，待复核／排除的提案另列，避免把候选当作已保留。

| 来源 | 旧→新通过段数 | 起止完全不变 | 变化组数 | 旧→新通过秒数 | 原覆盖词未再覆盖 |
|---|---:|---:|---:|---:|---:|
| 1. `naturalistic_V02_S3610_I00000088` | 19→18 | 12 | 7 | 96.826000→83.124000 | 25 |
| 2. `improvised_V01_S0122_I00000125` | 9→8 | 7 | 2 | 48.428000→41.593000 | 23 |
| 3. `naturalistic_V02_S5228_I00000302` | 21→21 | 17 | 3 | 140.115000→144.738000 | 0 |
| 4. `naturalistic_V03_S2098_I00000195` | 45→45 | 39 | 5 | 326.566000→329.519000 | 0 |
| 5. `improvised_V00_S0862_I00000794` | 53→55 | 40 | 12 | 300.811500→308.201500 | 0 |
| 6. `naturalistic_V02_S3579_I00000175` | 20→20 | 13 | 4 | 140.207000→142.667000 | 6 |
| 7. `naturalistic_V03_S1909_I00000205` | 38→39 | 31 | 6 | 215.876000→219.928000 | 0 |

计数按可靠已有词对齐统计，并非人工听写准确率；覆盖秒数含切片留白。第 6 条的 6 个词是 “maybe two or three days.” 和一个重复的 “the”。新增内容或更长范围不自动表示语义更完整。

## 为什么不只是最后一次小边界修复

本次比较跨越旧全量到最新版的全部累计变化：短尾／词尾修复、3→5 秒停顿阈值、局部反馈识别、完整句末优先、句首短回应补接。此外，当前实验入口与全量入口的流程也不同：旧全量在 `run_listen_local_full.prepare()` 中调用 `context.propose(previous["clips"])`，先让既有音质阻断区参与切分；当前实验 `rerun_fixed_pairs.main()` 调用 `ctx.propose([])`，先切完整候选再筛音质并局部恢复。因此，不能把本次结果都归因于几个边界补丁。即使音质评分和路由规则不变，评估窗口和候选分组仍会改变。这里只确认流程不同，不声称每一处差异都已完成单因素归因。

第 6 条原保留 119.767–128.597 秒中的 “maybe two or three days.”，新版在 118.066–121.528 秒完整候选里仍存在，但该候选因 `isolated_reference_match_unresolved` 待音质复核，未进入最终通过集。未放宽规则自动接纳。

## 全部变化：起止、原文与试听

### 1. naturalistic_V02_S3610_I00000088

[打开本条全部变化](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3610_I00000088&fixed=1&compare=production_diff1#fixedPairReview)

| 变化 | 旧全量最终通过：源秒／原文 | 最新版最终通过：源秒／原文 | 说明 |
|---|---|---|---|
| 边界收缩 · 25.160000–35.498000 秒 | 25.160000–35.498000：after that, I seen your ass at the Hollywood Boulevard in front of Mufasa movie theater, whatever, the Dolby Theater. | 32.337000–34.778000：movie theater, whatever, the Dolby | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 原通过、现未覆盖的对齐词：after that, I seen your ass at the Hollywood Boulevard in front of Mufasa Theater.。是否误排需试听，不能视作明确优化。 起点变化 +7.177000 秒，终点变化 -0.720000 秒。 |
| 边界收缩 · 37.850000–43.320000 秒 | 37.850000–43.320000：I was looking for some bitches. And my cousin, he was not on the right track. | 38.559000–43.000000：some bitches. And my cousin, he was not on the right | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 原通过、现未覆盖的对齐词：I was looking for track.。是否误排需试听，不能视作明确优化。 起点变化 +0.709000 秒，终点变化 -0.320000 秒。 |
| 边界收缩 · 47.561000–58.098000 秒 | 47.561000–58.098000：OK, fuck it. And I'm still, look, I'm just acting like every other fangirl, recording the people walking through at Mufasa. What | 50.715000–58.098000：And I'm still, look, I'm just acting like every other fangirl, recording the people walking through at Mufasa. What | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 原通过、现未覆盖的对齐词：OK, fuck it.。是否误排需试听，不能视作明确优化。 起点变化 +3.154000 秒，终点变化 +0.000000 秒。 |
| 旧通过片段未进入新通过集 · 104.470000–106.970000 秒 | 104.470000–106.970000：Cake. Cake. | 无最终通过片段 | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 原通过、现未覆盖的对齐词：Cake. Cake.。是否误排需试听，不能视作明确优化。 |
| 边界扩展 · 140.652000–146.002000 秒 | 140.652000–145.876000：I said, we got to wake up in the morning. Brody wasn't ready. Drunk, passed out, didn't get none. He's mad. | 140.652000–146.002000：I said, we got to wake up in the morning. Brody wasn't ready. Drunk, passed out, didn't get none. He's mad. He's | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：He's。新增词不自动等于语义完整。 起点变化 +0.000000 秒，终点变化 +0.126000 秒。 |
| 边界扩展 · 148.430000–154.183000 秒 | 148.430000–153.400000：And even though he get lucky every other time, but he ain't about it. | 148.430000–154.183000：And even though he get lucky every other time, but he ain't about it. He | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：He。新增词不自动等于语义完整。 起点变化 +0.000000 秒，终点变化 +0.783000 秒。 |
| 边界收缩 · 160.669000–163.550000 秒 | 160.669000–163.550000：We over here enjoying our time, talking to the bitches, | 160.669000–163.519000：We over here enjoying our time, talking to the bitches, | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 原先覆盖的可靠对齐词未减少；请结合原文判断留白或句子分组是否合理。 起点变化 +0.000000 秒，终点变化 -0.031000 秒。 |

### 2. improvised_V01_S0122_I00000125

[打开本条全部变化](http://127.0.0.1:8875/listen_full_source/?pair=improvised_V01_S0122_I00000125&fixed=1&compare=production_diff1#fixedPairReview)

| 变化 | 旧全量最终通过：源秒／原文 | 最新版最终通过：源秒／原文 | 说明 |
|---|---|---|---|
| 起止调整 · 54.775000–57.680000 秒 | 54.830000–57.680000：Yeah, because we've known each other since kindergarten | 54.775000–56.814000：Yeah, because we've known each other since | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 原通过、现未覆盖的对齐词：kindergarten。是否误排需试听，不能视作明确优化。 起点变化 -0.055000 秒，终点变化 -0.866000 秒。 |
| 旧通过片段未进入新通过集 · 110.126000–116.150000 秒 | 110.126000–116.150000：Why not? Because we want to travel. We were young at that time. We want to do something that's new for us. | 无最终通过片段 | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 原通过、现未覆盖的对齐词：Why not? Because we want to travel. We were young at that time. We want to do something that's new for us.。是否误排需试听，不能视作明确优化。 |

### 3. naturalistic_V02_S5228_I00000302

[打开本条全部变化](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S5228_I00000302&fixed=1&compare=production_diff1#fixedPairReview)

| 变化 | 旧全量最终通过：源秒／原文 | 最新版最终通过：源秒／原文 | 说明 |
|---|---|---|---|
| 边界扩展 · 83.020000–92.182000 秒 | 83.020000–87.900000：But I eventually was like, ah, I can play it, but I don't think I can be the one creating it. | 83.020000–92.182000：But I eventually was like, ah, I can play it, but I don't think I can be the one creating it. But I still think coding is really cool. I still think it's cool. I just don't think | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：But I still think coding is really cool. I still think it's cool. I just don't think。新增词不自动等于语义完整。 起点变化 +0.000000 秒，终点变化 +4.282000 秒。 |
| 边界扩展 · 105.600000–109.913000 秒 | 105.600000–109.592000：I was sweating. I was like, dude, trembling. | 105.600000–109.913000：I was sweating. I was like, dude, trembling. I was like, you | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：I was like, you。新增词不自动等于语义完整。 起点变化 +0.000000 秒，终点变化 +0.321000 秒。 |
| 重新分段：2→2 段 · 145.215000–160.050000 秒 | 145.215000–155.206000：our like lab or whatever, but it wasn't a lab, it was just like we'd be on the computer like coding, you know? And the teacher would kind of go over like what our homework was going to be. She'd do like a tester coding thing.<br>155.226000–160.050000：We'd go through and like code something with her and then it would be like something similar that we're doing on our own for our homework. | 145.215000–149.118000：our like lab or whatever, but it wasn't a lab, it was just like we'd be on the computer like coding, you know?<br>149.118000–160.050000：And the teacher would kind of go over like what our homework was going to be. She'd do like a tester coding thing. We'd go through and like code something with her and then it would be like something similar that we're doing on our own for our homework. | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 原先覆盖的可靠对齐词未减少；请结合原文判断留白或句子分组是否合理。 |

### 4. naturalistic_V03_S2098_I00000195

[打开本条全部变化](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V03_S2098_I00000195&fixed=1&compare=production_diff1#fixedPairReview)

| 变化 | 旧全量最终通过：源秒／原文 | 最新版最终通过：源秒／原文 | 说明 |
|---|---|---|---|
| 边界扩展 · 52.711000–61.939000 秒 | 52.711000–61.828000：people my age, they still haven't learned that lesson. Like, you know, not everybody has the same life as you. I sing that song a lot. | 52.711000–61.939000：people my age, they still haven't learned that lesson. Like, you know, not everybody has the same life as you. I sing that song a lot. | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 原先覆盖的可靠对齐词未减少；请结合原文判断留白或句子分组是否合理。 起点变化 +0.000000 秒，终点变化 +0.111000 秒。 |
| 重新分段：2→2 段 · 68.077000–82.365000 秒 | 68.077000–78.523000：Like, we like commonality. We like everyone to be like us. It's a human, you know, just a normal, natural reaction. Like, oh, yeah, no, I'm going to have somebody who's like me.<br>78.543000–82.365000：I'm not going to feel so alone in the world. That's probably what you were thinking, your 11-year-old self. | 68.077000–71.239000：Like, we like commonality. We like everyone to be like us.<br>71.259000–82.365000：It's a human, you know, just a normal, natural reaction. Like, oh, yeah, no, I'm going to have somebody who's like me. I'm not going to feel so alone in the world. That's probably what you were thinking, your 11-year-old self. | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 原先覆盖的可靠对齐词未减少；请结合原文判断留白或句子分组是否合理。 |
| 边界扩展 · 119.006000–123.770000 秒 | 119.006000–123.109000：And the thing is, is the people that everyone thought they were super cool, they really weren't. | 119.006000–123.770000：And the thing is, is the people that everyone thought they were super cool, they really weren't. They were just kind of | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：They were just kind of。新增词不自动等于语义完整。 起点变化 +0.000000 秒，终点变化 +0.661000 秒。 |
| 边界扩展 · 213.390000–224.564000 秒 | 213.390000–222.723000：And then there were other people who were like, oh, well, I'm starting a business and I'm starting this new project and I'm still, you know, like | 213.390000–224.564000：And then there were other people who were like, oh, well, I'm starting a business and I'm starting this new project and I'm still, you know, like maximizing my stock | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：maximizing my stock。新增词不自动等于语义完整。 起点变化 +0.000000 秒，终点变化 +1.841000 秒。 |
| 边界扩展 · 378.337000–385.139000 秒 | 378.337000–384.799000：So I've had a lot of times where I had difficult people to work with. If I didn't have a difficult person to work with, maybe I wasn't working. | 378.337000–385.139000：So I've had a lot of times where I had difficult people to work with. If I didn't have a difficult person to work with, maybe I wasn't working. I | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：I。新增词不自动等于语义完整。 起点变化 +0.000000 秒，终点变化 +0.340000 秒。 |

### 5. improvised_V00_S0862_I00000794

[打开本条全部变化](http://127.0.0.1:8875/listen_full_source/?pair=improvised_V00_S0862_I00000794&fixed=1&compare=production_diff1#fixedPairReview)

| 变化 | 旧全量最终通过：源秒／原文 | 最新版最终通过：源秒／原文 | 说明 |
|---|---|---|---|
| 重新分段：2→3 段 · 25.350000–43.298000 秒 | 25.350000–32.651000：Yeah, well, I think what you said, like, we've had such success with this. Like,<br>32.751000–43.298000：I think we can take that leap, and I do think that we're in a position with the numbers, with our audience, with the diners, you know what I mean? Like, I think this would be really good. | 25.350000–32.110000：Yeah, well, I think what you said, like, we've had such success with this.<br>32.150000–40.835000：Like, I think we can take that leap, and I do think that we're in a position with the numbers, with our audience, with the diners, you know what I mean?<br>40.855000–43.298000：Like, I think this would be really good. | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 原先覆盖的可靠对齐词未减少；请结合原文判断留白或句子分组是否合理。 |
| 边界扩展 · 54.378000–65.450000 秒 | 55.070000–65.450000：I think if we go to, I think if we, you know, move locations, I think maybe even somewhere in the Valley where the, you know, the | 54.378000–65.450000：Yeah. I think if we go to, I think if we, you know, move locations, I think maybe even somewhere in the Valley where the, you know, the | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：Yeah.。新增词不自动等于语义完整。 起点变化 -0.692000 秒，终点变化 +0.000000 秒。 |
| 重新分段：1→2 段 · 90.919000–103.165000 秒 | 90.919000–102.345000：feeling a little hesitant because I feel like we're almost like, you know, we've got our younger crowd, we're hip, we're city oriented. And then it's almost like we're gonna do a whole new concept to appeal to a different demographic. | 90.919000–99.960000：feeling a little hesitant because I feel like we're almost like, you know, we've got our younger crowd, we're hip, we're city oriented. And then it's almost like we're gonna do a whole new concept<br>99.960000–103.165000：to appeal to a different demographic. So I just I'm a little | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：So I just I'm a little。新增词不自动等于语义完整。 |
| 边界扩展 · 104.370000–107.710000 秒 | 104.370000–106.391000：gonna be completely different. | 104.370000–107.710000：gonna be completely different. I think we can have | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：I think we can have。新增词不自动等于语义完整。 起点变化 +0.000000 秒，终点变化 +1.319000 秒。 |
| 边界扩展 · 121.696000–133.142000 秒 | 121.696000–132.101000：You know most businesses don't succeed in the first five years, I mean we've had our spot here for seven I think that's a great success, I just I don't know if I can go through | 121.696000–133.142000：You know most businesses don't succeed in the first five years, I mean we've had our spot here for seven I think that's a great success, I just I don't know if I can go through all of that again | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：all of that again。新增词不自动等于语义完整。 起点变化 +0.000000 秒，终点变化 +1.041000 秒。 |
| 边界扩展 · 141.680000–148.399000 秒 | 141.680000–147.959000：if we have trouble, like, marketing? I mean, we're going to be in a new area where people don't know the Il DiGiorno name. | 141.680000–148.399000：if we have trouble, like, marketing? I mean, we're going to be in a new area where people don't know the Il DiGiorno name. I mean... | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：I mean...。新增词不自动等于语义完整。 起点变化 +0.000000 秒，终点变化 +0.440000 秒。 |
| 边界扩展 · 169.639000–175.101000 秒 | 169.639000–174.422000：After we get one in the valley, one in the city, maybe, I don't know, one Paso Robles. | 169.639000–175.101000：After we get one in the valley, one in the city, maybe, I don't know, one Paso Robles. I don't | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：I don't。新增词不自动等于语义完整。 起点变化 +0.000000 秒，终点变化 +0.679000 秒。 |
| 边界扩展 · 184.328000–191.149000 秒 | 184.328000–190.589000：know, and I love, you know, I love your optimism. You're the yang to my yin. It's great. | 184.328000–191.149000：know, and I love, you know, I love your optimism. You're the yang to my yin. It's great. And now I think | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：And now I think。新增词不自动等于语义完整。 起点变化 +0.000000 秒，终点变化 +0.560000 秒。 |
| 边界扩展 · 230.340000–234.995000 秒 | 230.340000–234.855000：too. And it's just, I don't know, it's just startup costs. It's kind of crazy. And then the lease? | 230.340000–234.995000：too. And it's just, I don't know, it's just startup costs. It's kind of crazy. And then the lease? I | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：I。新增词不自动等于语义完整。 起点变化 +0.000000 秒，终点变化 +0.140000 秒。 |
| 边界扩展 · 246.132500–250.381000 秒 | 246.132500–249.101000：seven years ago when I was just a little bit more loose. | 246.132500–250.381000：seven years ago when I was just a little bit more loose. And I had a lot less to | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：And I had a lot less to。新增词不自动等于语义完整。 起点变化 +0.000000 秒，终点变化 +1.280000 秒。 |
| 边界扩展 · 359.759000–365.110000 秒 | 360.100000–365.110000：fear so much as just being logical and just thinking about... | 359.759000–365.110000：it's fear so much as just being logical and just thinking about... | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：it's。新增词不自动等于语义完整。 起点变化 -0.341000 秒，终点变化 +0.000000 秒。 |
| 边界扩展 · 366.316000–370.210000 秒 | 366.354000–370.210000：Like, again, I appreciate that you want to take the risks, and obviously | 366.316000–370.210000：Like, again, I appreciate that you want to take the risks, and obviously | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 原先覆盖的可靠对齐词未减少；请结合原文判断留白或句子分组是否合理。 起点变化 -0.038000 秒，终点变化 +0.000000 秒。 |

### 6. naturalistic_V02_S3579_I00000175

[打开本条全部变化](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3579_I00000175&fixed=1&compare=production_diff1#fixedPairReview)

| 变化 | 旧全量最终通过：源秒／原文 | 最新版最终通过：源秒／原文 | 说明 |
|---|---|---|---|
| 边界扩展 · 5.743000–11.700000 秒 | 5.743000–9.689000：do the story together. Like after I went to San | 5.743000–11.700000：do the story together. Like after I went to San Diego? Okay. | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：Diego? Okay.。新增词不自动等于语义完整。 起点变化 +0.000000 秒，终点变化 +2.011000 秒。 |
| 重新分段：3→3 段 · 119.767000–146.220000 秒 | 119.767000–128.597000：maybe two or three days. One month it took and you know It<br>131.538000–142.463000：was crazy I almost lost my mind when I made it to my destination to Canada I needed to see a therapist because the the trip was<br>142.463000–146.220000：so crazy and mind-blowing and | 123.294000–134.099000：One month it took and you know It was crazy I almost lost my mind<br>134.119000–139.990000：when I made it to my destination to Canada I needed to see a therapist because the<br>141.323000–146.069000：trip was so crazy and mind-blowing and | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 原通过、现未覆盖的对齐词：maybe two or three days. the。是否误排需试听，不能视作明确优化。 |
| 重新分段：2→1 段 · 149.822000–162.153000 秒 | 149.822000–152.424000：During the time I see the therapist,<br>153.225000–162.153000：I find the therapist is very pretty and she's so mindful and I decided to marry her. | 149.903000–161.853000：During the time I see the therapist, I find the therapist is very pretty and she's so mindful and I decided to marry her. | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 原先覆盖的可靠对齐词未减少；请结合原文判断留白或句子分组是否合理。 |
| 重新分段：1→2 段 · 174.058000–186.780000 秒 | 174.058000–184.661000：But that's very expensive. So I have to get a new job in Canada for the purpose to get my appointment to see the therapist every single week. | 174.058000–176.179000：But that's very expensive.<br>176.179000–186.780000：So I have to get a new job in Canada for the purpose to get my appointment to see the therapist every single week. So I'm looking for a new job right now. | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：So I'm looking for a new job right now.。新增词不自动等于语义完整。 |

### 7. naturalistic_V03_S1909_I00000205

[打开本条全部变化](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V03_S1909_I00000205&fixed=1&compare=production_diff1#fixedPairReview)

| 变化 | 旧全量最终通过：源秒／原文 | 最新版最终通过：源秒／原文 | 说明 |
|---|---|---|---|
| 边界扩展 · 12.157000–17.007000 秒 | 12.157000–16.925000：I think going to college now really reinforces that I like to learn. | 12.157000–17.007000：I think going to college now really reinforces that I like to learn. I | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：I。新增词不自动等于语义完整。 起点变化 +0.000000 秒，终点变化 +0.082000 秒。 |
| 重新分段：2→2 段 · 89.277000–97.620000 秒 | 89.277000–94.199000：Yeah, it sounded like that time off, you did a lot of reflecting and like what stood out? Were you saying that you,<br>94.300000–97.620000：I think I heard you say that you learned that you like to learn | 89.277000–93.079000：Yeah, it sounded like that time off, you did a lot of reflecting and like what stood out?<br>93.099000–97.620000：Were you saying that you, I think I heard you say that you learned that you like to learn | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 原先覆盖的可靠对齐词未减少；请结合原文判断留白或句子分组是否合理。 |
| 重新分段：2→2 段 · 130.896000–143.473000 秒 | 130.896000–140.962000：I was like, this is not that bad. I like that. It was all right. And then I got into the classes, classes, and I was like, man,<br>140.962000–143.473000：I like to learn. This is great. | 130.896000–135.830000：I was like, this is not that bad. I like that.<br>136.319000–143.473000：It was all right. And then I got into the classes, classes, and I was like, man, I like to learn. This is great. | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 原先覆盖的可靠对齐词未减少；请结合原文判断留白或句子分组是否合理。 |
| 边界扩展 · 287.669000–290.230000 秒 | 287.669000–290.010000：Sluggish. Where's it going? | 287.669000–290.230000：Sluggish. Where's it going? Is it | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：Is it。新增词不自动等于语义完整。 起点变化 +0.000000 秒，终点变化 +0.220000 秒。 |
| 新增通过片段 · 302.394000–305.604000 秒 | 无最终通过片段 | 302.394000–305.604000：The globe is... How do you pronounce that? | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：The globe is... How do you pronounce that?。新增词不自动等于语义完整。 |
| 边界扩展 · 332.905000–336.655000 秒 | 332.905000–335.707000：I did, but it took a while for my car to heat | 332.905000–336.655000：I did, but it took a while for my car to heat up. | 左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。 新增覆盖的对齐词：up.。新增词不自动等于语义完整。 起点变化 +0.000000 秒，终点变化 +0.948000 秒。 |

## 完整逐段文件与验收

- [全部逐段起止、原文、状态 CSV](production_intervals.csv)：包含不变片段，分最终通过／全部终端片段两层。
- [机器可读比较、旧结果指纹](production_comparison.json)。
- [数据指纹验收](production_validation.json)；[浏览器验收](production_browser_validation.json)：七条逐段时间／原文、39 组主对照与 103 次视频播放通过；未通过证据独立、Range 206、手机布局和历史链接检查通过。
- 原 HTML 版本下拉可选“旧全量版”，P1–P7 是旧全量展示编号，按用户列表顺序；H 编号保持历史链接稳定。L/H 和 B/H 的既往对照已折叠归档，不作为主比较。

## 复现

```bash
python experiments/pair_head_seven_20261029/compare_production.py
python experiments/pair_boundary_real_audit_20261029/publish_review.py
PYTHONPATH=/tmp/v8_browser_deps LD_LIBRARY_PATH=/tmp/v8_browser_libs/extracted/usr/lib/x86_64-linux-gnu python experiments/pair_head_seven_20261029/check_production_browser.py
```

只读七条指定全量结果与已有媒体索引；缺少的旧版视频串行导出，相同区间复用已验证媒体。未扫描或重跑全量。
