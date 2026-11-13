# 七条指定来源：H 版切分与旧结果对照

> **以此为准：用户明确要求的是“此前认可的全量版→现在最新版”。** 主比较已统一为 `data_corpora/seamless_listen_local_full/results` → H，原 HTML 已替换主对照。[七条真实全量对比、全部 39 组起止与原文](PRODUCTION_COMPARISON.md) · [全部逐段 CSV](production_intervals.csv)。结果为 205→206 段，159 段起止完全不变；第 1、2、6 条有旧覆盖内容未进入新版通过集。以下 L→H 和混合基线记录属于历史排查，不是用户要求的主比较。

> **比较口径纠正：** 下文旧表比较了“前两条历史 B→H”和“后五条 H 同版重跑”，不应作为统一新旧方案比较。现在已补齐前两条 L 版结果，七条统一比较 **L→H**，逐段列出初始候选和最终通过的起止时间、原文、音质状态：[相邻 L→H 完整时间表](ADJACENT_INTERVALS.md) · [历史 B→H 时间与原文表](HISTORICAL_INTERVALS.md) · [CSV](adjacent_intervals.csv) · [机器可读数据](adjacent_comparison.json)。七条最终通过共 206 段，204 段起止完全不变；只有 442 秒来源两段提前起点，终点不动。其余六条全部候选与终端片段的起止、原文、音质状态／原因均一致。Right 的第三处扩展仅为未通过提案，最终回退不变。历史 B→H 的覆盖减少仍存在，但不是最后这次 L→H 补接导致。

2026-10-29。结论：**未达到“基本不变、只是明确优化”的全量启动条件，未启动全量。** 本轮保持 H 切分和原音质规则不变，串行处理七条指定来源。生产数据、训练数据未替换。

## 对比口径

“之前”指本轮开始前 HTML 中实际发布的结果，已冻结到 `baseline_views/`。前两条尚未运行过 H，基线为 2026-10-23 连续审核版；后五条基线为上一轮 H 验证结果。因此，后五条不变表示 H 结果可复现，不代表 H 与最早生产版本相同。页面仍保留 L/G/F 历史对照。

下表均为**最终通过片段**，不是初始候选数。覆盖秒数包含切片留白；未保留词数按已有可靠词对齐计算，不是人工听写准确率。

| 来源 | 时长 | 旧→新通过段数 | 完全同边界段数 | 旧→新通过秒数 | 原通过词未进入新通过集 |
|---|---:|---:|---:|---:|---:|
| 1. `naturalistic_V02_S3610_I00000088` | 170.000 | 19→18 | 12 | 96.826→83.124 | 25 |
| 2. `improvised_V01_S0122_I00000125` | 196.000 | 9→8 | 7 | 48.428→41.593 | 23 |
| 3. `naturalistic_V02_S5228_I00000302` | 176.000 | 21→21 | 21 | 144.738→144.738 | 0 |
| 4. `naturalistic_V03_S2098_I00000195` | 397.998 | 45→45 | 45 | 329.519→329.519 | 0 |
| 5. `improvised_V00_S0862_I00000794` | 442.000 | 55→55 | 55 | 308.202→308.202 | 0 |
| 6. `naturalistic_V02_S3579_I00000175` | 190.000 | 20→20 | 20 | 142.667→142.667 | 0 |
| 7. `naturalistic_V03_S1909_I00000205` | 353.987 | 39→39 | 39 | 219.928→219.928 | 0 |

## 需要复核的变化

前两条没有触发 H 的短词补接；差异来自此前累计切分、局部恢复与音质判断流程。48 个原通过对齐词都仍存在于新初始候选里，随后未进入最终通过集。切分边界改变了音质评估窗口，局部恢复又未保回全部旧内容；目前不能断言它们都是误杀，也不能把减少当作明确改善。

- 170 秒来源：原通过覆盖减少 13.702 秒，25 个词未进入新通过集。涉及 21.340–31.350 秒的参考匹配／稳态背景疑点、32.337–43.320 和 43.340–48.430 秒的环境声疑点，以及 104.440–110.9085 秒的参考匹配疑点。“Cake. Cake.” 原先有通过片段，现在只有待复核提案。
- 196 秒来源：原通过覆盖减少 6.835 秒，23 个词未进入新通过集。包括 kindergarten，以及原 110.126–116.150 秒完整话语。新 102.418–112.268 秒因 listener 背景底噪排除，新 112.288–116.150 秒因稳态背景疑点待复核。
- 170 秒来源另外两处结尾新增下一句开头的 He’s / He；不能把新增词数直接视为改善。还有一处仅减少 31 毫秒留白、词覆盖不变。

## HTML 入口

[七来源原审核页](http://127.0.0.1:8875/listen_full_source/)。

- [170 秒：全部七组变化，从第一组开始](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3610_I00000088&fixed=1&compare=seven_diff1#fixedPairReview)
- [170 秒：Cake. Cake. 未进入新通过集](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3610_I00000088&fixed=1&compare=seven_diff4#fixedPairReview)
- [170 秒：新增下一句开头 He’s](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3610_I00000088&fixed=1&compare=seven_diff5#fixedPairReview)
- [196 秒：原完整话语与新版未通过提案](http://127.0.0.1:8875/listen_full_source/?pair=improvised_V01_S0122_I00000125&fixed=1&compare=seven_diff2#fixedPairReview)

左右均为实际完整 pair，没有额外上下文。右侧未通过候选明确标出“待音质复核／排除”，不计入新保留集。B6/B7 是前两条的历史版本；H6/H7 是本轮新版。H1–H5 沿用此前编号，以免破坏旧链接；页面导航来源 1–7 按本轮用户列出顺序。

## 复现与证据

```bash
python experiments/pair_head_seven_20261029/run.py
python experiments/pair_head_seven_20261029/analyze.py
python experiments/pair_head_seven_20261029/export.py
python experiments/pair_head_seven_20261029/run_previous_l.py
python experiments/pair_head_seven_20261029/compare_adjacent.py
python experiments/pair_boundary_real_audit_20261029/publish_review.py
PYTHONPATH=/tmp/v8_browser_deps LD_LIBRARY_PATH=/tmp/v8_browser_libs/extracted/usr/lib/x86_64-linux-gnu python experiments/pair_head_seven_20261029/check_browser.py
```

重算和导出串行进行；已有完全相同边界的证据／媒体按精确区间复用。不扫描全量来源。进程内关闭 THP 避免已复现的 NumPy 分配停滞，不修改宿主机配置。

- [机器可读比较与全量启动条件](comparison.json)
- [七来源候选、对照视频与历史版本](review/manifest.json)
- [数据验收](validation.json)：H 模块指纹与上轮相同；38 个冻结代码文件指纹相符；原五条 H 结果不变；48 个词均在新初始候选内。
- [浏览器验收](browser_validation.json)：七条逐段起止／原文表与 45 个对照视频串行验收通过，Range 请求为 206；候选筛选计数、历史版本与旧链接、390px 手机宽度检查通过，无页面脚本错误。

本轮仅验证与展示，没有继续调整切分规则。后续应先判断两条来源的音质筛选与旧通过内容保护，再决定是否全量。
