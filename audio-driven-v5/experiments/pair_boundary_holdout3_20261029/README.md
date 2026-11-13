# 三条新来源：当前切分规则的随机验证

> 本文记录历史 L 版。后续 H 版已通过句首补接保留 Okay／Yeah，并对未通过音质的 Right 扩展完整回退；见[五来源补接验证](../pair_head_attach5_20261029/README.md)。本页历史对照仍可访问。

本轮比较上一版 5 秒规则（G）和当前“局部 backchannel 分类＋真正句末优先”（L）。切分模块与上一轮相同，不在验证过程中修复新发现的问题。

抽样使用 seed=20261030：打乱生产 catalog，取满足 120～600 秒、已有生产结果和归一化音轨的前 3 条，排除之前审计缓存及 HTML 中的来源。**在运行两版规则之前固定样本，没有按改进效果筛选。** [抽样合同](selection.json)。两条 naturalistic，一条 improvised；小样本结果不能外推全量发生率。

| 新来源 | 时长 | G→L 初始候选 | 边界完全不变 | 差异组 |
|---|---:|---:|---:|---:|
| naturalistic_V02_S5228_I00000302 | 176 秒 | 19→19 | 19 | 0 |
| naturalistic_V03_S2098_I00000195 | 397.9976 秒 | 44→44 | 44 | 0 |
| improvised_V00_S0862_I00000794 | 442 秒 | 54→55 | 48 | 5 |

前两条的初始候选、局部恢复终端及音质通过结果均保持一致。初始音质待复核父片段数量均保持 2、3、1 段；局部恢复后终端通过数为 21、45、54→55。两版共 253 条媒体记录（G 126、L 127），基线 20 条重算 QC、106 条复用，当前版 4 条重算、123 条复用；规则通过不代表语义完整。第三条不是所有变化都合理：发现一处真实词覆盖退步，因此当前版不应仅凭原两条成功例子直接推广全量。

## 直接试听

- [来源一：全部 19 段边界不变](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S5228_I00000302&fixed=1&compare=unchanged#fixedPairReview)
- [来源二：全部 44 段边界不变](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V03_S2098_I00000195&fixed=1&compare=unchanged#fixedPairReview)
- [来源三：重点看 Okay 遗漏](http://127.0.0.1:8875/listen_full_source/?pair=improvised_V00_S0862_I00000794&fixed=1&compare=change3#fixedPairReview)
- [来源三：句末优先的变化](http://127.0.0.1:8875/listen_full_source/?pair=improvised_V00_S0862_I00000794&fixed=1&compare=change1#fixedPairReview)

原 HTML 已增加三条来源，各自提供 G/L 切换、所有候选与音质疑点、完整双画面源视频和人工记录。差异区列出全部 5 组，不只展示改善例子；无变化的两条各展示中间一段左右相同片段。播放范围为实际 pair，没有额外上下文。原有四条来源及历史链接保留。

## 第三条的全部变化与判断

1. **25.350–43.298 秒：2 段变 3 段。** 旧版把 `Like,` 留在前段末尾，新版把它放回下一句，并在真正句末切分。已覆盖词未丢失，语义结构更合理；但中间一段尾部额外留白为 0，仍需试听声学收尾。
2. **169.639 秒附近：末尾延长 31ms。** 同样文字，未增加或丢失 speaker 词，属于留白变化。
3. **213.343–213.963 秒：新版遗漏 B 的 `Okay.`。** 旧 G 片段为 213.143–220.346，新 L 片段从 213.983 开始，直接进入 `Well, it's going to work.`。`Okay.` 到 `Well` 仅隔 20ms，原版把它们放在同一个 B 主讲片段。
   新分类先将粗转写 `But... Okay.` 按超过 5 秒的停顿拆开，`Okay.` 变成反馈，不再参与主说话区间生成；它又离前一位说话者结尾较远，未接入前段，最后从新候选中消失。它应该归哪一种回应可人工讨论，但旧版已保留的发声丢失是明确退步。后续修复应解决反馈与随后正式发言的衔接，而非全局放宽安全余量。
4. **359.759–365.110 秒：补回 `it's`。** 起点比旧版提前 341ms，补回 359.819–359.959 秒的单词。仍是带原始词对齐提示的句子片段；新起点进入对方 VAD 活动区（359.680–359.950 秒），带 `start_crosses_listener_vad` 提示。虽然通过原音质规则，仍需试听重叠声音，不能仅凭补回一个词判为改善。
5. **366.316 秒附近：起点提前 38ms。** 文字和终点不变，仅留白变化。

[单独启用改动的只读试算](ablation.json)：仅改句末排序时，`Okay.` 保留；仅改局部反馈分类时，它已丢失。因此这是局部分类与角色区间衔接的问题。此试算只用于归因，HTML 两侧仍展示原封不动的 G/L 规则结果。

## 证据与运行

- [全候选、终端与音质通过词覆盖比较](comparison.json)
- [上一版媒体及质量证据](review/gap5/manifest.json) · [当前版媒体及质量证据](review/localbc/manifest.json)
- 253 条新旧媒体记录全部解码通过；浏览器核对 127 条新版媒体、播放 17 个对照视频，桌面／手机无溢出，旧版链接、音轨切换及反馈导出通过。
- [数据、边界与冻结文件验收](data_validation.json) · [浏览器验收](browser_validation.json)
- 原规则、原音质门槛、2～11.95 秒时长约束均未修改；模型使用 CPU。按相同起止时间、角色及归一化音轨合同复用 QC 证据，其他区间重新计算。原生产切分和训练未改动。
- 运行期间发现宿主机透明大页分配停滞：NumPy 1024×1024 网格分配重复停在复制环节，进程级禁用 THP 后约 0.01 秒完成。执行入口仅对自身及子进程设置 `prctl(PR_SET_THP_DISABLE)`，未改宿主机配置、模型参数或数值计算；中断的运行从精确边界缓存续跑，保留原日志。

在项目根目录使用现有 Python 环境：

```bash
python experiments/pair_boundary_holdout3_20261029/prepare.py
python experiments/pair_boundary_holdout3_20261029/run_variant.py gap5
python experiments/pair_boundary_holdout3_20261029/run_variant.py localbc
python experiments/pair_boundary_holdout3_20261029/render_sources.py
python experiments/pair_boundary_holdout3_20261029/compare.py
python experiments/pair_boundary_holdout3_20261029/export_pairs.py
python experiments/pair_boundary_holdout3_20261029/publish.py
PYTHONPATH=/tmp/v8_browser_deps LD_LIBRARY_PATH=/tmp/v8_browser_libs/extracted/usr/lib/x86_64-linux-gnu python experiments/pair_boundary_holdout3_20261029/check_browser.py
```
