# 用户人工事件核验（2026-11-12，核验者01）

用户在会话提交旧版八个媒体的视觉32项、音频24项，共56项判断。按原判断和原中文 evidence 转存为标准CSV，rater_id保留字符串 `01`，不按模型答案更改。

- [原音频24项](submitted_audio_rater01.csv)、[原视觉32项](submitted_visual_rater01.csv)、[原始合并56项](submitted_combined_rater01.csv)。
- [新版可沿用45项](upright_carried_rater01.csv)、[完整56项表（11项空白）](upright_review_rater01.csv)、[仅待补11项](upright_pending_rater01.csv)。
- [逐项迁移依据](TRANSFER_AUDIT.json)：6个媒体完全一致，沿用42项；转正参考音频在同一样本同一角色侧的时长和波形SHA均完全一致，另沿用3项。该参考的4项视觉、重新生成素材的7项不沿用。
- [旧检测对照](old_detector_vs_human.json)、[新版检测对照](upright_detector_vs_human.json)与对应 `_differences.csv`。原56项对旧检测：视觉26/32一致；音频22/23确定项一致，另1项unknown。新版45项对新检测：视觉20/24一致；音频19/20确定项一致，另1项unknown。

新版视觉分歧：皱眉2项多报、点头1项漏报、微笑1项多报；音频有1项非词汇回应分歧（自动听辨为uh）、1项短词回应unknown。这里以用户判断为对照，不擅自改标签。人类标注没有摇头、可听笑声、非词汇回应正例，不能仅凭负例一致判断这几类的检测能力。只有一位标注者、八段素材，未校准阈值，也未完成独立正式可靠性验证。

[新版视觉页](http://localhost:8876/listen_benchmark_review/event_presence_upright_20261112/review_visual.html)预填24/32项；[新版音频页](http://localhost:8876/listen_benchmark_review/event_presence_upright_20261112/review_audio.html)预填21/24项。刷新后点击“定位未完成”，补视觉8项、音频3项，再导出。现有浏览器里已点选的新判断优先，不被导入覆盖。其他核验者的工作区不会注入01的答案。

`BROWSER_CHECK.json`验证预填、未完成导航、修改后刷新；浏览器导出与45项CSV逐项完全一致。`TEST_ONLY_seed_*.csv`是隔离浏览器测试输出，不作为新增标注者或第二次独立人工判断。

45项已通过原 `human_labels` 导入新版评测，报告在 `../seamless_upright_eval_fix_20261112/event_presence/human_detector_validation.json`；未核验类别保持未知，人工状态仍为部分完成。旧冻结结果未改写。

## 用户后续决定

用户接受大致正确的初步抽查，以检测流程对齐数据标注为后续目标。剩余11项保留未标注，不再要求补齐，不作为继续运行评测的门槛；已有人工判断不变，也不据这8个媒体单独调阈值。
