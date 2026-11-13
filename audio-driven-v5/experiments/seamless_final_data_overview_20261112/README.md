# Seamless最终数据与开放标注统计（2026-11-12）

本目录供绘图及汇报使用，未生成图像。只读取冻结统计数据库和最终成员索引；没有调用模型、读取媒体、重写标签或改变训练数据。

统计来自转正后的最终自然语言caption对应的v3开放行为提取。旧690,337条SFT、旧285,638条500h、旧4,764名称概览均不是当前口径。

| 范围 | 片段数 | 小时 | 开放名称数 | 视觉名称 | 音频名称 |
|---|---:|---:|---:|---:|---:|
| 最终处理库（train＋dev＋test） | 1,107,313 | 1,856.2412 | 4,830 | 4,687 | 143 |
| LTX最终500h（原train＋dev） | 284,218 | 500.0013 | 2,605 | 2,525 | 80 |
| 最新Qwen SFT train | 704,368 | 1,191.9879 | 3,214 | 3,083 | 131 |
| 实际训练数据去重并集 | 797,599 | 1,354.2351 | 3,772 | 3,638 | 134 |

LTX500h包含原train 281,060段／494.1701h和原dev 3,158段／5.8312h，按此前确认的范围实际供LTX使用，未把源split改名。Qwen实际SFT仅使用train 704,368段；配套dev 5,296段／9.2922h、test 7,087段／11.8259h不进入Qwen梯度训练。这里的Qwen筛选test清单不等于另行冻结的Seamless生成评测6624／200条清单。

两份实际训练集交集190,987段／337.7541h，LTX独有93,231段／162.2472h，Qwen train独有513,381段／854.2338h。因此实际训练数据去重并集是797,599段／1,354.2351h，不能把500h和1,191.99h直接相加。

## 统计单位

- `label`：原始采用的开放名称，含视觉与音频前缀；保留同义、近义及细分名称，不强制归并，不代表4,830个已验证互斥的backchannel语义类别。自然语言caption仍是训练监督本体。
- `clip_count`：包含该名称的片段数，每个片段每个名称只计一次。不是动作发生次数。一段可以含多个名称，类别条数、占比不能相加为数据总量。
- `covered_clip_hours`：这些片段的完整长度之和，不是该动作真实持续时长。
- `literal_occurrences_sum`／`derived_occurrences_sum`：只累计来源中可计的明确／支持性推导次数；未知次数保持未知，不能用覆盖条数冒充次数。持续状态组另存`state_group_sum`，不与发声或动作次数相加。
- 无采用名称不代表人工确认无回应。全库37,862段、LTX500h 9,901段、Qwen train 16,711段仍保留。
- 含音频正向名称也不等于波形非零或人工确认有声；LTX500h含音频名称74,051段，而原数据摘要净化后波形非零95,757段，两个口径不同。
- 4,830名称中2,005个只覆盖1段，3,619个不足10段；长尾包括稀有动作和命名细分，不能据此宣称4,830种独立行为能力。

## 直接取用的CSV

- [数据规模与类别数](dataset_summary.csv)：含全库、各split、LTX500h、Qwen train/dev/test及实际训练并集。
- [类别并排宽表](category_comparison_wide.csv)：4,830行，四个主范围的条数、占比、覆盖片段小时，缺失类别填0。最适合直接画分组柱状图。
- [全类别长表](category_counts_long.csv)：各范围／split的类别覆盖、可计次数、未知次数和状态组；适合筛选和透视表。
- [全库全部类别](full_all_categories.csv)、[LTX500h全部类别](ltx500h_all_categories.csv)、[Qwen实际train全部类别](qwen_sft_train_categories.csv)、[训练并集全部类别](actual_training_union_categories.csv)。
- [Top30类别对照](top30_categories_comparison.csv)、[Top20视觉对照](top20_visual_categories_comparison.csv)、[Top10音频对照](top10_audio_categories_comparison.csv)：仅按全库覆盖条数排序取前项，未修改原类别。
- [人脸大小、片长、每段类别数分布](distributions.csv)：人脸是原图短边p10/p50像素，不是512×512裁框大小；分桶为左闭右开。
- [仅视觉／仅音频／视听／无采用名称](modality_coverage.csv)、[训练集交并关系](training_set_overlap.csv)。
- [未采用的待核观测](pending_only_labels.csv)：不计入主类别覆盖；这里是未采用观测，不保证同名没有其他采用记录。

## 主要类别覆盖条数

| 原名称／中文显示 | 全库 | LTX500h | Qwen SFT train | 训练并集 |
|---|---:|---:|---:|---:|
| visual:head nodding／点头 | 751,609 | 203,050 | 522,745 | 579,011 |
| visual:smiling／微笑 | 391,881 | 97,014 | 242,358 | 274,119 |
| visual:gaze shifting／视线转移 | 216,755 | 53,841 | 135,993 | 154,472 |
| audio:speech／口头回应／说话 | 201,938 | 58,059 | 197,526 | 198,400 |
| visual:blinking／眨眼 | 188,272 | 44,127 | 118,260 | 133,127 |
| visual:head shaking／摇头 | 164,946 | 39,874 | 100,969 | 115,139 |
| visual:mouth opening／张嘴 | 145,628 | 26,139 | 89,659 | 98,423 |
| visual:eyebrow lowering／眉毛下压 | 110,085 | 38,153 | 78,169 | 88,043 |
| visual:eyebrow raising／抬眉 | 106,000 | 20,077 | 69,029 | 75,499 |
| visual:mouth movement／口部运动 | 93,325 | 18,275 | 55,942 | 62,439 |
| visual:head tilting／头部倾斜 | 76,375 | 20,421 | 48,671 | 55,381 |
| visual:frowning／皱眉表情 | 72,040 | 10,251 | 49,064 | 52,014 |
| visual:hand raising／抬手 | 70,378 | 15,841 | 43,757 | 49,367 |
| visual:eye closing／闭眼 | 67,655 | 16,231 | 43,343 | 48,586 |
| visual:brow lowering／眉部下压 | 64,790 | 23,588 | 43,845 | 50,486 |
| audio:vocalization／其他发声（未细分） | 34,094 | 13,048 | 33,547 | 33,646 |
| visual:lip pursing／抿嘴／噘嘴 | 31,574 | 7,480 | 19,633 | 22,222 |
| visual:eye opening／睁眼 | 29,708 | 7,536 | 19,208 | 21,689 |
| visual:hand lowering／放下手 | 26,737 | 6,658 | 16,853 | 19,216 |
| visual:eye widening／睁大眼睛 | 22,133 | 3,279 | 14,286 | 15,376 |

中文仅为部分常见名称的显示翻译，不作为新类别，不把眉部下压自动改写为皱眉或惊讶。

## 来源与复核

- 全库：`experiments/listen_j_upright_av500h_v3_20261105/tables/sample_stats.sqlite`，1,107,313段；相对早期1,107,323段，9段转正后画质排除、1段held。
- LTX：`experiments/listen_j_upright_av500h_train_dev_v3_20261105/selection/index.sqlite`。
- Qwen：`experiments/listen_j_upright_response_sft_20261105/final/index.sqlite`。
- 三者共同绑定最终标签SHA256：`0851714c52e17dec6c7b5145b82dc53998ab09af0cc7ceda0b9c807192c2b389`。

[完整机器摘要](summary.json) · [逐项核对与CSV摘要](AUDIT.json)。已对照冻结原表核对全库4,830类、LTX全部2,605类的条数与覆盖时长；Qwen全部成员、源split、总条数／时长及训练交并集验证通过，分布分桶均回加至对应总量。

复现命令（CPU单进程，不读取音视频）：

```bash
/private/software/conda/bin/python scripts/export_seamless_final_data_overview_20261112.py \
  --output experiments/seamless_final_data_overview_REPRO
```

主统计CSV由此脚本生成；TopN只是宽表的排序切片。
