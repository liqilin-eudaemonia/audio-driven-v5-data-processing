# 整段事件类别一致性报告

**Pilot／外部检测结果，尚非正式benchmark。** 不评价语境适当性或生成内部视听协调性；不匹配GT时间，不因重复或双空加分。
主宏平均类别由冻结参考正例支持确定；无参考正例类另报FP／误报率。未知和技术失败不作负例。
人工核验：pending。GT自对照只验证执行一致性，不能证明检测语义准确。

## stage2_dev

| 方法 | 模态 | 完整集Macro-P | 完整集Macro-R | 完整集Macro-F1 | 有效draw/请求draw |
|---|---|---:|---:|---:|---:|
| stage2 | visual | N/A | N/A | N/A | 0/1 |
| stage2 | audio | N/A | N/A | N/A | 0/1 |

逐类TP/FP/FN、参考支持、失败／未知ID及每类共同可评集合见summary.json；逐类数值见per_class.csv。
跨方法共同子集另列在common_subset_draws，不能把不同子集的条件均值直接排名。

