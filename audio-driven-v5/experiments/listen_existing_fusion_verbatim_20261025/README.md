# 50段原文整合（2026-10-25，当前版本）

页面：http://127.0.0.1:8876/listen_benchmark_review/existing_fusion_verbatim/ 。使用已有结果在CPU上重放，未调用GPU、未重新分析媒体、未应用到全量结果。

## 本次修改

删除将重复“嘿”关联为笑声并替代独立词候选的规则；删除局部Yeah替换整段Ed等词语替代规则；删除Says模板。ASR候选使用参考转写的原词／原短句，其他来源的转写和描述继续作为原始证据保存，不决定哪个词正确。

视觉和非语言音频候选优先直接展示source_descriptions中的模型原文；跨窗关联有多个版本时，保存全部原文，并选其中一条用于主显示，不另写摘要。原文可能同时涉及多个动作，来源关联不意味着该整句话在每个关联小区间内全部成立。

保留已有去重、词序约束和候选时间选择逻辑，不将性质相同但时间分开的事件合并。七类字段沿用上游模型归类，不重新分类，也不用于替代描述。与词候选重叠的其他音频描述保留为来源差异／相关证据，不额外断言一次发声，也不根据内容作纠正。原始侧优先展示模型原文，原始窗口JSON与完整观测不变。

## 当前输出与限制

50段，175个独立候选（134视觉＋41音频），均有有效起止时间；另2个无法用已有区间定位的词单列失败。682条输入观测逐字节保留。ASR130条记录整理成85个候选索引，不等于真实发声次数。15段无独立候选，不能当作负例。

S2043保留三条“嘿”，已有笑声描述也保留；不根据人工反馈替换。S0835参考序列保留Ed，窗口Yeah仍在原始转写和候选时间证据里；选择参考序列不代表Ed正确。S0805仍为Right两次和Yeah五次，S0699两次Okay仍分开。取消内容修正后，候选数量变化不代表准确率提升。未验证全部边界达到±0.2秒，training_admitted=false。

## 实现与复现

src/listen_benchmark/existing_result_fusion.py为整合逻辑；scripts/build_listen_existing_fusion.py为构建入口；scripts/listen_existing_fusion.html为页面。生成过程无人工反馈输入，也无特定样本分支。

运行：

```bash
python scripts/build_listen_existing_fusion.py \
  --source experiments/listen_response_episodes_20261024T120000Z \
  --align experiments/listen_split_validation_20261024T074733Z/word_alignment \
  --out experiments/<新的输出目录> \
  --preview data_previews/listen_benchmark_review/<新的页面目录>
```

输出目录须为空，可选--font用于网页字体。observations.jsonl保留原始输入；review.json包含完整来源；labels.candidate.jsonl保存候选、关联和原文来源；manifest.json与code/保存哈希和代码快照。10项单元测试验证重复次数、原文保留、冲突词不替换、无时间不均分等；50段结果及浏览器检查见validation.json。
