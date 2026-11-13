# 最新Qwen规划LoRA：同24条样本推理对照（2026-11-02）

已完成未微调和最新LoRA各24条生成，48条均正常EOS、无截断。本次未微调基线与历史同prompt基线24/24条生成token完全一致。

[可播放对照页](http://localhost:8876/listen_benchmark_review/planning_sft_filtered_compare24_20261102/)（Cursor转发8876）；[独立HTML](review.html)内嵌全部实际输入图片与speaker音频。逐条结果见[comparison.csv](comparison.csv)、[assessment.json](assessment.json)，输入核对与权重哈希见[protocol.json](protocol.json)。

## 结果与结论

| 描述性统计 | 未微调 | 旧500h一轮 | 旧500h两轮 | 新1,155h一轮 |
|---|---:|---:|---:|---:|
| 正常结束 | 24 | 24 | 24 | 24 |
| 平均英文词数 | 10.0 | 82.9 | 81.8 | 80.4 |
| 引用不存在的补充信息 | 0 | 4 | 4 | 0 |
| 明确安排发声 | 0 | 1 | 2 | 5 |
| 明确安排静音 | 24 | 21 | 21 | 19 |
| 发声状态未充分说明 | 0 | 2 | 1 | 0 |

最新模型在输出规范性上有明确改善：旧两轮#1/#2/#4/#17虚构补充信息依据的问题没有再出现；全部24条明确交代发声或静音。新输出发声为#3“Yeah”、#6“Yeah”、#8“Oh yeah”、#23“Yeah”、#24短笑声。

回应的多样性和上下文适配仍不充分：18/24条安排一次点头，大量使用“保持姿态／中性表情＋一次点头＋静音”的句式。描述长度没有增加，发声数量增加也不代表更合理。仍有外观铺陈、报告式措辞和未经检测支持的精确动作时间；这些时间属于模型提议，不是实际检测标注。

4条原dev：#21由点头改为静止，没有明确优劣；#22闭眼代替抿嘴，其他变化较小；#23补齐开头“Yeah”，符合发声说明要求；#24由“Yeah”改为笑声，是否更符合speaker内容需要人工复核。因此结论是“格式／证据表述改善，未证明对话语义规划或最终AV效果提升”。

这些是助手基于24条完整新旧输出和既有speaker转写的定性检查；没有进行音频盲评、人工偏好统计或LTX生成评测。已有listener caption是弱标注，回应有多种可能，不能把与caption不同直接当错。新旧训练集与轮数均不同，也不能把变化单独归因于数据筛选或数据量。

## 输入、训练归属与权重

沿用原`p2_j_omni3b_autonomous24_20261101/requests.json`中的24条样本及顺序。实际调用原review cache读取器，24条最终裁框首帧RGB、16k speaker波形均与此前保存输入逐值一致。只输入speaker音频、listener首帧、时长及system/user指令，不输入未来listener视频、listener目标音轨、caption、转写或标签。

原抽样为20条train＋4条dev。最新训练包含其中12个train样本ID；另8条原train未按同ID进入新清单，但可能与训练源会话重合，不能当独立测试。4条原dev均未参与梯度更新，其中2条仍在最新dev清单、2条被筛掉；这不改变它们原本的留出角色。本轮没有换样本，页面优先展示原dev。

最新权重：`../p2_planning_sft_filtered_3b_24gpu_inputfast_isolated_20261102/final_adapter/`，原始Qwen2.5-Omni-3B＋语言层PEFT LoRA rank32。训练690,337条／1,155.44h，完成2,697步、1轮；COMPLETE成功，未执行dev loss验证。

所有比较使用相同SFT prompt、batch24、greedy、max_new_tokens512。旧零样本最初版本的prompt不同，不混入本次数字对照。当前推理脚本重放基线完全一致，也核对了system/user字符串与历史记录一致。

用户最初指定172.29.189.119空卡，但核查8卡均满载；按后续明确授权改在本机node12（172.29.252.37）GPU6和GPU7叠加短时推理，各进程显存上限40GiB。没有停止缓存或其他任务。两组推理已结束；生成阶段基线6.76秒、新LoRA23.23秒，含不同输出长度及与缓存共卡竞争，不能用来宣称模型速度差异。

## 复现

旧配置文件在此仅用于选取原24条review cache、相同模型和prompt；实际adapter由新路径明确指定。不要直接换成最新训练配置，否则会从新reference_jsonl抽出另一批dev。

```bash
PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
CUDA_VISIBLE_DEVICES=7 PLANNING_MEMORY_GIB=40 "$PY" scripts/p2_planning_generate.py \
  --config configs/p2_planning_sft_3b_32gpu.json \
  --adapter experiments/p2_planning_sft_filtered_3b_24gpu_inputfast_isolated_20261102/final_adapter \
  --output /tmp/planning_latest24_new_output.json --batch-size 24
python scripts/build_p2_planning_filtered_compare24.py
```

复跑前检查空闲资源／授权范围；输出文件必须不存在，防止覆盖历史结果。`base.json`、`latest.json`保留完整原文和token；历史一／两轮仍在原目录。`launch.json`保留实际命令、PID与GPU。
