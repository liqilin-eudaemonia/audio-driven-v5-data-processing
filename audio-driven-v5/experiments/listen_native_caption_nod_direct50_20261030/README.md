# 音视频＋点头线索：独立重新打标50条（不输入旧段落）

2026-10-30，按用户要求将此前“修订旧段落”的测试改为真正的独立打标。50 条全部重新调用 Qwen3-Omni，没有把原有自然语言描述作为输入，也没有复用无点头候选样本的旧结果。

[有声复核 HTML](http://localhost:8876/listen_benchmark_review/native_simple/natural_caption_nod_direct/)。页面展示本轮独立输出、实际完整 prompt、点头时间按钮和人工复核入口；没有“补充前段落”输入栏。历史事件仅折叠展示为参考，未输入模型。

## 实际输入与执行

- 媒体：历史 `native_simple` 同一批 50 条，listener 512×512 裁框、原生视频8 fps、净化 listener 音轨。播放端50条净化波形与实际读取入口哈希一致。
- 补充点头：复用同人物、同裁框和同一起点的 GLARE recall 结果，38条有50个候选区间，12条为空列表。
- 实际发给模型的区间仅包含 `id/start/end/observation`。Prompt 解释四个字段，说明起止时间相对当前 pair 起点，删除 `status` 和帧率说明。
- 不含 `PRIOR PARAGRAPH`，不含旧 caption，不要求修订或保留旧段落。空区间列表也照常从媒体打标，不能解释为确定没有点头。
- 输出：一段英文自然语言，表达动作顺序、持续状态、同时发生与真实重复，引用听清的发声原词。当前原文完整保留，不事后改写模型错误。
- node4 `172.29.225.51` 启动检查时八卡均空闲，每卡一个副本、显存比例0.85；0–7卡分别完成6/6/6/7/6/6/6/7条。使用既有单请求完整prefill、decode-only CUDA Graph、两路CPU预准备、temperature=0、repetition_penalty=1.05、4096 token预算及单段文本格式约束。

## 结果与问题

50次独立推理完成，全部非空；49条正常结束，1条重复生成后被长度上限截断。**本轮没有通过全部输出正常结束的质量门槛**，不能写成“50条全部验收通过”。

- 截断：`improvised_V03_S1806_I00000480_local_57844a87e674`，达到4096 token，多次重复同一句点头描述。原始长文本完整保存，页面可滚动查看。
- 明显静音发声幻觉：S5288、S0173、S2110、S5259 的模型输入全为零，却分别写出 “I don't know.”、“I'm not sure.”、“I don't know.”、“I think,” 等说话内容。页面标为复核问题。其他未引用具体词的虚构发声也可能存在，未做穷尽准确率统计。
- 点头：38条有补充线索，其中27条文本出现 nod 词（包含上述截断输出）；关键词数量不是有效点头数、真实动作数或准确率。另有无候选样本在否定句或元数据解释中提及 nod，不能计作检出。
- S0805仍写出两次 Right 和五次yeah，并提到片尾点头，但没有明确描述其早先点头线索。S5193写出Mhm和Yeah，仍有thoughtful／contemplative推演词。
- S0326等样本把“provided supplementary data”写进正文，违反了自然语言描述应只写观察内容的要求。页面新增元数据泄漏提示。

这些是这轮实际测试暴露的问题。没有用旧段落填补失败，也没有把错误描述加工后冒充模型原始输出。

## 文件与复验

- `manifest.json`：只有原媒体、静音配方和点头来源信息；点头补充字段不含旧段落或旧结果引用。
- `prompts/`、`prompt.txt`：本轮实际任务说明和逐样本请求。
- `results/`：全部50条新原始结果，含渲染后prompt、输入证据、生成终止原因、卡号和耗时。
- `protocol.json`、`preflight.json`、`code/`、`code_hashes.json`：冻结的配置、输入检查及推理源码。
- `audit.json`：输入身份和独立推理检查通过，旧段落输入0、复用输出0、四字段检查通过；`passed=false`明确保留截断导致的输出验收失败。
- `browser_validation.json`：播放与区间跳转正常，50条浏览器净化PCM哈希一致，旧段落面板隐藏，页面无JS错误。
- `delivery_code/`：最终展示、准备和审计代码的本地备份。

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
python scripts/audit_native_caption_nod_direct.py --run experiments/listen_native_caption_nod_direct50_20261030
python scripts/build_listen_native_caption_review.py --run experiments/listen_native_caption_nod_direct50_20261030 --output-name natural_caption_nod_direct
```

Prompt入口：`src/listen_benchmark/native_caption_nod.py`。准备入口：`scripts/prepare_native_caption_nod_direct.py`。GPU入口：`scripts/run_listen_native_caption_pilot.py`，协议任务名 `nod_guided_direct_annotation`。未启动全量处理或训练。
