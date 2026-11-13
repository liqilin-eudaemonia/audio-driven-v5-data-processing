# 整段事件评测：真实小规模验收

2026-11-08已完成检测、评分、报告与盲化播放核验。协议和使用入口见[评测模块](../../evaluation/event_presence/README.md)，[主评测索引](../../evaluation/README.md#整段事件类别一致性2026-11-08)已指路。

**这是执行pilot，尚无独立人工检测可靠性／阈值校准结果，不能作为正式benchmark成绩。** 没有使用这四条样本调整检测阈值或选择生成权重。

## 样本与真实执行

从已有6000步＋SFT2752完整生成清单中，按每数据集GT安静／短发声分组，先选不超过8秒的最短片段，再按ID打破并列；没有按生成质量或检测分数挑样本。共ResponseNet两条、Seamless两条，时长2.02–2.86秒；原始模型名、选择规则及ID见[SELECTION.json](SELECTION.json)。

四对真实GT／生成之外加入四对明确标识的GT自对照，共八条pair记录；相同的GT素材去重后为八个不同媒体。自对照验证媒体／标签复用及评分一致性，不证明语义检测准确。

运行使用node11 GPU7，复用Qwen3-Omni-Instruct BF16及原MediaPipe／反应候选；CPU offload 28GiB，单序列，未量化或替换权重。视觉输入为无声8fps，候选25fps；音频独立16kHz。同一冻结模型先在各自波形上生成ASR候选，再单独听辨，未使用另一侧转写或GT门控。两条GT全零波形直接记有效静音。

八个媒体均完成候选与原模型检测，没有媒体读取、候选后端、生成或解析技术失败。七类共56个媒体标签单元中55个确定、1个unknown：Seamless `naturalistic_V03_S0478_I00000065_local_316903151c3c`的参考“短词回应”。原始模型理由对非词汇`uh`与短词边界给出未决状态，原样保留，没有人工改成absent。该状态在自对照中也复用；Seamless完整音频主表因此为N/A，可评子集另列。

## 结果仅用于验收

| 数据集 | 实际模型pair | 视觉Macro-F1 | 音频完整集Macro-F1 | 音频可评子集Macro-F1 |
|---|---:|---:|---:|---:|
| ResponseNet | 2 | 0.6667 | 0.0000 | 0.0000 |
| Seamless | 2 | 0.5000 | N/A（参考有unknown） | 0.0000 |

宏类别由各数据集这次冻结参考中的正例支持确定。ResponseNet视觉支持点头／笑态／皱眉，音频支持短词；Seamless视觉四类均有支持，音频支持非词汇。小样本支持有限，不能把这个结果用于排名或推断全量表现。GT自对照的可评支持类F1为1；无正例类别仍为N/A且单列误报，没有通过双空获得满分。

此次真实样本没有可听笑声参考正例，均不足3秒且只有一个draw；不能声称已用真实媒体覆盖所有正例、长窗口或多draw场景。多draw及类别统计边界由CPU测试验证，长片段执行仍应在扩大pilot时核验。

- [逐类TP／FP／FN、覆盖和共同集合](run/summary.json)、[可读报告](run/REPORT.md)、[CSV](run/per_class.csv)、[逐pair标签](run/event_pairs.jsonl)。
- [模型及适配contract](run/detector_contract.json)、[原始模型响应和提示](run/media/)、[执行结束状态](run/RUN_STATUS.json)。`execution_complete_partial`表示有语义unknown，并非仍在运行或技术崩溃。
- [独立算术与身份核验](VALIDATION.json)：28张逐类计数表与宏F1独立重算一致；源码、媒体、检测SHA核对通过。重跑：`python experiments/p2_event_presence_pilot_20261108/validate_pilot.py`。
- [缺生成文件实测](MISSING_GENERATION_CHECK.json)：故意缺失生成视频／音频时，GT仍独立解码成功，生成明确保留两个错误；该控制不参与模型成绩。
- 15项CPU契约／计数／共同子集／多draw／未知边界测试通过，入口：`python -m unittest discover -s tests -p test_event_presence.py -v`。

本次Qwen模型GPU进程1284176已退出；结束后GPU7的compute进程列表中已无该PID，其余任务继续运行。只使用先前授权的GPU7。

## 盲化人工核验

[打开视频／音频分离核验页](http://localhost:8876/listen_benchmark_review/event_presence_pilot_20261108/)。页面隐藏方法名、GT／生成角色和自动标签；八个无声视频与八段原波形均可播放，手机布局无横向溢出。[浏览器验证](../../data_previews/listen_benchmark_review/event_presence_pilot_20261108/BROWSER_CHECK.json)。预览H264只用于人工观看，自动检测使用冻结的无损媒体。

空白标注CSV和盲素材索引已生成。填写`present/absent/unknown`、证据和标注者ID后，可按模块说明导入；空白不会当负例，人工分歧先保留unknown。当前[人工状态](run/HUMAN_REVIEW_STATUS.json)为pending；没有伪造人工准确率、校准结果、语境适当性或视听协调性分数。

正式发布前还需独立人工子集核验与校准、冻结后进行测试隔离验证，再按相同协议跑全测试集／方法。此次授权范围内的框架接入和小规模实际执行已完成。

## 过程留痕

`prep_attempt1_missing_media_dependency_path/`保留第一次媒体环境缺少PyAV路径的失败；随后复用已有paired-video依赖缓存成功，没有修改训练环境。`prep/`为成功的四对源媒体准备，`run/`是最终联合验收。权重身份目前为config／index SHA及分片大小／mtime，未宣称逐分片密码学核验。
