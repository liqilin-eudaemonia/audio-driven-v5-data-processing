# 整段事件评测：线索传递检查与修复

日期：2026-11-12。返回[评测总索引](../../evaluation/README.md)／[模块说明](../../evaluation/event_presence/README.md)。

本次按用户要求检查七类事件评测中的线索传递遗漏，参考既有标注pipeline进行修复。不新增训练或评测类别，不改变P/R/F1、弱标签策略或已有动作检测门槛。

## 修复内容

| 问题 | 修复 |
|---|---|
| 微笑曲线被保存，却被提示构建的字段白名单丢弃 | 提取当前观察窗口内的mouthSmile左右均值，以每0.5秒的均值、峰值和有效帧数传给Qwen。峰值保留短暂变化；0.5秒只是提示压缩粒度，不是微笑持续时间或判别阈值 |
| 检测到脸但缺少某一侧blendshape时，用0代替缺失 | 两侧测量齐全才计算均值，否则保留null；真实零值仍为零 |
| ASR做过规范化，但原始transcript／nonlexical_syllables仍混入提示 | 复用标注规范化函数，只传规范化候选、发声说明和uncertain。原始响应保存在audio_*_asr_raw.json，不在提示中绕过过滤 |
| 长片段每个窗口使用整段人脸覆盖率，其他窗口漏脸也影响本窗口负例 | 保存逐帧人脸检测状态，按当前窗口计算覆盖率；动作时间与微笑摘要均转为窗口相对时间。旧缓存缺逐帧状态时明确使用保守的clip_fallback |

涉及代码：[hints.py](../../src/p2_event_presence/hints.py)、[候选提取](../../src/p2_event_presence/candidates.py)、[实际推理](../../src/p2_event_presence/inference.py)、[提示模板](../../src/p2_event_presence/prompts.py)。

## 检查后保留的行为

- 标注pipeline提供候选线索，最终类别仍由实际媒体判断；不直接从信号阈值生成微笑标签。
- 点头、摇头、皱眉的原有检测规则及方向／表情门控继续复用。
- 无声视频与纯音频分别检测；音频ASR与类别判断均来自同一侧波形，生成音轨不按GT发声区间剪切。
- unknown／error继续显式保留，不改作负例；多窗口存在性汇总、参考正例宏平均和共同可评集合维持原定义。
- 候选后端或ASR调用失败仍按原错误路径记录；不增加隐式回退标签或放宽原有判定门槛。

## 验证

原15项计分测试通过；新增7项线索回归测试，共22项通过：

```bash
python -m unittest discover -s tests -p 'test_event_presence*.py' -v
python experiments/event_presence_cue_fix_20261112/replay_prompts.py
```

回放使用原pilot的8个真实媒体缓存，重新构建8份视觉提示和6份音频提示；两条全零GT音频沿用直接静音判定，没有ASR提示。检查微笑线索进入实际模板、窗口局部时间和覆盖率、ASR原始内容过滤。长窗口局部缺脸、窗口外峰值排除和缺失测量由针对性测试覆盖，真实pilot只有短片段。

同时重新计算历史event_pairs.jsonl的分数，逐draw、逐类与共同子集结果和原summary一致；核验读取的历史媒体缓存／报告哈希均未变化。

结果：[PROMPT_REPLAY.json](PROMPT_REPLAY.json)；提示：[prompt_replay](prompt_replay)；测试：[test_event_presence_hints.py](../../tests/test_event_presence_hints.py)；修改前文件备份：[before](before)。

本次未启动GPU检测，也未生成新的事件标签或模型分数。验证证明线索传递和统计回归正确，不证明语义检测准确率提高；检测可靠性仍需人工核验。

## 版本与后续运行

检测提示版本为`annotation-pipeline-presence-visual4-audio3-20261112-v2`，默认配置中的detector改为`annotation-pipeline-qwen3-omni-eval-v2`。配置路径仍沿用`configs/p2_event_presence_v1.json`，其中v1是原文件名，不是此次提示版本。

已有source identity包含新hints模块；原工作目录绑定了旧源码与提示，不能继续运行新版本或混用新旧检测结果。后续按模块说明在新工作目录bind/run；历史pilot保持原样，本次未修改在运行的评分任务。
