# 韵律提取器排查：全量暂停，标签定位未通过

> 后续用户已明确要求官方代码优先、以训练跑通为目标。全量已按官方完整生成与官方NULL处理恢复；本报告保留为诊断记录，不作为额外筛选或暂停门槛。[最新策略](../full/execution_policy.json)。

2026-11-10。全量controller与worker均按用户要求停止；`full/STOP`保留。仅使用node11 GPU7完成有界诊断，不恢复LIA，不触动其他任务。停前累计6,546块，所有缓存与原媒体保留。诊断没有写入生产label cache，也没有导出正式筛选清单。

**结论：当前失败主要反映提取器的结构与时间定位不可靠，不能据此淘汰原训练pair。补时长能让文本通过格式检查，但静音对照证明“能解析”不等于“标注正确”。全量保持暂停。**

[可播放的8条原音频/混合静音对照](index.html) · [诊断摘要](SUMMARY.json) · [参数消融原始回答](RESULT.json) · [静音/时移控制](../grounding_controls/RESULT.json) · [含语音但预测区间严格静音的控制](../mixed_silence_controls/RESULT.json)。

## 输入、权重和包装核验

- 5个权重分片SHA与[公开Qwen2-Audio-7B-Instruct资产](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct/tree/main)的LFS SHA全部一致，[逐文件记录](OFFICIAL_WEIGHT_IDENTITIES.json)。
- speaker源音轨SHA、参与者身份、源网格38400采样点、16kHz输入保持原约束。8条真实块RMS为0.071–0.154，峰值0.689–0.960，没有静音/空输入。8条log-mel特征SHA互不相同，mask均为240帧，对应2.4秒，不是把同一条音频复制给整个batch。
- ASR控制识别出了各自原转写中的不同短语，如“me be more confident in myself”“he would kiss her”“made fun of the beanie”。这支持音频进入了模型且样本对应正确，不表示做过精确逐词WER验收。静音ASR还会幻觉出中文，不能把ASR的文字输出单独当作有语音的证据。
- 诊断使用官方模型/分析类和原始回答；关闭生产early-stop。API关键字audio→audios适配仍明确保留。官方两个仓库相关源码无本地修改。

## 单因素/组合消融（8条已失败真实块）

| 条件 | 结构/时间通过 | 具体状态 |
|---|---:|---|
| official_greedy_inherited_rep1.1_max64 | 0/8 | {"parse_error": 2, "refusal": 1, "truncated": 5} |
| only_disable_repetition_penalty | 0/8 | {"parse_error": 1, "truncated": 7} |
| only_raise_token_limit | 0/8 | {"parse_error": 7, "refusal": 1} |
| rep1.0_max256 | 0/8 | {"parse_error": 8} |
| only_explicit_duration | 8/8 | {"events": 8} |
| duration_rep1.0_max256 | 8/8 | {"events": 8} |
| short_prompt_rep1.0_max256 | 0/8 | {"parse_error": 6, "refusal": 2} |
| full_original_pair_rep1.0_0 | 0/1 | {"parse_error": 1} |
| full_original_pair_rep1.0_1 | 0/1 | {"parse_error": 1} |

这些样本刻意来自先前失败块，比例只用于诊断，不能外推全量保留率。ASR和自然语言描述是独立控制任务，不使用prosody parser的通过数评价它们。

- 单独把max_new_tokens从64提高到256：仍0/8；多数回答只是继续输出更长的越界区间。
- 单独关闭重复惩罚：仍0/8。它改变数字/换行等输出形式，但没有解决定位。
- 仅补充真实2.40秒时长：8/8结构通过，6条完全相同地报告`1.39, 1.70, 0.50`，另2条同样主要报告1.39附近事件。
- duration＋无重复惩罚同样8/8结构通过，但主要机械输出1.29–1.67秒。这仍不能据此采用。
- 用原完整pair（8.567s/5.351s）替换2.4秒块也没有简单解决：仍输出到9.77s/10.37s，超出真实音轨窗口。因此短分块本身不是唯一原因。

公开模型[默认generation_config](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct/blob/main/generation_config.json)包含repetition_penalty=1.1。当前Transformers的该处理器也惩罚prompt中已出现的token，所以“贪心”不等于没有重复惩罚；但实测说明这一配置不是根因的充分解释。

## 反例：格式正确，标签仍错误

1. **全零2.4秒**：8/8都输出`1.39, 1.70, 0.50`，而PCM严格全零。
2. **真实2.4秒末尾追加2.4秒静音**：部分合法输出被移到3.29–3.70秒，整个预测区间都落在新增静音中。
3. **仍保持2.4秒，保留0–0.8秒真实语音、0.8–2.4秒置零**：8/8仍输出`1.39, 1.70, 0.50`。8个预测区间的实测RMS全部严格为0；整块RMS为0.0325–0.1006，全部高于官方可选min-RMS门限0.003。**仅启用整块静音门控不能修复这个反例。**
4. 语音后移0.6秒、尾部加静音时，输出时间没有可靠跟随声音的位置。模型对文本给定时长很敏感，不能把它生成的时间数字当成已验证声学定位。

第3项排除了“纯静音本来就不在实际数据域内，做个整块能量门控就好了”的简单解释：样本有真实语音、能量合格，模型仍在已知静音区间打出韵律事件。

## 处置和恢复前提

全量与原LIA均继续暂停，诊断进程完成后退出。6,546条现有记录保留用于故障追溯；其失败状态不作为原媒体质量不合格的结论，所有已有韵律标签仍未准入正式训练。未部署duration提示改法、未放宽解析规则、未重启全量。

恢复前必须更换或验证可用的韵律定位方法，并通过静音、含语音的混合静音、时移、实际时长与人工抽听对照。可以评估声学F0/能量等直接定位路线，或者另行设计不依赖不可信时间戳的条件；这些将是GLARE数据适配，需要明确披露。当前没有擅自切换监督定义。

之前把“完整条件子集在方法上可行”推进到全量，缺少必要的定位负对照；本次证据不支持继续用这套提取器的失败率筛掉训练数据。
