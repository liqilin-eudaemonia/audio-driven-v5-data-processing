# 原文标注 → Gemma逐样本指导推理

按2026-10-27确认的方案实现。输入仍是speaker音频与listener首帧，经原Qwen/Connector得到U；Gemma的正向条件改为固定场景说明加本片段整合标注。不是新的训练实验。

## 文本合同

- 输入现有 `listen-existing-result-fusion-verbatim-v2` 整合JSONL；不重新打标、翻译、概括、修改情绪解释或补写回应。
- 按相对片段起点的时间排列“视觉回应／音频回应”。使用原有效 `intervals`，没有时使用原 `start/end`，不四舍五入边界，不填补不连续区间。
- 音频描述和发声文本合为“音频回应”：二者不同则并列，具体词句加引号；完全相同或描述已包含完整引号内的同一句时只展示一次。不会把子串误当成重复，例如Right和Bright。
- 原事件和整个源记录保存在每条prompt JSON内。每行关联事件ID；所有原描述、转写、来源及不确定性均可复核。
- 只沿用输入已经做过的去重；不进一步合并事件。不把视觉laugh推断成笑声，不把unknown补成静音。
- 后处理数据应传入同版本的整合标注；模块不会自行复活被删除的音频事件。原始数据与后处理数据使用不同实验目录。

示例结构（占位符，不对应某条真实样本）：

```text
<原固定场景说明>

[原开始时间–原结束时间秒]
视觉回应：<原视觉描述>
音频回应：<原音频描述>；发声内容："<原转写>"
```

## 编码与模型边界

新增 `src/p2_guided_prompt/` 和 `scripts/p2_guided_prompt.py`，未修改冻结的p2_listen训练/生成代码。单进程推理包装器临时替换native generator的文本加载函数，退出时恢复；不作为多线程共享生成服务使用。

Gemma与LTX原文本处理器保持冻结，按原方法产生video/audio两路文本特征。正向文本按1024/2048/4096分档补齐（均满足connector寄存器长度要求），默认上限4096；可指定1024或2048。每次检查实际token ID序列及attention mask，禁止静默截断。超过配置上限明确拒绝，不删标注。长文本是本次推理接入扩展，质量及时间遵循能力需另行评测，不宣称模型保证按时间戳执行。

负向CFG文本严格复用旧缓存并逐张量核对；speaker/首帧输入、Qwen U、权重、LoRA和采样配置保持原样。新增文本缓存按完整指导内容、标注记录、代码、文本长度设置和Gemma/LTX资产建立身份，不覆盖原固定prompt缓存。

较早权重源码需恢复时，仅使用训练记录SHA匹配的历史副本；所有其他源码检查正常执行。四组权重由现有source-aware bootstrap选择对应训练包。

## 命令

在项目根目录执行，Python使用原训练虚拟环境路径（不要resolve为系统Python）：

```bash
PY=../Omni-LTX-stage2/.venv/bin/python

$PY scripts/p2_guided_prompt.py prepare \
  --observations <observations.json> \
  --annotations <integrated_labels.jsonl> \
  --output <新实验目录> --max-text-tokens 4096

$PY scripts/p2_guided_prompt.py cache-text \
  --work <新实验目录> --run <权重对应run目录> --device cuda:0

$PY scripts/p2_guided_prompt.py generate \
  --work <新实验目录> --run <权重对应run目录> \
  --checkpoint <step_001268.pt> --sample-id <sample_id> \
  --output <独立输出目录> --seed 137 --device cuda:0 \
  --variant guided
```

`cache-text --sample-id` 可重复指定，只编码选定样本。`generate --variant fixed` 使用原固定文本作同输入/seed控制。E0/E1历史源码恢复可加：

```text
--restoration experiments/p2_train24_fourway_eval_20261027/source_restoration.json
```

## 评测身份

指导包含真实目标标注，元数据明确记录 `target_annotation_access=true`、`evaluation_role=target_annotation_guided_diagnostic`；不读取listener目标视频或音频。它是给定回应计划的生成诊断，不能混入原“只给speaker/首帧”的E0/E1正式成绩。

每个输出保留 `prompt.txt`、视频、独立音轨、权重和指导哈希；原生成目录不改。已有HTML不改。

## 实际产物与验证

- [原train24逐样本prompt](../experiments/p2_guided_prompt_train24_20261027/prompts/)
- [独立test24逐样本prompt](../experiments/p2_guided_prompt_test24_20261027/prompts/)
- [GPU验证与运行记录](../experiments/p2_guided_prompt_train24_20261027/README.md)

单测覆盖原文/原时间保留、模态分离、重复文本、unknown、不连续区间、内容篡改拒绝、冻结记录身份、CLI入口、临时加载器恢复与长度边界。GPU完成状态以实验目录的验证记录为准。
