# 方案A：原文截取与去重选择——检查、修复与交接

2026-10-27。按用户明确选择的方案A修复。模型为本地 `/private/ckpts/Qwen3-30B-A3B-Instruct-2507`，仅输入已有文本。原始标签、媒体、音视频latent和正式训练配置没有修改。

## 检查发现

原有修改并未完全符合方案A：

1. `compile_plan`仍调用正则`strip_inference`生成最终描述，而非使用模型选定的原文子串。该正则对许多`showing ...`一律截断，也可能误删可见细节。
2. 默认`strict=False`会自动提升、交换保留条目、改接support目标；部分改接只依据少量动作关键词，无法证明语义相同。
3. 错误quote会被静默替换或修正大小写。即便输出看似原文，也不是严格验证模型的选择。
4. 动作检查有时比较目标的完整原文，而非真正输出的quote；被截掉的动作可能仍被算作“已经覆盖”。
5. 运行脚本可能复用旧响应并重写协议哈希，review可能针对旧提案；集群入口还有跨版本响应复用。
6. HTML构建脚本固定读取旧8B目录、写死100/66/34及旧统计；逐条`zip`音频比较也无法发现列表尾部丢失。

修改前源码与哈希保存在[检查目录](../experiments/listen_semantic_plan_a_audit_20261027/before/)。既有运行的冻结源码、结果和页面保留，不能把它们视为新方案的验证结果。

## 当前约束

- 模型只输出`source_event_id/action/quote/targets/reason`和冲突，不再提供可自由改写的`description`。
- `quote`必须从原文第一个字符开始，是原文精确前缀。代码只规范句末标点，内部措辞、主语、时态、分词开头不变。原文没有主语时不会凭空补出“The listener”。
- 预先列出通过边界检查的原文前缀供模型选择，结构化解码限制引用只能取这些原文片段；代码不生成新句子，不因模型错误而自动修补。
- 截掉的内容必须是可识别的推演尾部，`showing teeth`等可见信息不能当作推演删去。若推演之后还有新动作，无法安全使用一个连续前缀，则明确拒绝，不能静默漏掉。
- 模型从同语义、时间重叠的来源中选择可见信息最完整的一条。原始分开的发生不能仅因描述相同而合并。
- support只指向一个完整的keep来源；不含其显式动作的条目会在结构化解码之前从可选目标中排除，而非在生成后自动改接。模型仍判断候选间是否语义等价。
- 决策数组按输入来源固定位置输出，避免大量`anyOf`对象分支导致约束编译耗时。实测S1400完整语法编译约1.8秒，最长样本约8秒，不使用GPU完成该编译。
- vLLM设置`structured_outputs_config.disable_any_whitespace=True`，禁止JSON结构层无限空白循环；原文字符串中的空格和文字不受影响。模型自检只看一次主动输出，不把证据引用副本重复列为输出，完整证据仍保存。
- 动作遗漏检查比较support的可见quote与实际保留quote；覆盖不足、循环support、错误时间／来源等均拒绝并重试，没有自动改选、改接、替换原词。
- 动作词检查是保守的遗漏报警，不是新类别映射。它不能证明所有语义细节都完整；方向、力度、否定范围、非词表动作等仍需要模型语义检查。模型自检不是独立真值审核。
- 音频事件所有字段完整复制，重复发声次数与有效区间保留；视觉描述不会被转成新增音频。

## 代码入口

| 入口 | 职责 |
|---|---|
| [semantic_prompt.py](../src/listen_benchmark/semantic_prompt.py) | 原文候选、JSON约束、严格编译、遗漏报警、来源时间验证、输出 |
| [run_listener_semantic_prompt.py](../scripts/run_listener_semantic_prompt.py) | 30B推理、反馈重试、精确缓存恢复；失败不写通过标记 |
| [run_listener_semantic_prompt_cluster24.py](../scripts/run_listener_semantic_prompt_cluster24.py) | 三机分片、模型／代码／请求身份检查、保留worker来源的汇总；禁止跨协议旧响应复用 |
| [audit_listener_semantic_prompt.py](../scripts/audit_listener_semantic_prompt.py) | CPU重放、quote与描述相等、完整音频及review与实际提案绑定检查 |
| [build_listen_semantic_prompt_review.py](../scripts/build_listen_semantic_prompt_review.py) | 显式指定run/inputs/output；显示结果原文，动态统计，不再固定旧8B结果 |
| [listen_semantic_prompt_review.html](../scripts/listen_semantic_prompt_review.html) | 动态模型、数量、quote展示；不改写生成描述 |
| [核心回归](../tests/test_semantic_prompt.py)、[缓存回归](../tests/test_semantic_prompt_cache.py) | S1400漏点头、主语、可见尾部、重复okay、宽区间误合并、错误缓存和失败发布防护 |

已有output只允许同一身份续跑。模型、输入、代码或prompt变化时新建目录。不要使用旧`--reuse-from`把原响应伪装成新协议结果。旧版本重放请使用其冻结源码。

## 命令与验证范围

先运行CPU测试：

```bash
python -m pytest tests/test_semantic_prompt.py tests/test_semantic_prompt_cache.py -q
```

小样本推理（在已确认可用的卡上执行）：

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_PRELOAD=/private/software/conda/envs/vllm/lib/libstdc++.so.6 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
/private/software/conda/envs/vllm/bin/python scripts/run_listener_semantic_prompt.py \
  --model /private/ckpts/Qwen3-30B-A3B-Instruct-2507 \
  --inputs experiments/listen_semantic_plan_a_audit_20261027/inputs.jsonl \
  --output experiments/listen_semantic_plan_a_NEW --batch 4 --memory .48 --attempts 3
```

叠加运行时需检查实时剩余显存，`.48`是本次H200测试预算，不是适用于任意机器的保证。无需重启其他训练。

CPU验收：

```bash
python scripts/audit_listener_semantic_prompt.py \
  --inputs experiments/listen_semantic_plan_a_audit_20261027/inputs.jsonl \
  --run experiments/listen_semantic_plan_a_NEW
```

后续需要展示时再显式构建新页面，不会覆盖之前的复核页：

```bash
python scripts/build_listen_semantic_prompt_review.py \
  --run experiments/listen_semantic_plan_a_NEW \
  --inputs experiments/listen_semantic_plan_a_audit_20261027/inputs.jsonl \
  --output data_previews/listen_benchmark_review/semantic_plan_a_NEW
```

先做了8条重点样本验证，随后按用户要求扩展为三机24卡的100条原有样本验证（`inputs100.jsonl`、`cluster24_30b_plan_a_final/`）。24卡均与既有训练叠加，单卡新增预算`.46`，不是停止原训练。最终100条处理完成，60条通过、40条拒绝，24个worker无运行失败；耗时约8.2分钟。34项CPU回归通过，全部60条发布候选通过独立CPU重放。S1400提案已保留点头，但整段仍因时间关联冲突被拒，未发布训练标签。详见[结果、限制与S1400原文](../experiments/listen_semantic_plan_a_audit_20261027/README.md)。历史回归数据不代表当前静音后训练标签。没有重处理50h或全量数据，没有重建训练缓存。
