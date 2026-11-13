# Qwen 自主规划 → 二次 forward → Gemma/LTX 推理评测

2026-10-28：按用户确认的链路补充代码。**只完成代码、CPU验证和任务准备，未加载GPU模型、未启动自主规划或生成。** 现有HTML仍是目标标注指导结果，不是本实验结果。

## 本次实现与对照边界

输入仅有 speaker 音频、listener 首帧、片段时长及固定任务指令。冻结的原生 Qwen2.5-Omni 先根据这些输入生成带时间段的 listener 视觉/声音回应规划。代码校验后，把规划按既有全英文指导模板渲染；同一文本用于：

1. 拼到原有 listening 多模态输入后，执行第二次 Qwen forward。
2. 提取原配置的第8、18、28、35层完整 hidden，包含输入和规划全部有效位置，送入原 Connector。
3. Gemma 编码完全相同的文本；与 Connector 的 U_video/U_audio 共同指导 LTX 生成。

第二次forward保留训练时的原 listening 输入模板，追加的只是新规划。没有复用第一次自回归生成的KV或中间hidden。Connector读取全序列，不只读取计划位置。模型与原生生成器的checkpoint、source、assets、LoRA目标层、采样设置检查保留。

本次默认固定原生Qwen、不装规划LoRA：每条样本规划一次，跨权重和生成seed复用。planner使用greedy，seed=20261028；AV生成仍用137/251。第一轮最多2048个新token，校验失败最多3次；修正请求仅提供同一输入及校验错误。原回答、生成token、修正请求与最终文本全部留档。失败样本不回退真实标注，不截断文本，不补写计划、不静默丢样本。

这是对整段speaker音频的离线规划，不是流式/因果在线回应评测。

### 为什么采用这一条链路

用户已明确要求“自主规划 → 同一规划进入第二次Qwen forward和Gemma”。仅替换Gemma可以更单纯地比较文本影响，但没有实现用户要求的规划hidden条件；直接采用第一次生成时的hidden会偏离已训练的第二次forward编码方式。本次因此实现上述完整双路链路，真实标注指导结果单独保留。

**与旧四组的标注指导对比有两项变化：** 旧四组此前只替换Gemma，Qwen仍为固定输入hidden；本次旧四组也把自主规划加入Qwen。因此这两次旧四组之间不能把差异仅归因于规划质量。新版guided权重两次都采用“计划进Qwen＋Gemma”，更适合比较真实指导与自主规划。另有训练数据差异时，也不能只归因于prompt或某个loss。

## 规划格式与安全边界

Qwen输出JSON，仅允许`events`，每条事件仅允许`modality/start/end/description/text`。时间必须满足`0 ≤ start < end ≤ 输入时长`；visual需描述、无发声文本；audio可描述笑声、简短应答、明确静音，文字是**模型拟定的listener词语**，不是speaker ASR。

所有描述、词语要求英文，沿用已有英语脚本检查；不新增自动翻译。校验也拒绝重复JSON键、非有限时间、未知字段、模型控制token和未正常EOS的截断回答。允许空事件列表，不强迫回应，也不把空列表解释成来自GT的静音标签。English脚本检查不等于完整语言识别，实际输出仍需GPU S0抽查。

最终Gemma格式沿用原 `GEMMA + INTRO + [start–end seconds] + Visual response / Audio response`，复用原音频描述与vocal content合并规则。`source_record`的兼容schema仅供旧formatter使用，不代表这些内容来自真实标注；外层始终明确标记 `plan_source=qwen_autonomous`。

准备接口没有labels/targets参数；observation拒绝这些字段。规划、缓存和生成不读取GT媒体或标注；只有生成结束后的独立指标命令可以读取GT来计分。未复用oracle指导缓存或旧固定Qwen hidden。首帧VAE仍可复用原模型实现。

## 代码与已准备范围

- [CLI](../scripts/p2_autonomous_eval.py)：`prepare / plan / cache / verify / generate / worker / metrics`。
- [规划与校验](../src/p2_autonomous_eval/planning.py)、[观察输入与冻结协议](../src/p2_autonomous_eval/protocol.py)。
- [缓存及读取](../src/p2_autonomous_eval/runtime.py)：独立`plan`、Gemma `text`、Qwen `input_plan`；同文本校验、文件校验、模型/代码/参数身份及共享锁。
- [两种原生权重推理](../src/p2_autonomous_eval/generation.py)：旧4组historical source恢复，新版guided Stage1完整strict load；失败尝试保留在独立目录，成功原子发布，可续跑复用。
- [指标适配](../src/p2_autonomous_eval/evaluation.py)：复用本批已固定的OmniResponse/GLARE实现；输入输出独立标记，不覆盖现有结果。
- [测试](../tests/test_p2_autonomous_eval.py)、[准备好的实验目录](../experiments/p2_autonomous_test24_20261028/README.md)。

使用原test24，speaker/首帧/GT/生成seed/30步采样/video CFG3/audio CFG7均与当前评测一致。任务清单包含旧4组最终1268步、新版AV与AV＋Event最终1249步，共6组288输出；为新版AV＋Feature最终1249步另留48项，共336项。缺少最终权重的任务会记录waiting，不改用step200或中间checkpoint。准备任务不等于已生成结果。

本入口明确支持当前旧4组与新版冻结Qwen的guided Stage1。**不声称支持S2 Qwen LoRA权重**：S2需加载各组LoRA在线执行第二次forward，不能继续共用本次冻结hidden缓存；传入S2会拒绝。

## 执行方式（尚未启动）

工作目录与已有oracle结果分开。以下命令在GPU测试获安排后执行，不应直接与当前24卡训练叠加。

```bash
PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
ROOT=/private/liqi_workspace/project_code/audio-driven-v5
WORK=$ROOT/experiments/p2_autonomous_test24_20261028

# 已完成：纯CPU，锁定原test24和7组任务，不调用模型。
$PY $ROOT/scripts/p2_autonomous_eval.py prepare \
  --config $ROOT/experiments/p2_train50h_postprocessed_reselected_20261027/config.json \
  --observations $ROOT/experiments/p2_reintegrated_final_eval_20261028/test24/observations.json \
  --jobs $ROOT/experiments/p2_reintegrated_final_eval_20261028/jobs.test24.json \
  --work $WORK

# 单卡样本分片S0：只取排序后的第0个样本，执行规划、两份缓存、校验与可用权重生成。
CUDA_VISIBLE_DEVICES=0 $PY $ROOT/scripts/p2_autonomous_eval.py worker \
  --work $WORK --shards 24 --shard-index 0
```

24卡运行时，每机8个独立worker：node17用shard 0–7，node12用8–15，node4用16–23；各进程仅暴露对应单张GPU。样本数24时每个worker负责同一条样本的规划与各组双seed生成。每个生成job启动独立子进程，隔离旧/新版Python包与历史源码恢复。模型按阶段卸载；现有缓存通过身份检查后复用。worker每个GPU阶段前要求至少120GiB空闲，避免误叠加正在训练的卡；缺权重/失败会报告并返回非零，不能标为全完成。

也可以分开执行三个缓存阶段，均支持相同的`--shards / --shard-index`：

```bash
$PY $ROOT/scripts/p2_autonomous_eval.py plan --work $WORK
$PY $ROOT/scripts/p2_autonomous_eval.py cache --work $WORK --part text
$PY $ROOT/scripts/p2_autonomous_eval.py cache --work $WORK --part input_plan
$PY $ROOT/scripts/p2_autonomous_eval.py verify --work $WORK
# 单项生成使用 $WORK/jobs/ 下的已冻结job JSON：
$PY $ROOT/scripts/p2_autonomous_eval.py generate --job "$WORK/jobs/<job_hash>.json"
```

## 计分与报告

各组48输出齐全后，单独执行指标。例如：

```bash
CUDA_VISIBLE_DEVICES=0 $PY $ROOT/scripts/p2_autonomous_eval.py metrics \
  --work $WORK --arm autonomous_guided_av_step1249
```

先验证所有输出、规划和协议，再构造原生pairs；GT沿用上一批同一test24的视频与reference_audio。复用FVD-32、UTMOSv2、LSE-D/LSE-C、GLARE R-F1/R-tIoU/R-ATD/R-FID；GLARE严格和去额外门槛两版均保留。指标模块会读取GT用于计分，但完成记录明确区分`generation_target_access=false`和`gt_media_access_for_scoring=true`。

每组结果与覆盖率位于`metrics/<arm>/test24/`。组间比较仍需取各指标的共同支持，静音/不适用保留N/A；严格GLARE不同子集不能直接排名。不把“规划与标注相似”单独作为成功判据，还要看实际生成的声音、动作和与speaker的关系。自主规划可能存在多种合理答案，与单一GT的偏差不全是回应失败。

当前先交付可执行链路与原生评测结果格式，未把尚未生成的结果填进HTML，也未宣称取得自主规划验证结论。

## 验证状态与首轮GPU验收

CPU新增测试29项，加原缓存回归20项，共49项通过：合法/非法规划、EOS、时间范围、英文脚本、输入隔离、同一规划两路条件、native负向CFG保持、缓存污染/串用拒绝、恢复、元数据、原子发布及指标前置检查。旧4组真实run配置在各自原生包下完成CPU兼容与历史source恢复检查；新版配置也核对原生编码与采样配置。训练冻结源码117项未改变。

尚需在空出的GPU上执行S0：检查真实模型规划内容、时间合理性、两次forward与LTX全链路、非静音计划对应的音轨，以及一条样本的指标冒烟。S0通过再安排全量24条；未做GPU实测不能称作端到端已通过。本次没有停止、修改或叠加当前训练。
