# Qwen-Omni-3B 回应规划 LoRA SFT：代码与四机入口

2026-11-01。代码准备完成后，用户已授权在S1第2,300步保存后切换四机32卡训练，现已启动正式LoRA SFT。参考方案见 [VITA-1.5与冻结3B测试](P2_VITA15多阶段训练与自主回应规划试验_20261101.md)。

## 训练内容

输入是system指令、speaker音频、listener最终裁框参考首帧、回应规划指令。assistant目标为现有listener自然语言caption全文，不输入listener音轨、后续视频帧、speaker转写或事件标签。训练采用teacher forcing：目标描述作为assistant历史token参与因果计算，以前文预测下一个token；自由生成时完全不提供目标描述。

监督范围为描述token和一个原生`<|im_end|>`。system、user、多模态输入、assistant起始前缀和padding的标签均为`-100`。损失是回答token交叉熵，无AV／Event损失。不能把这次训练与此前由AV／Event反传更新Qwen hidden的S2混为一谈。

LoRA覆盖Thinker语言模型全部36层的q/k/v/o投影与gate/up/down投影，rank=32、alpha=64、dropout=0。实际可训练参数59,867,136个；视觉编码器、音频编码器、原语言模型参数及lm_head冻结。使用标准PEFT adapter保存；不自动转换或加载到原有AV训练的自定义Qwen LoRA接口。

现有caption原文保留，包括动作顺序、自然语言时间描述和部分人物外观信息，不自动改写成另一套标注。新user prompt没有上轮“必须简短一段／禁止时间描述”的限制，以免与训练目标冲突；仍要求说明声音状态、保持listener角色、不复述speaker、不输出推理或JSON。完整prompt在 [data.py](../src/p2_planning_sft/data.py)。本轮代码只支持该prompt，不更改此前24条冻结3B基线。

## 数据与缓存

- 正式train：`experiments/p2_j_train500h_cache_20261101/`，全部285,638条，约500h。没有按19类事件标签筛样本。
- 源数据通过冻结manifest及逐样本摘要核验。参考PNG取已验收的最终裁框首帧，并对照`media_audit`的RGB哈希；speaker取配对音轨的原时间窗，并核对原始float32哈希。
- 音频按原训练规则重采样到16k，最多去掉重采样多出的一个端点。speaker不套用listener静音门控。
- 不需要重提Gemma、AV latent或原Qwen hidden缓存。SFT在线执行Qwen语言层；复用的是已验收的首帧和音频来源，不把冻结hidden作为可微语言层输入。
- 每个DataLoader worker持有独立processor；音频padding按本批最长片段加余量，避免默认300秒补零。超出4096-token上下文直接报错，禁止默默截断描述。
- 当前dev loss只使用原24条中的4条dev，作为流程诊断；这些数据不进入训练。它不是代表性验证集，不能据此宣布规划或AV质量提升。后续正式模型筛选仍需更大独立验证集和实际AV评估。

## 默认训练配置

[训练配置](../configs/p2_planning_sft_3b_32gpu.json)与[四机配置](../configs/p2_planning_cluster_32gpu.json)。

| 项目 | 默认值 |
| --- | --- |
| 原始模型 | `/public/2_CKPT/3_PreTRAINED/Qwen2.5-Omni-3B` |
| 机器 | node17 / node8 / node12 / node4，每机8卡 |
| 预算 | 完整train 1轮 |
| batch | 每卡8，梯度累积1，32卡global batch=256 |
| 更新步数 | 约1,116步；分布式尾批可能有少量补齐重复，不删除原样本 |
| 学习率 | LoRA 1e-4，3% warmup，cosine |
| 优化器 | fused AdamW，weight decay 0.01，梯度裁剪1.0 |
| 精度与注意力 | BF16、SDPA、TF32 |
| 内存 | 正式运行关闭gradient checkpointing以省去重算；训练／loss评估关闭KV cache |
| 供数 | 每卡4个持久worker，pin memory，按时长代理长度分组 |
| checkpoint | 每100个optimizer step，保留最近3份；完成后另存final_adapter |
| 记录 | 每步TensorBoard loss／LR／grad norm；每100步dev loss |

损失按所有rank、当前梯度累积窗口中的有效回答token数归一化。transformers 4.57.6的原生Omni loss没有使用`num_items_in_batch`，因此这里用Trainer的`compute_loss_func`显式计算sum/global-token-count，保留其DDP世界大小补偿。先筛出监督位置，再转FP32计算交叉熵，减少无效前缀位置的FP32显存。

这些是已落地的首轮配置，32卡正式训练已实测，初始第71—100步平均端到端0.6923秒／步，约369.8条／秒；不含随后checkpoint和评估开销，不宣称该batch已最优。完整预算保留在配置文件中；冒烟测试输出独立保存，不能用作正式模型。

## 启动和恢复

从项目根目录执行，共享Python为`/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python`。

仅查看四机命令，不启动：

```bash
python scripts/p2_planning_cluster.py
```

纯CPU、四机32进程通信探针：

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python scripts/p2_planning_cluster.py --probe --launch
```

后续用户确定正式开跑时执行：

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python scripts/p2_planning_cluster.py --launch
```

该命令保持前台监督四个节点，可放入tmux。启动前先读取四机显存状态，任一卡不足80GiB空闲便退出；不会终止当前S1或其他GPU任务。沿用已验证的网卡／NCCL配置，新master端口29771与S1分开。四机均使用22端口，node8地址为`172.29.104.233`。

恢复同一正式运行：

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python scripts/p2_planning_cluster.py --launch \
  --resume /private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_planning_sft_3b_32gpu_20261101/checkpoint-100
```

checkpoint含adapter、optimizer、scheduler、Trainer状态及各rank RNG。恢复核对数据、配置、代码与world size，要求同一输出目录；不支持把smoke检查点当正式训练起点。正式输出目录为`experiments/p2_planning_sft_3b_32gpu_20261101/`。

已保存adapter的24条自由生成入口：

```bash
CUDA_VISIBLE_DEVICES=7 /private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python scripts/p2_planning_generate.py \
  --config configs/p2_planning_sft_3b_32gpu.json \
  --adapter experiments/p2_planning_sft_3b_32gpu_20261101/final_adapter \
  --output experiments/p2_planning_sft_3b_32gpu_20261101/generated24.json
```

该入口使用本轮SFT的同一prompt；省略`--adapter`可测原始3B作同prompt对照。默认24条合批、greedy、512新token。只调用`source_inputs`，不附加目标描述或labels，结果保留token及EOS状态。

## 验证记录

证据目录：[p2_planning_sft_preflight_20261101](../experiments/p2_planning_sft_preflight_20261101/)。

1. 7项单元检查通过：assistant-only mask、EOS、padding、拒绝超长截断、因果移位、忽略位置梯度、不同rank回答长度下的全局token归一化、source消息结构。
2. 全24条输入RGB和16k speaker波形与上轮冻结3B试验逐值一致；标签解码与原caption全文一致，额外仅一个EOS；完整285,638条train保留，dev没有泄漏。
3. 指定推理机物理GPU7真实反传通过，冻结参数无梯度，LoRA梯度有限且非零。小批量按20 train训练、4 dev评估，未把dev当训练数据。
4. 四机32进程Gloo all-reduce及rank/hostname收集通过。此项是CPU通信检查，不等同于32卡NCCL训练实测。
5. 已从第1步checkpoint恢复到第3步，完成20条训练样本的1轮；峰值reserved显存18.625GiB。与同数据连续执行相比，optimizer参数组和scheduler一致，504个adapter张量最大绝对差为0.00013587；optimizer动量张量也有数值差异。当前BF16／SDPA未启用严格确定性，**不宣称断点恢复逐位一致**。记录见`resume_equivalence.json`。4条dev loss从1.8296降至1.5773，仅说明小批量训练流程可运行，不代表正式质量提升。

6. 保存的`final_adapter`已重新加载，24条source-only自由生成均正常EOS结束，见`reloaded_generated24.json`。这是3步冒烟训练产物的接口检查，不作为正式训练效果对比。
7. 实际TensorBoard文件中已确认`train/loss`、`train/learning_rate`、`train/grad_norm`、`eval/loss`曲线。准备阶段GPU验证在指定推理机GPU7完成；后续32卡正式启动与吞吐记录见下一节。

前一次CPU探针因node8缺少主机记录失败；node8经本机IP核对后，使用本机`/etc/ssh`公钥显式登记，严格SSH主机校验未关闭。第二次探针成功。代码没有加入自动接受未知主机密钥的设置。


## S1切换与32卡正式运行（2026-11-01 10:52 UTC）

用户授权：等待S1第2,300步保存后停止，用四机32卡训练Omni LoRA。S1的`step_002300.pt`已完成SHA256验证；检测到保存完成后立即写入原生STOP标记。训练循环当时已经进入下一步，因此正常退出时额外保存`step_002301.pt`，最终暂停在2,301／8,927。两份权重及完整状态均保留，没有强杀训练进程。

正式SFT使用本节所列配置：全部285,638条train，1轮、约1,116步，global batch256、LoRA LR峰值1e-4。teacher forcing的next-token prediction保持不变：预测描述第t个token时使用真实前t−1个token；因果注意力不可访问未来目标，loss仅覆盖描述与EOS。

新增实际优化：在独立推理卡上测试关闭梯度重算，20条train／4条dev的3步检查通过，峰值reserved显存48.85GiB，最终dev loss与先前连续重算测试相同。正式运行使用`gradient_checkpointing=false`，避免在充足显存下重复前向计算；这不是增大显存本身带来的收益承诺。其他已启用配置为BF16、SDPA、fused AdamW、时长分组、每卡4个持久数据worker、pin memory、短音频动态padding、回答位置筛选后计算FP32交叉熵、训练与loss评估关闭KV cache，以及原四机NCCL网络配置。

正式运行的32-rank LoRA梯度检查通过，冻结参数无梯度。第71—100步实测平均0.6923秒／步、369.8样本／秒，属于不含保存／评估的初始稳定窗口。正式目录为`experiments/p2_planning_sft_3b_32gpu_20261101/`，逐步耗时写`step_metrics.jsonl`；四机日志在其`launch_logs/`下。

切换证据与后台启动器PID：[switch_from_s1](../experiments/p2_planning_sft_3b_32gpu_20261101/switch_from_s1/)。原S1保留STOP标记，恢复时需要显式移除该标记并从保存权重接续，本次不自动恢复S1。

首个正式`checkpoint-100`已核验：含adapter、optimizer、scheduler、Trainer第100步状态及全部32份rank RNG；TensorBoard已记录train loss／LR／grad norm及第100步dev loss。检查时训练已继续推进超过第175步，四节点日志无异常。


## 用户追加第2轮（2026-11-01）

第一轮已于北京时间19:03:59完成1,116步，实际训练773.56秒。用户随后要求再训1轮，因此新增独立配置`configs/p2_planning_sft_3b_32gpu_epoch2.json`，输出到`experiments/p2_planning_sft_3b_32gpu_epoch2_20261101/`。原第一轮目录及final_adapter保持不变。

第二轮从第一轮最终`checkpoint-1116`恢复LoRA、AdamW动量、Trainer进度及32卡RNG；总预算由1轮扩展为2轮，累计2,232步，只新增第1,117—2,232步。完整285,638条train、global batch256、teacher forcing回答token交叉熵及全部提速设置保持不变。每100个累计optimizer step保存，第二轮首个周期检查点为第1,200步。

第一轮cosine已在终点降为零。续训保持原34步warmup计数，将cosine总长度延长至2,232步，并在首个新增optimizer更新前重新计算当前LR，约5.12e-5，随后下降至零。不清空AdamW状态、不重复第一轮，也不从零重新初始化LoRA。新增`continuation.py`只允许在独立目录从已完成的最终检查点追加1轮，数据、prompt、模型、LoRA和其他训练参数必须一致。

9项测试通过，其中新增检查覆盖：终点零LR重新对齐、AdamW动量和步数保持、禁止借续训改动数据／学习率等配置。正式加载审计写`extension_load.json`，用于确认所有504份优化器参数状态的step均恢复到1,116；后台启动器记录为新目录的`launcher.json`。

实际四机恢复已验证：504份AdamW状态全部恢复到step1,116，起始LR为5.1214782e-5；首个新增训练步为1,117，四节点已持续产生epoch>1的损失日志，LoRA梯度有限且非零。验证快照见第二轮目录的`continuation_verified.json`。


## 第2轮结束后自动恢复AV＋Event，改为200步保存

用户授权在当前LoRA第2轮完成后，恢复之前暂停的“双通路文本指导＋AV＋Event”S1，并每200步保存权重。后台队列已部署，控制目录为`experiments/p2_j_av_event_resume_after_lora2_20261101/`，控制器PID记录在`controller.json`，实时状态在`state.json`。准备验收时状态为`waiting_lora`，尚未切换。

触发条件是LoRA第二轮`COMPLETE.json`确认成功、累计2,232步及epoch=2，最终adapter已保存，随后等待四机32卡释放显存。条件满足后恢复S1第2,301步，继续原预算至第8,927步；使用原GT自然语言指导缓存、19类整段Event监督及512窗口时长分组。该接续恢复原S1，不把刚训练的规划LoRA替换进S1的条件抽取器。

检查发现原训练循环写死`step % 100`，配置校验也固定要求100，因此本次同时修改为读取正整数`training.save_every`，接续配置设为200。下一个周期检查点为2,400步，随后2,600、2,800等；结束或正常停止仍保存最终一步。TensorBoard继续逐步记录。当前仍在运行的LoRA第二轮保存间隔保持原设置。

为保留原S1检查点的身份和记录，恢复副本放在`experiments/p2_j_guided_av_event_s1_32gpu_resume200_20261101/`。迁移仅更新输出位置、保存间隔及这两处源码指纹；模型、optimizer和32份rank RNG已逐位比较一致，原时长采样计划和数据身份也已核验。旧目录与旧STOP标记保留，新目录是后续AV训练的正式入口。

46项matrix／clip-event测试通过。完整迁移证据在新目录的`migration.json`，配置为控制目录的`av_resume200.json`，队列任务为`job.json`；实现入口是`scripts/p2_resume_av_after_lora.py`。队列不会在LoRA未完成时抢占GPU，不终止其他GPU任务；若等待条件或启动检查失败，会写`state.json`中的failed状态及异常信息。
