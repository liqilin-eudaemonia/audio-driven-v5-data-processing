# 完整pipeline：原生LTX全部attention与FFN使用LoRA

用户确定的范围：保留speaker Whisper→视频／音频两路attention全参数训练，原生LTX各类attention及FFN使用LoRA；保留AV＋整段Event＋双路Feature目标。这是独立版本，原完整模型、消融模型及正在debug的原第二阶段代码均不修改。

## 具体更新范围

| 模块（每个Transformer block） | 更新方式 |
|---|---|
| `attn1`、`audio_attn1` | Q/K/V/输出及存在时的attention gate使用LoRA |
| `attn2`、`audio_attn2` | 保留原Q/K/V/输出LoRA，并补充存在时的attention gate LoRA |
| `audio_to_video_attn`、`video_to_audio_attn` | Q/K/V/输出及存在时的attention gate使用LoRA |
| `ff`、`audio_ff` | `net.0.proj`、`net.2`使用LoRA |
| `qwen_attn2`、`audio_qwen_attn2`（实际输入speaker Whisper） | 全参数训练，不注入LoRA |
| 原生norm、AdaLN参数及block之外的输入／输出等投影 | 冻结 |
| Connector、Whisper adapter、Feature Decoder、Event头 | 沿用原全参数训练 |
| 原始Qwen、Whisper encoder、Gemma、teacher、VAE | 沿用冻结／缓存 |

沿用rank8、alpha8和LoRA LR1e-5。实际22B模型配置含48个block，启用了attention gate，所以每层6×5个attention投影＋2×2个FFN投影＝34个，**共1,632处LoRA**，其中原有384处，新增加1,248处；LTX LoRA共84,221,952个参数。两路Whisper attention共2,779,450,368个参数，仍全部训练。这些计数来自实际基座配置的meta构建，见[META_COVERAGE.json](../experiments/p2_ltx_all_attention_ffn_lora31_20261108/META_COVERAGE.json)。

## 与旧代码及数据的关系

采用在旧text LoRA之后补充注入的实现：原有adapter的顺序和权重键前缀保留；新增adapter的B初始化为0，并在独立RNG作用域创建。新增零增量时，与旧完整pipeline的前向逐值一致。renderer的可训练参数必须恰好等于LTX LoRA加两路Whisper分支，优化器同时核对所有活动模块且不重复登记。

新包：[src/p2_ltx_all_lora](../src/p2_ltx_all_lora)；入口：[scripts/p2_ltx_all_lora_train.py](../scripts/p2_ltx_all_lora_train.py)。训练、原生隐藏状态Event读出、完整保存／严格恢复和推理加载均已接入。新检查点使用独立schema，拒绝将旧text-only LoRA检查点作为本组严格resume；本组准备配置从预训练基座独立初始化，不自动继承旧Adam或启动训练。[代码验收](../experiments/p2_ltx_all_attention_ffn_lora31_20261108/CODE_ACCEPTED.json)；[源码／配置备份](../releases/p2_ltx_all_attention_ffn_lora_20261108/MANIFEST.json)，原备份保留。

数据仍为最终train＋dev的284,218段／500.001259h和最新完整caption；复用原始Qwen H、Gemma、AV VAE、speaker Whisper、listener teacher及guarded数值Event标签。损失仍为AV＋0.05整段Event＋0.025视频Feature＋0.025音频Feature，未知／冲突mask和有效类别门槛不改变。没有新增时序标签或重新打标。

准备了与既有当前消融匹配的31卡全轮样本／noise计划（9,169步）、all重计算、预取2批、512MiB FP64归约、每500步保存；node11 GPU7预留。全轮预算是准备配置，不是已获启动指令。

## 验证及启动

[准备目录](../experiments/p2_ltx_all_attention_ffn_lora31_20261108)包括：12项原生小型LTX测试、完整模型meta覆盖、32条真实全分支缓存回读、来源绑定和冷CLI GPU门槛检查。测试覆盖原前向逐值不变、all重计算、AV／Event对各类新增LoRA的梯度、Whisper保持全参数训练、冻结基座不更新、优化器完整性、原生gate投影，以及模型／Adam下一步严格恢复。

尚未执行这个扩展LoRA版本的完整规模GPU训练。正式启动时仍须用最长实际样本做两次真实更新以检查新增训练状态和峰值；本轮Qwen联合debug沿用此前text LoRA范围，不能将其106.46GiB峰值当作本新版本的实测。

```bash
PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
CFG=/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_ltx_all_attention_ffn_lora31_20261108/config.json
"$PY" scripts/p2_ltx_all_lora_train.py preflight --config "$CFG"
"$PY" scripts/p2_ltx_all_lora_train.py commands --config "$CFG" --stop-after 2
# 显式分配资源后才运行launch；该版本没有自动队列。
```
