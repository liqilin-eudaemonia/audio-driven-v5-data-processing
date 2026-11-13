# LTX全attention／FFN LoRA＋Whisper＋Feature（无Event）

独立消融版本，在既有`no_event` pipeline上仅扩展LTX LoRA范围；当前完整Event训练不修改、不停训，本版本未自动排队或启动GPU。

## 模型和损失

- 原生LTX视频／音频self-attention、条件attention、双向AV交叉attention的Q/K/V/输出和实际存在的attention gate，以及视频／音频FFN，均接入rank8／alpha8 LoRA，LR1e-5。
- 两路新增speaker Whisper cross-attention保持全参数训练；Connector、Whisper adapter、双路Feature decoder沿用全参数训练。
- 原始Qwen H读取缓存；Qwen、Gemma、Whisper encoder、AV VAE及teacher保持冻结。
- `L = L_AV + 0.025 L_video_feature + 0.025 L_audio_feature`，AV内部音频／视频各权重1。Event关闭，既无Event头参数，也不打开Event标签reader、不向batch加入events或hidden_event_layout。
- 从预训练基座独立初始化，不接续旧1500步去Event模型或当前完整模型。

## 数据和执行

最终284218段、500.001259h，train281060＋dev3158，test不进入训练；使用当前已确认的完整caption和对应五项缓存。统一入口为配置的`current_cache_index`及`sample_list`。`data_config`保留历史提取器参数路径，不代表使用历史成员；真实dataset／caption来自当前cache index，已逐条核对32条真实样本及来源合同。

沿用global batch31、duration496同一sample/noise计划、预取2、512MiB streaming FP64 SUM、GPU Adam、BF16、全重计算、每500步完整保存和9169步一轮预算。四机node12/node8/node4 GPU0–7，node11 GPU1–7，GPU0保留；单网卡沿用当前稳定配置。

启动器默认保留空卡保护。当前vLLM并存训练的108GiB node11约定属于活动训练的独立叠加入口；未来安排本组GPU验收时应沿用已授权的受控叠加方式并先做最长样本检查，不能把当前完整模型的显存测量当成本组实测。

## 验证与边界

16项原生tiny LTX CPU测试通过：完整注入／Whisper全参数／优化器参数归属、零增量与原noEvent前向一致、与当前扩展完整模型共有参数及AV/Feature输出逐值一致、AV梯度抵达全部attention/FFN与Whisper、Feature独立优化Connector及decoder、冻结基座不更新、Event读取保持关闭、完整模型/Adam下一步严格恢复与跨schema拒绝、数据/损失/scope漂移拒绝。

32条真实缓存全分支回读通过，涵盖train/dev，完整caption prompt与SHA绑定，teacher和Whisper齐全，Event类别数恒0。

实际22B safetensors配置在meta设备构建：48block、1632处LoRA（原384＋新增1248）、84221952个LoRA参数；Whisper全参数2779450368；Event参数0，优化器仅connector/ltx_lora/whisper/features四组。未创建CUDA上下文。

31rank命令、独立输出及stop-after传递通过；隐藏CUDA的冷train正确到达GPU门槛，未创建run.json。尚未执行本版本完整规模GPU最长样本及多机更新测试。

## 入口

- 代码：`src/p2_ltx_all_lora_noevent/`
- CLI：`scripts/p2_ltx_all_lora_noevent_train.py`
- 配置、preflight、META_COVERAGE、tests及命令：`experiments/p2_ltx_all_lora_whisper_feature31_20261108/`
- 准备和预检不启动GPU；只有显式`launch --logs ...`才会启动。

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python scripts/p2_ltx_all_lora_noevent_train.py preflight
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python scripts/p2_ltx_all_lora_noevent_train.py commands --stop-after 2
```
