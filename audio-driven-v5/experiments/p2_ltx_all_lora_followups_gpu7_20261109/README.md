**已适配GPU7外部服务占用（2026-11-09）：** 5500全部生成后，保留200条latent，原队列协调器已换成共享显存调度；实际使用`scripts/p2_all_lora_followups_gpu7_overlay_20261109.py`，未改原模型／采样助手及已准备的条件清单。另一GPT-OSS服务保持；当前两路解码、每路28GiB，未来7000／7500按空闲显存设进程预算，OOM等待资源。见`../p2_ltx_all_lora_eval_watch9h_20261109/resource_overlay_r3/`。

# 全LoRA完整pipeline后续Seamless评测队列：5500／7000／7500

**2026-11-10当前状态：** 5000、5500、7000各200条已生成和评分验收；按用户新指令，7500已改用node11 GPU4–7四卡，入口为`scripts/p2_all_lora7500_node11_gpu4_20261110.py`。原GPU7协调器已退出，已完成83条latent逐条摘要保持，四卡严格加载及新增产出已确认。[四卡接续与状态](../p2_ltx_all_lora7500_seamless200_sft2752_gpu7_20261109/README.md)。下文GPU7限制为早期历史授权，当前以此更新为准。

用户2026-11-09要求接在当前step5000评测后，继续只使用node11 GPU7。顺序：5000完整生成／解码／评分通过并退出控制器 → 5500 → 7000 → 7500；共新增600条输出。每轮全部评分通过才启动下一轮。不能用生成结束代替评分结束；失败或用户STOP不自动跳过。

5500完整权重来自`../p2_ltx_all_attention_ffn_lora31_20261108/step_005500.pt`（batch31，282源码）。7000／7500来自`../p2_ltx_all_lora_full28_batch28_20261109/step_007000.pt`、`step_007500.pt`（6428之后batch28续训，295源码）。仍为48层1632处原生attention／FFN rank8 LoRA，speaker Whisper两路全参数，AV＋整段Event＋双路Feature。不同阶段batch与样本展示数变化，不能视为严格等预算消融。

全部使用与5000／6000／6428一致的Seamless固定200、SFT2752规划/H/Gemma、speaker Whisper、listener首帧、seed137、Euler30、CFG3／7、512及原FPS／帧数。逐项检查规划器、参考清单、采样、Whisper身份及样本清单SHA相同。不重新规划或抽缓存，不读listener未来GT作为生成输入。

CPU提前校验完整checkpoint SHA、模型／Adam／31或28份RNG，逐张量导出相同模型；推理仅mmap读取导出。batch28来源严格核验原28卡计划与295文件摘要，按实际训练的runtime28.model规范化配置构建相同模型，再strict加载全部1632目标；保存的原始batch28身份不改。原训练源码及当前5000控制器不改。

- 总队列：`scripts/p2_all_lora_followup_queue_20261109.py`。
- 单步入口：`scripts/p2_all_lora_followups_gpu7_20261109.py prepare|queue|worker --step 5500|7000|7500`。
- 适配及导出：`src/p2_all_lora_followups_eval_20261109.py`。
- `PLAN.json`固定步骤、完整来源SHA及代码摘要；`status.json`当前准备／等待／评测阶段，`controller.log`及各`prepare_*.log`、`evaluate_*.log`。
- 每轮独立目录：`../p2_ltx_all_lora<step>_seamless200_sft2752_gpu7_20261109/`。媒体在`generated/ltx_all_lora_event_feature_whisper_<step>_sft2752/seamless/<id>/generated.mp4`、`listener.wav`；指标`RESULTS.json`。4个评分分片只是逻辑分片，物理始终仅GPU7。
- 继续沿用共享领片、最长优先、4路解码、2路评分及CPU预取／像素并行、GT缓存、完整200特征汇总FID/FVD；生成／解码清空PYTHONPATH，评分才使用vendor环境。
- `TESTS_ACCEPTED.json`：6项来源、规范化、GPU隔离及接续门槛检查通过；实际GPU完整模型加载在各自轮次启动时检查，不能将CPU通过当作GPU已运行。
- 当前5000评测继续；本队列CPU准备不占GPU。总目录创建`STOP`可取消后续并停止本队列当前轮次，只终止本队列自己的进程；旧5000评测不会因此被停止。不创建Goal或定时聊天汇报。
