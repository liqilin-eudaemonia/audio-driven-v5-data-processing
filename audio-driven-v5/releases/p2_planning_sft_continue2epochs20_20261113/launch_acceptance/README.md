# Qwen规划LoRA从SFT2752追加两轮（三机20卡）

**已实际续训验收（UTC 2026-11-13）：** 最新核验step2811；20个rank的adapter初值、Adam step2752、scheduler、RNG及首梯度均通过，连续59次更新的loss／梯度／LR均为有限值。下一保存点3000，累计目标8256；运行继续，非完成标记。[实际启动验收](LAUNCH_ACCEPTED.json)。

用户2026-11-13要求优先接续Qwen LoRA；5500／6000／6500／7000的Seamless全集评测后续再定，本任务不启动该评测。

- 起点：`../p2_planning_sft_upright_3b_32gpu_resume_20261106/training/checkpoint-2752`，完整adapter／Adam／scheduler及原32份RNG保留。
- 模型：原Qwen2.5-Omni-3B；沿用Thinker语言层attention及MLP LoRA rank32、alpha64、dropout0；视觉和音频感知塔冻结，speaker音频＋listener第一帧输入，当前caption为规划目标。
- 数据：沿用最终upright筛选后的704,368条train（1191.9879小时），train清单SHA `2ba36a307613942c905c148313c59ce29658d33975b05fcf4f941d425a41163d`。这份规划SFT全量数据与生成模型的500h子集用途不同；不把dev/test混入梯度训练、不重筛标签。
- 追加两个完整epoch，累计epoch3，2752→8256，新增5504次更新。有效batch256、长度分组256；20卡每个microbatch128条、各6或7条，梯度累积2次形成全局256；末尾112条各5或6条，不补样本、不丢样本。
- bf16／TF32、SDPA、fused AdamW、无gradient checkpointing、4路worker／prefetch4、验证过的PNG复用与隔离解码、父流程媒体输入加速保持；通信使用同一11.1.3子网对应的单HCA RDMA。
- 原训练末端LR为0；保留Adam及scheduler的step2752，将cosine总长度延长至8256、原warmup83保持，恢复首个LR约7.59153e-5，避免零学习率空跑。该调度变更已明确记录，不声称与从头三轮的数值轨迹相同。
- 三机20卡：node12／node8各GPU0–7，node11仅0–3；node11 GPU4–7保留。原四机28卡尝试的模型／Adam／RNG恢复均通过，但node4约209万系统进程、创建线程失败导致首批DataLoader退出；未产生任何更新。旧尝试保留于`../p2_planning_sft_continue2epochs28_20261113`，不清理其他任务，先在其余三机推进。主节点及NCCL／GLOO控制通信采用11.1.3子网、各机对应单HCA；八HCA启动曾在大模型同步阶段停滞，不重复空等。
- 原32-rank RNG保留在起点；新任务明确映射原rank0–19的Python／NumPy／CPU状态，CUDA状态按本机可见卡数保留。拓扑变化可改变浮点轨迹，不宣称逐位复现。
- 每500个累计step原生完整保存（下一次3000），保留最近3个；8256最终完整保存并导出`training/final_adapter`后退出。TensorBoard在`training/tensorboard`。

## 状态与产物

- [恢复和数据CPU检查](PREPARED.json)、[检查点／RNG映射](CHECKPOINT_MAPPING.json)、[资源实查](GPU_PREFLIGHT.json)。
- `training/step_metrics.jsonl`为实际完成更新；`logs/rank-00.log`包含loss／LR。
- `training/communication`、`optimizer_restore`、`rng_restore`、`gradient_audits`逐rank记录实际通信、恢复及首个梯度；进程启动本身不代表已经更新。
- `STATUS.json`／`node*_process.json`为自身进程状态，`ERROR.json`或`node*_ERROR.json`表示失败，错误不会伪报完成。
- 新权重：`training/checkpoint-<step>/adapter_model.safetensors`；结束后`training/final_adapter/adapter_model.safetensors`。

## 启动及恢复

首次启动已执行：

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -u scripts/run_qwen_sft_continue2epochs20_20261113.py
```

故障时先确认本任务所有rank退出、归档错误和旧node状态、检查最新完整原生checkpoint并移除本任务STOP，再用同一入口`--resume <本任务training/checkpoint-N>`。不重新运行数据准备，不覆盖父检查点、不停止其他任务。未保存更新需要重算。

所有实现改动限制在本任务独立vendor及启动入口；父SFT源码、旧adapter及LTX训练／评测代码均保持。准备脚本用于首次CPU准备，不用于运行中重写RNG。
