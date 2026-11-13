# 三组全LoRA消融：Qwen7500 × 固定64条 × 三机24卡

2026-11-13用户明确授权启动12组快速测评；GLARE对比权重和256条晋级口径稍后确认。本目录保留全部12组结果，不按预期排名隐藏结果，不自动启动256条。

| Pipeline | LTX step |
|---|---|
| 去Event | 5500、6000、6500、6573 |
| 去Feature | 5500、6000、6500、6643 |
| 去Whisper | 5500、6000、6500、6643 |

统一Qwen7500，原固定test64，seed137，原30步／CFG3和7，512px／25fps。共768条生成及评分。此为test子集上的权重筛选，不能作为独立验证集结果。三组训练预算不同于完整6800，比较时保留该差异。

## 2026-11-13动态调度修复

用户指出空卡后，实查旧控制器每份权重固定最多2个生成worker，导致剩余3个慢组无法使用已释放GPU；node8同时出现额外占用，部分卡显存接近满载。

已切换至 `accelerate.py`：按尚未生成完的权重数量动态分配，每组上限为 `max(2, floor(20 / 剩余组数))`，例如3组各最多6个、2组各最多10个、1组最多20个。仍以剩余样本数、文件锁、GPU剩余显存准入，优先node12／node11；解码和评分继续重叠。其他任务不动。

切换前控制器和其持有的worker全部退出，1315份已完成媒体／潜变量／评分收据SHA保持不变；未提交的计算可以重跑。权重、输入、seed、采样、原生worker代码及评分公式均未修改，原 `PREPARED.json`／`PLAN.json` 哈希不变，GLARE3935的后续依赖保持有效。记录见 `DYNAMIC_MIGRATION_BEFORE.json`、`DYNAMIC_MIGRATION_RESUMED.json`，日志为 `dynamic_controller.log`。

此后恢复应运行：`/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -u experiments/p2_ablation12_test64_qwen7500_24gpu_20261113/accelerate.py`。下方原启动命令为初始版本；不要重新运行旧的固定每组2worker控制器。

## 复用与加速

- 直接复用已验收64条Qwen7500规划、该adapter的hidden states和Gemma文本条件；不重算、不改上游缓存。
- 逐组使用自己的原生 `p2_ltx_all_lora_noevent` / `p2_ltx_all_lora_components` 加载器，严格加载整个state dict及1632处LoRA。batch28训练元数据按原训练runtime规范化，模型结构／采样计算不改。
- 去Whisper既不读speaker Whisper缓存，也不传入Whisper条件；其他分组沿用原生渲染函数。Event/Feature是训练辅助分支，推理不读取GT标签或teacher。
- 单次CPU顺序校验原权重SHA，每份通过即准入；mmap读取，不另存数百GB的推理权重。
- 每组最多2个生成worker，按样本文件锁领取任务；解码、GPU检脸/情绪评分并行交错；解码器进程内复用；评分batch64、像素线程4、GT特征缓存复用。
- node12/node8/node11 GPU0–7；生成显存限96GiB，解码38GiB，预留6GiB准入。node11现有其他任务保留，剩余显存不足生成时可接解码／评分；24是授权调度上限，不承诺每时刻24卡满载。

## 状态、结果与恢复

- `status.json`：实际生成数、潜变量数、完成组、活跃worker和日志。
- `model_acceptance/`：每张生成卡实际载入的权重及组件存在性检查。
- `RESULTS.md` / `RESULTS.json`：各组全部指标及有效覆盖；声音指标适用于本方法。GLARE不生成音频，后续不能用音频N/A当零分排序。
- `COMPLETE.json`：12组全完成的最终收据；未出现即未全部完成。
- `PREPARED.json`：冻结源代码、协议与清单；`CHECKPOINTS.json`：原权重路径、step、SHA及文件stat。
- `INTEGRITY_COMPLETE.json`：12份权重完整SHA均通过；`INTEGRITY_FAILED.json`则停止队列。
- `controller.log`、`integrity.log`与`logs/`记录运行异常；失败最多重试3次（SSH255最多6次），不能伪造结果或将失败记零分。

启动／恢复（先确认旧控制器已退出；不要并发启动两份）：

```bash
PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
W=experiments/p2_ablation12_test64_qwen7500_24gpu_20261113
$PY "$W/run.py" prepare
nohup "$PY" -u "$W/run.py" integrity >> "$W/integrity.log" 2>&1 &
nohup "$PY" -u "$W/run.py" activate >> "$W/controller.log" 2>&1 &
```

`STOP`文件使队列停止，仅关闭本次guard持有的worker；恢复前明确移走本次STOP并核验失败原因。已提交媒体和评分分片自动跳过，进程死亡后文件锁释放，未提交样本可重跑。FID/FVD使用完整64条特征合并，不平均分片分数。
