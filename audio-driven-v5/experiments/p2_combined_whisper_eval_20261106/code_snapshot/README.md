**已完成（2026-11-06 02:23 UTC，北京时间10:23）：** 6000／8000步×Seamless200／ResponseNet200，共800条生成、导出、62个评分分片及全局FID/FVD汇总完成，最终逐条验收通过。Seamless两组各1条情绪／运动支持不足（199/200），其余图像指标200/200；音频执行错误0，UTMOS因短回应为N/A。训练保持停在8000步，巡检goal保持暂停。 [结果表](RESULTS.md) · [CSV](summary.csv) · [逐条验收](FINAL_AUDIT.json)。

# Event＋Feature＋Whisper：8000步停训后自动评测

用户于2026-11-06要求把评测接在第8000步完整保存之后。本接续按上下文理解为自动接续**测评**；不恢复后续训练，不恢复已暂停的巡检goal。

## 范围与比较

- 当前31卡联合训练的6000、8000步，各测固定Seamless200和ResponseNet分层随机200，共800条新生成。
- 直接沿用历史两份references清单、原生Qwen规划／H／Gemma、listener首帧、speaker Whisper缓存。GT listener未来画面、音轨及标注描述不作为输入。
- 采样沿用seed137、30步Euler、video CFG3、audio CFG7、512×512、各pair原FPS及原时长。CFG两路保留U与Whisper。
- 训练global batch31与历史实验不同，6000同step并不意味着完全相同样本曝光；这一差异写入比较记录。
- 历史12组结果保持原样；新结果写`RESULTS.json`，`COMPARISON.json`引用历史固定200条结果与本次结果，不重新生成历史组。

## 自动接续条件

入口：[p2_combined_whisper_eval.py](../../scripts/p2_combined_whisper_eval.py)。后台控制器见`CONTROLLER.json`和`controller.log`。

1. 等待原训练`STOP_AT_8000_COMPLETE.json`，并确认`status.json`确为8000步。原停训脚本仍运行，训练正常完成该步保存后退出。
2. 严格检查完整6000／8000权重的SHA、所有模型张量和训练来源身份；导出逐值相同的模型推理副本，省去每个GPU读取Adam状态。
3. 连续两次确认授权31卡释放，跳过node8 GPU7。SSH失败视为观察失败，不当作空闲。
4. 先做两个数据集短／长片段×两个权重共8条真实GPU生成与解码检查，核对权重、帧数及无GT未来输入；这些产物直接复用于正式800条。
5. 31卡共享领取其余样本，模型及VAE常驻、兼容权重切换、长片段优先。每样本flock及原子提交，失败后保留已完成结果重试。
6. 原生解码导出后，自动计算完整图像／情绪／运动／音频指标；GPU检测与表情、batch64、4线程像素指标、每卡最多4评分worker。复用无损GT RGB缓存及两套GT特征缓存。FID/FVD从每套200条的全局特征重算，保留有效覆盖及N/A。

阶段失败会保存`FAILURE.json`与明确状态，整体最多3次接续尝试，已有正式结果不重复计算。`HANDOFF_COMPLETE.json`只在评测结果生成后写入。

## 资源

node17 `172.29.15.91` GPU0–7；node12 `172.29.252.37` GPU0–7；node4 `172.29.225.51` GPU0–7；node8 `172.29.104.237` GPU0–6。**node8 GPU7继续留给用户。** 不终止其他任务。

## 验证与当前状态

- 真实400条输入的CPU预演通过：两份清单SHA、样本唯一性、Qwen／Gemma／Whisper缓存来源、两套负条件逐值一致、各rank均覆盖两套数据和预留卡均核对。2358个历史GT特征NPZ可复用。预演没有加载GPU或冒充真实生成完成。
- 新加载路由与历史Whisper评测回归共5项通过；训练绑定源码身份仍逐项相同，新路由退出后恢复旧入口。
- 6000步推理副本正在独立校验／导出；正式prepare在8000步保存并停训后执行。真实GPU验收状态看`GPU_SMOKE_COMPLETE.json`，未生成前视为待运行。
- `status.json`当前应为`waiting_complete8000_and_training_stop`；后续以实际文件为准。

停止本评测可在本目录创建`STOP`，不影响原训练停在8000步的安排。训练目录的STOP也保持，评测结束不自动恢复训练。

```bash
../Omni-LTX-stage2/.venv/bin/python scripts/p2_combined_whisper_eval.py queue \
  --work experiments/p2_combined_whisper_eval_20261106
```

已有控制器时不要重复启动；该命令受独占锁保护。
