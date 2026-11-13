# Seamless固定诊断200条：旧联合8000 vs 全LoRA6500，双seed

**已全部完成：四轮各200条，共800条生成、解码、视频／音频评分与最终核验通过，耗时约43分钟。** 双seed均值：全LoRA6500的FID14.677403、FVD173.409345，旧联合8000为14.965719、179.976908；全LoRA的画质／分布指标更好，旧联合的情绪准确率及ΔResp更好。[双seed原值和平均分](RESULTS.md) · [完整JSON及覆盖](RESULTS.json) · [最终复核](FINAL_DELIVERY_REVIEW.json)。


用户最新范围为**固定诊断200条**，不运行完整test6624。两条pipeline各跑seed137、238，共800个视频和对应listener WAV。代码已接通并通过CPU核验；四机28卡已实际开始首轮，已验证28份真实latent，尚未完成解码／评分。其余三轮自动接续。实际进度见下方状态。

| pipeline | 原始训练权重 | 规划器 | seeds |
|---|---|---|---|
| 旧联合8000＋SFT2752 | `../p2_j_whisper_event_feature_train31_20261105/step_008000.pt` | SFT2752 | 137、238 |
| 全LoRA6500＋SFT2752 | `../p2_ltx_all_lora_full28_batch28_20261109/step_006500.pt` | SFT2752 | 137、238 |

两组均为AV＋Event＋Feature＋Whisper配置，但生成器权重和训练过程不同。本轮比较完整pipeline，不宣称严格单变量消融。

## 数据与设置

- 沿用已冻结Seamless固定200条／19.99分钟，`seamless_quick200_latest_20261105/prepared_200/references.jsonl`，SHA256 `d4bda633eefebfd0b0a1b055684a749d86a6b28ec7c9d1a032e4a151b0468b07`。不重新抽样。
- 复用 `p2_ltx6000_latest_sft_dual200_20261107` 中已验证的200条SFT2752规划、H、Gemma、Whisper、listener首帧及空Gemma条件，四轮共享完全相同的条件。两个seed只改变生成噪声，不重跑规划。
- 输入只有speaker音频和listener首帧，不输入未来listener画面、音轨或GT描述。
- 512×512／25fps；Euler30步，video CFG3／audio CFG7，沿用原生精度／采样／VAE补帧和裁尾方式。每片段独立设置相应seed，不做多候选筛选。
- 沿用原视频8项：PSNR、SSIM、LPIPS、FID、FVD、Emo-Acc、Emo-Div、ΔResp；另外保留已有UTMOS及音频活动评分、有效覆盖和N/A。face-padding0，batch64，CPU有界预取、像素计算4线程、已有GT RGB与特征缓存。

## 四机28卡与代码复用

| 节点 | 允许物理GPU |
|---|---|
| node12 / 172.29.252.37 | 0–7 |
| node8 / 172.29.104.237 | 0–7 |
| node4 / 172.29.225.51 | 0–7 |
| node11 / 172.29.189.119 | **0–3** |

node11的4–7卡从调度和worker核验中排除，不停止既有其他任务。使用原28卡共享领取、逐样本锁和断点提交，模型mmap加载，常驻VAE encoder；各卡生成结束后立即转原生解码／就绪分片评分。四轮顺序为：旧联合seed137、全LoRAseed137、旧联合seed238、全LoRAseed238；每轮可使用全部28卡。显存不足时等原调度准入条件，不挤占其他卡。

[执行入口](../../scripts/p2_seamless200_pipeline_two_seeds_20261112.py)复用[28卡渐进队列](../../scripts/p2_all_lora6428_seamless_eval28_20261109.py)，原评分公式和模型实现不改。旧模型复用其checkpoint绑定的历史源码，导出SHA完整核验一次后按文件身份mmap加载、严格state加载；全LoRA6500沿用已验证的batch28模型加载器。已验证两组绑定源码、6500的295个训练源码身份及batch28归一化配置；[7项CPU检查](../../tests/test_seamless200_pipeline_two_seeds_20261112.py)覆盖保留卡排除、节点映射、双seed平均及缺失值。

## 平均与结论

每轮先完成全部200条评分并用全量特征汇总FID/FVD，再对同pipeline的seed137和238最终分数求算术平均。不平均分片FID，不把两个seed混为400条后计算一个FID。保留每个seed原值、GT参考、支持数及N/A；只有两个seed都存在的指标才给平均值，不以0补缺。

最终按FID/FVD平均值报告分布质量更好的pipeline；如果两项胜者不同，明确报告取舍。PSNR、SSIM、LPIPS、情绪、响应和音频成绩分别展示，不造一个未经约定的综合分数，不基于两个seed宣称统计显著。

## 入口、结果与恢复

- [实际28卡启动与首批产物核验](STARTUP_ACCEPTANCE.json)、[准备与来源核验](PREPARED.json)、[28卡配置](cluster.json)、[提交PID](LAUNCH.json)、[控制器PID](CONTROLLER.json)、[总状态](status.json)、[日志](controller.log)。
- 四轮目录：`joint8000_seed137/`、`all_lora6500_seed137/`、`joint8000_seed238/`、`all_lora6500_seed238/`。各自`protocol.json`冻结权重、seed和条件来源；`status.json`、`devices/`、`logs/`记录实际执行。
- 各轮生成：`generated/<arm>/seamless/<id>/generated.mp4`及`listener.wav`；各轮`RESULTS.json`为全200评分，`FINAL_AUDIT.json`为完成核验。
- 最终双seed对比：本目录`RESULTS.md`和`RESULTS.json`；全800输出通过后才写`COMPLETE.json`。这些文件未出现时不能视作已完成。
- 设置根目录`STOP`会传播至当前子轮，原队列安全停止；各子轮也支持自己的`STOP`。恢复前先核实`CONTROLLER.json`的实际PID，已有控制器仍在运行时不重复启动。删去用户停止标记仅在用户要求恢复时执行。

项目根目录启动／恢复命令：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python \
  scripts/p2_seamless200_pipeline_two_seeds_20261112.py queue
```
