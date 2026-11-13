# ViCo43：固定联合8000，替换最新SFT2752

用户要求用最新SFT2752重新测评。本轮保持联合8000生成权重、43条正式ViCo test、speaker波形／listener首帧、Whisper、seed137、Euler30和CFG3／7不变；只将Qwen换为已完成2752步的r32/alpha64规划LoRA，重新生成描述、H与Gemma条件。提示词模板已逐条核对与原版相同。

**已完成：43/43条SFT2752生成、官方画质及BFM／DECA动作评分，数值／媒体／浏览器检查通过。** [全部43条同步视频与评分](http://localhost:8876/listen_benchmark_review/vico43_sft2752_20261107/) · [完整对比与配对区间](RESULTS.md) · [四模型JSON](COMPARISON.json) · [逐条规划差异](planning_comparison.csv)。原始Qwen与两种baseline封存结果保持不变。

PSNR 19.66829 → 19.33555；SSIM 0.66245 → 0.65587；FID 30.11631 → 29.42624。动作距离同时结合GT方差和来源聚类区间解释，人工语境适当性评分尚未开展。

**结论是取舍：** 平均表情方差／GT由37.2%升至71.1%，表情Delta-Fréchet由0.03584降至0.02591；逐帧表情L1由0.15394增至0.16719，差值来源聚类95%区间为[0.00589, 0.03396]。头旋转L1均值下降，但相对原始Qwen的区间跨零；PSNR、SSIM、CSIM、CPBD均下降。动作幅度更充分，不能据此宣称语境反应全面更好。训练使用base H，推理改为SFT H是本次替换的一部分，尚未分离描述变化与H变化各自的影响。

[论文表格PDF（含完整DECA附表与英文结果描述）](paper_tables/preview.pdf) · [主表LaTeX](paper_tables/main_comparison.tex) · [完整DECA附表](paper_tables/deca_complete_appendix.tex) · [全部配对区间附表](paper_tables/paired_complete_appendix.tex) · [英文结果描述](paper_tables/observations.txt) · [逐条DECA CSV](paper_tables/deca_per_clip.csv) · [逐条BFM CSV](paper_tables/bfm_per_clip.csv) · [表格、来源与编译校验](PAPER_TABLES.json)。L2L官方四人物迁移仍单独列出，保持停止重训；没有人工评分。

[完成状态](status.json) · [GPU7执行记录](GPU7_LAUNCH.json) · [模型、adapter与采样协议](protocol.json) · [输入及模型CPU验收](CPU_ACCEPTANCE.json) · [分片／提示词／适配代码预检](ADAPTER_CPU_CHECK.json) · [训练版本源码执行记录](PINNED_EXECUTION.json)。`RENDER_RUNTIME.json`是早期三卡调度快照，不表示仍有活跃worker。

Qwen使用 `experiments/p2_planning_sft_upright_3b_32gpu_resume_20261106/training/final_adapter/`；联合8000仍是 `experiments/p2_j_whisper_event_feature_train31_20261105/step_008000.pt` 对应的封存推理导出。没有换成LTX6000或6500，也没有新训练。训练H来自base Qwen，本轮推理H来自SFT2752，保留这一条件分布差异。

## 运行与恢复

- [生成入口](../../scripts/p2_vico43_combined8000_sft2752.py)：`prepare`、`run`、`worker`。源输入只复用首帧／输入审计及Whisper、空Gemma负条件；qwen/和text/均为新目录。
- [动作提取、配对比较与数值检查](../../scripts/p2_vico43_sft2752_compare.py)：观察指定的真实生成PID，完整视频／图像评分成功后接2路BFM／DECA提取，再聚合与独立复算。
- [新旧Qwen视频对照页面构建](../../scripts/build_vico43_sft2752_review.py)。页面在分数齐全后构建，避免发布半成品排名。

**执行记录（本轮已完成）：后续仅node11物理GPU7。** 发现另一项31卡训练已占GPU0–6后，已停止本次测评的0／1／2卡worker，保留已提交的8条视频与全部有效latent，移到预留推理卡接续。将原有采样与解码分成独立进程，减少Transformer与VAE同时驻留；逐项采样／解码运算、模型、种子及输入不变。初始CUDA分配上限40%；若仅因显存不足失败，则保留输出并等待GPU7空闲显存达到88000MiB再恢复，不占训练卡。[迁移记录](GPU7_MIGRATION.json) · [新入口](../../scripts/p2_vico43_sft2752_gpu7.py) · [执行进程记录](GPU7_LAUNCH.json)。

此前0／1／2卡三分片运行记录位于 `revisions/reserved_gpu7/`；[早期资源调整](EXECUTION_REVISION.json)为历史，不代表当前仍在三卡运行。

旧联合8000绑定的config.py与当前工作区不同；本轮执行训练时 `code_snapshot/dependencies/` 中SHA匹配的版本，逐项验证其余绑定文件，保留当前训练源码。没有放宽模型来源检查。

不要仅凭RUNTIME.json或队列文件重启任务；先检查其中实际PID和现有进程。生成按已提交latent／媒体断点复用，分片互斥。旧L2L训练保持停止。

## 输出位置

- 描述、H与Gemma：`conditions/vico/<uuid>/{qwen,text}/`。
- 原生视频／生成声音：`generated/combined8000_sft2752/vico/<uuid>/{generated.mp4,listener.wav}`。
- 实际评分副本：`scoring/combined8000_sft2752/<uuid>.listener.mp4`，256×256／30fps；`conversion/`记录完整时间映射，保留35秒长片段。
- 图像指标：已写入 `RESULTS.json`／`IMAGE_RESULTS.json`。
- 最终四模型对照：已写入 `COMPARISON.json`、`RESULTS.md`、`image_comparison.csv`、`planning_comparison.csv`及网页；动作参数在 `features/{bfm,deca}/`。

本轮未宣称DIM原表复现，BFM／DECA／EMOCA分开。ViCo自动视觉／动作距离不能替代语境适当性或自主声音质量的人工评价。

[原始Qwen版三模型测评及可视化](../p2_vico43_unified_20261107/README.md)。

[媒体／独立L1复算](FINAL_NUMERICAL_AUDIT.json) · [浏览器验收](BROWSER_AUDIT.json) · [评分代码及权重复用核验](SCORER_REUSE_AUDIT.json) · [声音完整性](AUDIO_INTEGRITY.json)。首批9条pilot页面为历史中间结果，不代替完整43条表。
