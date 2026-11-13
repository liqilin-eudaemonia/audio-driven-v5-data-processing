# 参考方法训练接入核查

**后续已推进到七方法真实训练短跑，最新状态见[实际训练接入与验收](../baseline_training_realdata_20261109/README.md)。本页保留最初核查证据；下表的待办是当时状态。纠正：DyStream 的 git 跟踪 `main.py` 实际包含训练步骤，最初“无训练循环”的判断有误。**

2026 年 11 月 9 日按 GLARE → DyStream／Avatar Forcing → L2L／DIM／ViCo／ListenFormer 的顺序核查。**尚未启动 500h 基线训练。** 当前已核实训练池和原始标注绑定、冻结七个上游版本，并跑通 ListenFormer 原生网络的 CPU 更新、GLARE 训练 CLI 和缩小宽深度的原生 FMT 两次更新。ViCo 飞书表格核对见 [逐表结论](FEISHU_VICO.md)。

## 统一训练数据

使用 [最终 train 加 dev 选集](../listen_j_upright_av500h_train_dev_v3_20261105/README.md)，共 **284,218 段／500.00125913 小时**，train 281,060、原 dev 3,158，test 为 0。全量 CSV 已重新核对 SHA、成员唯一性、split 和总时长；16 条当前 reader 回读通过 caption 文件 SHA、原文、角色和最终裁框检查。本次没有重新解码媒体。

[数据合同](DATA_CONTRACT.json) · [全量审计](AUDIT.json) · [16 条输入检查](INPUT_CANARIES.json) · [可复跑入口](audit.py)。选集 JSONL 的 SHA 使用原封存记录，CSV 的 SHA 是本次重新计算值。

各方法从同一清单解析最终 source 几何、转正信息、配对 speaker 音频、listener 监督视频及当前 caption。直接复用 `scripts/p2_j_current_qwen_hidden.py` 的 `Reader.row`／`item_from_row` 及原生媒体 reader；不要从旧裁框重新生成输入。已有 LTX latent、Qwen H、Gemma、Whisper 缓存不等于 LIA、EMOCA 或 BFM 特征。

当前 500h 已包含原 dev；原 dev 不能同时作为独立验证。验证划分需要固定并披露，不能根据 test 选择配置。没有文本输入的原生方法保留原输入形式，同时记录最终 caption 来源，不能仅为“用标注”而暗中改架构。

## 方法进度和阻塞

| 方法 | 本次确认 | 下一项必要工作 |
|---|---|---|
| GLARE | 训练 CLI 和原生 FMT 小规模 CPU 更新通过；官方 requirements 的 transformers 4.30.2 与现代码 Qwen2Audio 导入冲突已在隔离依赖中解决 | 确定短片段窗口方案；导出 LIA／音频／reaction／intensity HDF5；完整模型真实样本更新 |
| DyStream | 初审遗漏 `main.py`，其中有原生训练步骤；现已从公开权重完成真实短跑 | 后续固定代码／恢复验收与正式全量训练见新记录 |
| Avatar Forcing | ViCo43 共同输入推理已有验收；当前上游只有最小推理代码 | 补训练数据接口、diffusion forcing 及所选训练阶段；不能只改推理 YAML 来开训 |
| L2L | 既有停止的重训诊断使用 ViCo397，不是当前 500h | 导出本轮动作／音频并固定论文与发布配置差异；不直接恢复旧诊断冒充 500h 训练 |
| DIM | 有公开训练代码；500h 尚无对应成套特征 | 导出配对 EMOCA 与 HuBERT；检查冻结统计和训练范围 |
| ViCo | 有公开训练代码；现有项目部署不等于本轮训练 | 导出 BFM／音频特征及训练统计，之后执行真实训练样本更新 |
| ListenFormer | 官方代码已取到 `compare/ListenFormer`；CLI 和原生 CPU 网络训练更新通过 | 导出 73 维动作、45 维音频和所需 metadata／统计，验证真实数据 loader |

上游 tree、README 与 commit 保存在 [upstream](upstream/)，采集时间和 SHA 见 AUDIT。更正：DyStream 的 `main.py` 提供原生训练步骤；Avatar Forcing 的 README 声明最小推理管线，本轮已据其原生 forward／scheduler 补训练适配器，未宣称完整论文训练复现。

ListenFormer 的缺失依赖装在 `compare/.envs/listenformer-deps`，与现有 DIM 环境组合使用，未修改既有环境。新增了其 requirements 未列出的 `rotary-embedding-torch==0.8.6`。原生网络源码保持官方版本；[实测记录](LISTENFORMER_NATIVE_SMOKE.json)是合成输入 B2×8 帧、73 维输出，160 个参数张量获得梯度，`driven_proj.weight` 在 Adam 后改变。它只证明代码链路，不证明 500h 数据接入或收敛。

```bash
PYTHONPATH="$PWD/compare/.envs/listenformer-deps" CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  compare/.envs/dim/bin/python experiments/baseline_training_readiness_20261109/listenformer_smoke.py
```

## GLARE 的窗口阻塞

500h 所有片段为 **2.00–11.95 秒**。官方示例使用 250 帧音频＋125 帧参考＋10 帧历史，合计 385 帧／15.4 秒；原 loader 又按完整 60 帧 HDF5 chunk 取窗，至少需要 420 帧／16.8 秒。因此无填充、无越界上下文时，**可用片段为 0**。

仅把音频窗改成 2 秒仍至少需要 9.6 秒，投影覆盖 55,867 段／162.8532h；参考改成 25 帧也只覆盖 175,476 段／402.9275h。这些是时长投影，未采用为训练设置。直接缩窗口并宣称“同 500h”不成立。完整数值见 AUDIT 的 `glare_window_projection`。

需要选择并记录短片段的参考／历史与有效帧策略，再检查真实样本覆盖和训练损失。不能串接无关片段，也不能无记录引入选集外源上下文。reaction 是原生数值监督，intensity 是声学强弱分数，不能从自然语言 caption 随意伪造。

## ViCo 原表核查

从本地 DIM 原论文第 11 页重新提取 Table 1，8 行共 **96 个数值**与既有引用表完全一致。[数值审计](VICO_TABLE_AUDIT.json) · [论文页文本](DIM_table1.txt) · [当前两版完整成绩](../p2_vico_dim_var_per_coefficient_20261108_v2/RESULTS.md)。

当前本地单次生成的表情 FD：base-Qwen 25.89253、SFT2752 24.33909；DIM 发表 DIM-Listener† 为 23.88。它们仍有测试有效范围、原生系数与 RGB 反提取、候选采样等差异，不能据此作严格同协议排名。

最新已确认 Var 使用逐系数跨帧方差再平均。本地 GT 为 Exp **0.904523**／Pose **0.012038**，论文为 **0.93／0.01**；原先拉平方差约 1.456 不是当前采用值。当前 GT SID 为 5.04158／3.88317，论文为 5.03／4.07。统计更接近不能证明作者使用相同公式。十次择优仍为旧暂停状态，本次未恢复。

用户随后提供 [Writing 本地副本](../../feishu_docs/Writing/Writing.md)，已读取相关截图并核对原论文。**Avatar Forcing 表4继承 DIM 的三行24个数字完全一致。**其他截图的 FD 有逐帧 L1／分布 Fréchet 之别，并有乘100／1000的显示单位，不能拼成同一排行榜。†在 Avatar Forcing 表中指转引、在 DIM 中指 CANDOR 预训练，不能混读。[完整核查与我们的分数对照](FEISHU_VICO.md)。

## 运行边界

本轮使用 CPU 审计、原生网络短测和隔离依赖安装；没有分配 GPU、启动长训练、恢复 L2L 或十次择优队列。当前正在运行的模型训练源码和任务保持原状态。

GLARE 使用 `compare/.envs/avatarforcing/bin/python` 加 `compare/.envs/glare-deps`，保留原 Avatar Forcing 环境。隔离覆盖 transformers4.46.3、tokenizers0.20.3，并补齐 torchaudio2.10.0 等。[CLI 验收](GLARE_CLI_ACCEPTED.json)、[FMT 实测](GLARE_FMT_SMOKE.json)和[依赖清单](requirements-glare-overlay.txt)已保存。FMT 短测仅把宽度／层数减至64／2，使用原始512维运动、20维统计及六路统计条件；没有调用完整预训练音频／动作编码器，也未通过真实HDF5 loader，因此不等于完整GLARE训练已跑通。
