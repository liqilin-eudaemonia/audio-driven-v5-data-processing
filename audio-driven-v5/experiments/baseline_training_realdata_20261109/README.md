# 七个参考方法：真实训练数据接入与测试

对同事交接请先读[交付 README](../../related/P2_参考方法训练与评测_交付README_20261109.md)，其中汇总三项任务状态、代码与结果入口、复跑命令和下一步工作；本页保留训练接入的技术细节。

已将 GLARE、DyStream、Avatar Forcing、L2L、DIM、ViCo、ListenFormer 接到最新 500h 训练池中的真实样本，完成完整原生网络的前向、反向、参数更新、验证损失、断点重放和新进程恢复。七个方法、十个训练阶段、每阶段四次更新，共四十次原始更新，均通过固定代码验收。这里的完成指**训练链路短跑验收**；尚未完成 500h 全量特征预处理、正式训练或收敛评估。

可以向同事描述为：“七个参考方法已经用最终训练数据实际更新过模型，训练代码和断点恢复已按固定版本验收通过。DyStream、Avatar Forcing 可从公开权重继续训练；当前结果是链路测试，不能作为重训后的论文指标。”

## 测试范围与证据

- 数据池：[最终 train＋dev 选集](../listen_j_upright_av500h_train_dev_v3_20261105/README.md)，284,218 段／500.001259h，test 未加入。
- 真实测试样本：3 个训练片段、1 个分组独立验证片段，包含 2.00 秒短片段、11.95 秒最长片段及恢复方向的片段；样本、group、原 split、caption 和清单 SHA 见 [DATA_PROTOCOL.json](DATA_PROTOCOL.json)。该划分仅用于诊断；原 dev 已纳入 500h，不能冒称独立官方验证集。
- 每个训练阶段执行 4 次真实 Adam／AdamW 更新，然后从第 3 步重跑第 4 步，核对损失、参数和优化器状态；两阶段模型还检查调度器和模型内部随机数状态。DyStream 包括 EMA。
- 使用完整原生网络宽度和深度，无随机合成输入。每个方法检查预期分支有非零且有限的梯度，冻结参数没有梯度。
- 另起全新 Python 进程，从磁盘第 3 步断点恢复并比较第 4 步最终状态。执行状态见 [RUN_STATUS.json](RUN_STATUS.json)、[suite.log](suite.log)，完整汇总见 [SUITE_ACCEPTED.json](SUITE_ACCEPTED.json)，逐项交付校验见 [FINAL_AUDIT.json](FINAL_AUDIT.json)。
- [七项回归检查](../../tests/test_baseline_training_realdata.py)已通过，见 [日志](regression_tests.log)：最终源文件／caption／裁框绑定、分组隔离、原生特征空间、未知标签屏蔽、输入损坏拒绝、说话者转正像素，以及两项确定性运算与原运算的数值／梯度一致性。

| 方法 | 本次真实更新的阶段 | 主要输入与范围 | 验收记录 |
|---|---|---|---|
| GLARE | 完整 FLOAT，1024 宽／8 层 | 原生 LIA512、W2V、SER、Qwen2-Audio intensity、六类 reaction；短片段及完整 299 帧最长样本 | [ACCEPTED](runs/glare/ACCEPTED.json) |
| DyStream | 公开权重继续训练，完整发布架构 | 原生 LIA3D512 与双音频分支；50／96 帧真实片段 | [ACCEPTED](runs/dystream/ACCEPTED.json) |
| Avatar Forcing | 公开权重继续训练，完整 1024 宽／8 层 | 原生 LIA512、真实说话者音频和动作、听者静音；50 帧 | [ACCEPTED](runs/avatarforcing/ACCEPTED.json) |
| L2L | VQ 编码器 4 步＋FACT 预测器 4 步 | DECA exp50＋pose6、原生 128 维 mel；64 帧窗口 | [ACCEPTED](runs/l2l/ACCEPTED.json) |
| DIM | 说话者 VQ 4 步＋听者 VQ 4 步＋预测器 4 步 | EMOCA-v2 pose6＋exp50、真实 HuBERT768；60／64 帧 | [ACCEPTED](runs/dim/ACCEPTED.json) |
| ViCo | 原生 LSTM 4 步 | BFM73＋音频45；60／90 帧 | [ACCEPTED](runs/vico/ACCEPTED.json) |
| ListenFormer | 原生 Transformer 4 步 | BFM73＋音频45；60／90 帧 | [ACCEPTED](runs/listenformer/ACCEPTED.json) |

各方法的 loss 定义不同，不能将验收 loss 横向排名。这些更新次数也不足以说明模型已经学好、优于预训练权重或达到论文指标。

## 接入时修正的实际问题

**最终数据身份。** 复用当前 `Reader.row`／`item_from_row` 和媒体 reader，核对最终自然语言 caption 原文与文件 SHA。听者使用最终裁框和已恢复方向的源；额外说话者画面使用独立记录的原生裁框。一个说话者原片横置 90 度，经逐像素核对后转正，记录在 [SPEAKER_ORIENTATION.json](SPEAKER_ORIENTATION.json)。不能把听者转正方向直接套用到其他说话者源。四个样本的 BFM/FAN 检测均覆盖全部真实帧。

**GLARE 窗口。** 原示例所需上下文超过本数据所有片段。接入采用首帧作 1 帧参考、片段开始不要求历史，其余帧按真实长度监督；没有串接无关视频或补造视频帧。改动参考／历史长度及 attention mask，保留原生参数规模。冻结 LIA、W2V、SER；修正上游 `inference()` 对 audio projection 的整体 `no_grad`，恢复这层的训练梯度。Qwen2-Audio 输出的四份强弱分析均为 `NULL`，按原生解析处理，并保留文本回执；没有用 caption 人造强度。

**检测缺失。** 一个短片段 FER 检测不完整。训练专用适配器将整段四个 FER 类别设为未知并屏蔽损失，保留有完整检测支持的点头／摇头标签；没有插值伪造目标。基准评测的检测器与评分规则没有改。

**训练监督与推理条件。** GLARE 保留原生的动作统计条件，本次从训练目标动作计算均值、标准差、速度和加速度统计。DyStream 也保留原生的 GT anchor 训练逻辑。这些是训练目标的一部分，本次 loss 不代表只给首帧和音频的部署推理效果。媒体回执中的 `future_listener_input=false` 描述部署输入约定，不表示原生训练步骤不读取 GT；未来正式推理评测必须独立检查可用条件。

**公开权重。** DyStream 的动作空间是其自己的 LIA3D，不能与其他模型的 512 维 latent 混用。Avatar Forcing 两个文件的名字与内容容易误判：`flow_transformer.pth` 实际包含 motion autoencoder，`motion_autoencoder.pth` 包含其余生成参数；本次按实际 key 严格加载，另对 W2V 做逐张量核对。GLARE 的 LIA 使用前者中全部 212 个原生张量。

**DyStream 训练代码的更正。** 前一份核查把它记成缺失训练循环，这个判断有误。当前上游 git 跟踪的 `main.py` 包含 `_step` 和损失；本接入直接执行其中原始方法。该方法监督干净动作，并在训练时从末尾 10 帧 GT 取 anchor；本次保留并披露，验证 loss 也是该训练目标，不是部署推理效果。发布 sample 配置的学习率为 0，诊断明确改为 AdamW 1e-5，保留 EMA；未声称复刻完整训练配方。

**Avatar Forcing 训练范围。** 上游提供最小推理管线及原生训练 forward／scheduler。本次补了本地 blockwise diffusion-forcing 训练适配器，使用原生加噪、目标和权重计算加权 MSE，验证公开权重可以继续更新。**没有复现论文偏好优化阶段，也不将这个适配器称为官方完整训练代码。** 听者输入音频为明确的静音，真实说话者音频与动作作为条件，听者参考只取首帧。

**L2L／DIM。** 分别保持 DECA 与 EMOCA 的维度顺序，DIM 使用 HuBERT 最后一层的真实 768 维音频特征，没有拿 W2V 或全零冒充。L2L 保留发布代码的优化器和 100000 步 warmup，按原生方式先逐窗口平滑，再仅用训练窗口拟合统计；两条 60 帧短片段不足原生 64 帧窗口，明确排除，见 [WINDOWS.json](runs/l2l/WINDOWS.json)。这也说明正式训练前必须统计各方法的有效数据覆盖，不能默认都完整使用 500h。

**确定性恢复。** ViCo 使用原生 ATen LSTM 代替 cuDNN 的不透明 dropout RNG 缓存；DIM 将同一 token CE 展平为确定性 2D kernel；DyStream 的 linear 音频插值反向转 CPU 运算并保留 autograd。数值／梯度回归通过。它们保留原生模型架构和目标，并在协议中记录运行差异。

## 可复跑入口

适配源码：[src/baseline_training](../../src/baseline_training)；统一 CLI：[p2_baseline_train.py](../../scripts/p2_baseline_train.py)；全套真实训练与新进程恢复：[p2_baseline_suite.py](../../scripts/p2_baseline_suite.py)。[代码快照](code_snapshot/)及 [SHA 清单](CODE_FROZEN.json)固定了本轮适配代码；各方法 `protocol.json` 另绑定原生源码、配置、权重和特征身份。

在项目根目录复跑全部七项（会重写这个诊断目录的测试权重和日志）：

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python \
  scripts/p2_baseline_suite.py run
```

只复跑指定方法可追加 `--methods dystream avatarforcing`。默认完整套件依次执行七个方法，每个阶段 4 步，随后独立进程恢复。它只使用本机物理 GPU7，有 24GiB 进程分配上限，并在开始时要求至少 26GiB 空闲显存；不停止其他任务。

单项原始入口示例（L2L）：

```bash
BASELINE_PROJECT_ROOT="$PWD" PYTHONPATH="$PWD/src:$PWD/scripts" \
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
compare/.envs/l2l/bin/python \
  experiments/baseline_training_realdata_20261109/code_snapshot/scripts/p2_baseline_train.py \
  validate --method l2l --steps 4
```

GLARE 使用 `avatarforcing` 环境并叠加 `compare/.envs/glare-deps`；ViCo／ListenFormer 使用 `dim` 环境并叠加 `compare/.envs/listenformer-deps`；DIM、DyStream、Avatar Forcing 各使用同名隔离环境。主项目 Python 用于媒体导出；BFM／DECA／EMOCA／HuBERT 导出使用 `vico_more` 环境。套件脚本已封装这些选择，优先使用套件。

Python 与关键依赖的实测版本见 [ENVIRONMENTS.json](ENVIRONMENTS.json)；没有改动主训练环境。

全套回归：

```bash
PYTHONPATH=src:scripts CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 \
  compare/.envs/l2l/bin/python tests/test_baseline_training_realdata.py
```

## 正式 500h 训练仍需完成的部分

当前只物化了上述四个真实样本的媒体和各方法特征。完整选集 reader 与媒体／特征 `--all` 导出入口已接入；**尚未全量导出、审核失败样本、冻结有效覆盖或执行长训练。** GLARE／ViCo／ListenFormer 有全池惰性训练入口；全池 BFM 必须提供仅由训练数据拟合的统计。L2L／DIM／流式方法当前入口用于分阶段真实诊断，正式全池仍需补惰性窗口索引、特征准入和阶段调度，不能把本次短跑直接当作 500h 正式作业。

正式运行前还需固定各方法可用样本与窗口覆盖、独立分组验证集、训练步数／初始化／公平预算。Avatar Forcing 若要宣称论文完整方案，还需偏好训练阶段。当前检查点仅用于接入与恢复测试，不替代公开权重作为性能基线，也不计作训练完成。

ViCo 飞书表格核查与 LM Listener 仍分别见 [表格证据](../baseline_training_readiness_20261109/FEISHU_VICO.md)、[LM Listener 当前卡点](../lm_listener_readiness_20261109/README.md)，这次训练验收未产生它们的新正式评分。
