# P2：GLARE 四项反应评测实现与运行

新增独立 [OmniResponse 音视频与可选文本评测](P2_OmniResponse评测实现与运行.md)，默认 FVD-32、UTMOSv2、SyncNet，并提供 FD/文本接口。原版 E1 和新版 E0 的生成请使用该文档的 `p2_eval_generate.py` 自动选择对应训练源码；本文件以下四项 GLARE 指标口径保持不变。

2026-10-26。按用户确定的范围实现公开四项反应指标与完整视频评测流程；评测数据由用户稍后提供。**尚未启动正式 E1 生成评测。** 本次工程验证使用既有 debug 视频，与正式效果结果区分。

## 已实现的流程

`最终检查点 → source-only AV 生成 → MediaPipe/FER 独立检测 → 六类规则与互斥 CSV → 四项指标 → JSON/Markdown 报告`。

入口：[scripts/p2_glare_eval.py](../scripts/p2_glare_eval.py)；实现：[src/p2_glare_eval](../src/p2_glare_eval/)；数据示例：[examples/p2_glare_eval](../examples/p2_glare_eval/)。也可直接给出已有成对视频，跳过生成。

生成复用原生 `p2_listen.generation.generate`，加载当前 Connector、Event head 和 text-attention LoRA 检查点。Event head 的 `probe.pt` **不进入本评测**。模型仅使用 speaker 音频、listener 首帧、固定文本及输入时长；完整 listener GT 视频独立存放，只在检测/评分阶段读取。生成的 listener 音频保留，但这四项指标只评价视觉反应，不评价七类训练标签中的音频回应。

本次新增独立包及 CLI；没有修改 P2 训练、v4 复用源码或训练虚拟环境。检测器使用单独的 Python 3.10 CPU 环境；FID 的 `pytorch-fid` 安装到独立目录，不向训练环境安装包。

## 指标口径

直接调用固定版本 [GLARE eval](https://github.com/lzk901372/glare/tree/26f621e1620066885828d01739437beb71ed5204/eval) 的函数，原源码/模型与版本哈希保存在 [upstream lock](../configs/p2_glare_upstream.lock.json)。

| 指标 | 实际计算 | 方向与特殊情况 |
| --- | --- | --- |
| R-F1 | 全数据集按类累计 TP/FP/FN，计算六类 F1 再等权平均 | 越高越好；缺席类计 0 |
| R-tIoU | 所有成功匹配事件的时间 IoU 均值 | 越高越好；无匹配计 0 |
| R-ATD | 相对 GT 时长归一化的起点、终点、时长偏差，负偏差权重 2，正偏差权重 0.5，三项相加再对匹配事件平均 | 越低越好；报告乘 100，并同时保存原始值；无匹配为 null |
| R-FID | 两侧分别提取自己的全部反应帧，用 FID 专用 Inception-v3 2048 维 pool3 计算总体 FID | 越低越好；任一侧少于两帧为 null |

六类固定为 smiling、laughing、frowning、surprised、nodding、head_shaking。分数 **严格 >0.5** 才形成事件，单个 neutral 帧也会断开事件；区间为 `[start,end)`。同视频同类别内按 tIoU 降序贪心一对一匹配，阈值为 ≥0.5，不跨视频匹配。

R-FID 包含未匹配事件的反应帧，不强行令两侧帧数相同，不抽帧、重新裁脸、按类别均衡或逐片段算 FID 再平均。输入视频只做 Inception 原生缩放和归一化。模型权重离线准备并校验 SHA256；计算使用 float64 累计均值和样本协方差。

多个生成 seed 各自使用完整相同样本集合评分，再等权汇总；每个 seed 的 FID 池中 GT 只出现一次。某个 seed 的某项为 null，该项跨 seed 均值也为 null，不静默只平均有效 seed。

## 检测器及适用边界

使用作者公开的 [visual_reaction_annotation](https://github.com/lzk901372/visual_reaction_annotation/tree/8e7cd4b8ea1d54a55e127f530476b7033b327f78)：MediaPipe face landmarker、FER 情绪模型＋MTCNN、六类规则脚本及原始合并器。所有规则参数在 [p2_glare_detector_v1.json](../configs/p2_glare_detector_v1.json) 固定，点头采用 v2 Python 默认参数。

上游批量 FER 脚本调用了非标准 `Video` 扩展接口，这里改为公共 `FER.detect_emotions(BGR)` 逐帧接口；保留原 FER 模型。最终类别使用公开合并器的 **逐帧最大值选择**，平分按固定列顺序，不将其描述为论文中按事件均值裁决的完整复现。R-ATD ×100 是所固定代码的输出约定，不能据此声称恢复了论文未说明的尺度。

本版采用严格检测准入：每帧 MediaPipe 关键特征完整、FER 恰好一张有效人脸、两者 bbox IoU≥0.1；任何漏检、缺失字段、非有限值、丢帧均留下失败记录，不填 neutral。输出 `face.pkl`、`fer_raw.json`、`fer.csv`、六类强度、最终 `scores.csv`、`quality.json`，可逐帧追查。

GT 与生成视频必须同尺寸、同帧数、25 FPS、PTS 从零连续；检查实际解码帧，不只读取容器标称 FPS。原生输入允许非整数帧时长，预期 GT 帧数为 `ceil(duration_s * 25 - 1e-6)`。不为通过检查偷偷补帧、截短或重定时。

**这些结果是独立检测器给出的参考一致性，不是人工真值，也不直接证明回应 speaker 的适当性。** 稀有类及漏检率必须结合逐类计数查看；完整 E0/E1 配对实验才能判断 Event loss 的增益。本轮不包含 PSNR、SSIM、FID、FVD、LPIPS、Var、rPCC、DI-Sync 的额外实现。

## 环境准备

当前机器已准备独立检测环境、原仓库、FER/MediaPipe/FID 权重。新机器可运行安装命令；原机器只需检查：

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
P2_PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
"$P2_PY" scripts/setup_p2_glare_eval.py --check-only
# 新环境才运行：
# "$P2_PY" scripts/setup_p2_glare_eval.py
```

安装脚本仅创建 `.venv-p2-glare-detector`、`.eval_deps`、`.eval_models` 及缺失的公开参考仓库，不修改训练 Python 的 site-packages。依赖入口为 [requirements](../requirements-p2-glare-detector.txt)，完整已验证环境为 [freeze](../configs/p2_glare_detector_environment.freeze.txt)。需系统 `ffmpeg`/`ffprobe`、Python 3.10 和 uv。

## 已有 GT / 生成视频

先按 [清单示例](../examples/p2_glare_eval/pairs.example.json) 准备 `pairs.json`，路径相对该文件解析：

```bash
"$P2_PY" scripts/p2_glare_eval.py prepare-pairs \
  --manifest /absolute/path/pairs.json --output /absolute/path/eval_v1
"$P2_PY" scripts/p2_glare_eval.py evaluate \
  --work /absolute/path/eval_v1 --workers 4 --device cpu --batch-size 16
```

`--workers` 是并行 CPU 视频检测进程数；每进程限制 Torch/TF/BLAS 线程。`--device` 只控制 FID。已有经独立检测并后处理的 CSV 可以通过清单传入，需注明来源。`--detection-only` 可先检查整个集合的检测覆盖与视频一致性，暂不算四项。

## 当前 E1 检查点生成后评测

先提供与 train 会话/参与者身份隔离的 observations 和 GT targets。`prepare-generation` 会校验输入媒体哈希、split、训练身份交集和 GT 几何；不会启动模型。

```bash
P2_RUN=/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_e1_train50h_lora_tb_20261026/runs/20261023/E1
EVAL_GEN=/absolute/path/e1_generation_v1
"$P2_PY" scripts/p2_glare_eval.py prepare-generation \
  --run "$P2_RUN" --observations /absolute/path/observations.json \
  --targets /absolute/path/targets.json --output "$EVAL_GEN" --seeds 137 251
"$P2_PY" scripts/p2_glare_eval.py run \
  --work "$EVAL_GEN" --checkpoint "$P2_RUN/step_001268.pt" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  --wait --wait-timeout 86400 --evaluation-output /absolute/path/e1_eval_v1 \
  --workers 4 --device cpu --batch-size 16
```

`run` 调度本机所列 GPU，各样本/seed 分配给一个 worker。仅接受此 run 的最终完整检查点（step=budget），并等待最终 `status.json`；原生 test split 的 release JSON 自动绑定 config、checkpoint、sample IDs 和 seeds。`--wait` 同时等待显存被训练释放；显存占用 >2 GiB 的卡视为忙，避免撞上当前训练。

三机共同使用相同路径的 work 目录时，可分别启动 `generate --shards 24 --shard-index N --device cuda:M`，N 为 0…23、每机 M 为 0…7，每个 N 只启动一次。所有分片结束后在一台机器执行 `export-pairs`、`prepare-pairs` 和 `evaluate`。本版不会自动 SSH 启动其他机器。

```bash
"$P2_PY" scripts/p2_glare_eval.py generate \
  --work "$EVAL_GEN" --checkpoint "$P2_RUN/step_001268.pt" \
  --shards 24 --shard-index 0 --device cuda:0 --wait
# 等待全部 24 个分片完成后：
"$P2_PY" scripts/p2_glare_eval.py export-pairs --work "$EVAL_GEN" --output "$EVAL_GEN/pairs.json"
"$P2_PY" scripts/p2_glare_eval.py prepare-pairs --manifest "$EVAL_GEN/pairs.json" --output /absolute/path/e1_eval_v1
"$P2_PY" scripts/p2_glare_eval.py evaluate --work /absolute/path/e1_eval_v1 --workers 4 --device cpu
```

生成复用当前原生逐样本加载流程，本次未增加常驻模型的提速改写。缓存有效的已完成样本可跳过；失败样本在相同协议下可以重试。

## 断点、报告与失败

冻结 `generation_protocol.json` / `protocol.json`，绑定输入、源码、检测器参数、环境与权重身份。媒体全 SHA256 校验；生成绑定训练原有 assets 身份，检测模型和 FID 权重额外固定 SHA256。完成标记最后原子写出，复用时核对输出哈希；不同模型、seed 或协议不能混用旧目录。

- `generation_shard_NNN.json`：分片完成/失败情况；`generation_failures/` 保留错误；临时失败输出留在 `attempts/`。
- `progress.json`、`detected_pairs.json`：逐对检测/对齐结果；检测缓存每个唯一视频只计算一次。
- `results.json`：整体状态、每 seed 四项、六类 TP/FP/FN/P/R/F1、事件匹配、ATD 原值、反应帧数及 null 原因。
- `report.md`：简明结果和口径；详细错误见 JSON 与检测 worker.log。

任何必要样本失败，状态为 `incomplete`，完整集合的 metrics 为 null，CLI 非零退出。评分本身异常为 `scoring_failed`，同样非零退出。不会将有效子集分数冒充完整集合分数。修改代码、检测协议或冻结输入后请新建 work；已发布 pairs 中的失败记录在成功重试后也应导出到新版本 pairs / eval 目录，保留旧报告。

## 验证

验证记录见 [工程验收目录](../experiments/p2_glare_eval_validation_20261026/README.md)。测试覆盖公式金标准、阈值、无匹配、缺席类、非有限输入、多 seed、反应帧选择、时间轴错位、视频/CSV 不一致、失败保留、缓存损坏、检查点绑定、训练划分隔离和生成不读取 GT。

已实际运行 CPU MediaPipe＋FER＋六类规则＋FID Inception 完整链路；工程自配对的 R-FID 为 0。真实视频漏检也实际触发失败路径。生成调度使用 mock 原生生成器验证；本次没有占用训练 GPU 重新生成 dev/test。真实 GPU 生成仍复用此前已经验证的 P2 原生入口，待用户提供正式数据后执行这一轮端到端模型评测。
