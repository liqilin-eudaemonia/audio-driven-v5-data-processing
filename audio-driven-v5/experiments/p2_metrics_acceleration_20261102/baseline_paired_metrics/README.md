# 配对视频评测框架使用说明

需要同时评测视觉回应与音频时，使用 [Listener完整入口](../../listener_metrics/README.md)；它调用本文八项指标，再接GLARE、音频事件、UTMOSv2和SyncNet。本文入口本身仍只处理视频。

统一入口：`python -m evaluation.video.paired_metrics.run`（在 OmniResponse 仓库根目录运行）。

**2026-11-02 复核修订：** FVD 预处理改为直接复用原框架的双线性缩放＋取整；DeepFace 改为先归一化、再浮点缩放，与官方 analyze 的同裁图路径一致。新增独立公式／上游实现交叉检查，见 [复核报告](validation/AUDIT_20261102.md)。旧验收记录保留，不代表修订版代码哈希。

新增这套入口覆盖 PSNR、SSIM、FID、FVD、LPIPS、Emo-Acc、Emo-Div、ΔResp；原有 FD/FVD、音频、文本和已封存的 official_responsenet 评测保持原状。**这里的 FID 是图像 Inception FID，不是旧 `fd/` 的任意特征 FD。**

## 从输入到结果

已接入的数据集：[seamless（6,624 条／11.05 小时）](../../seamless/README.md)、[ResponseNet J 切分 pair（586 条）](../../responsenet_pairs/README.md)，含已核验的参考清单、生成结果映射模板和专用自检入口。

1. 准备同一测试片段的 **GT listener 视频**与**生成 listener 视频**，保持同一身份、同一时间起点、同一 FPS，以及相同的画面尺寸／裁剪语义。
2. 写一个 `pairs.json` 数组或 `pairs.jsonl` 清单。为计算默认 FVD，至少准备两对视频，每对共同有效长度至少 32 帧；25 fps 时约 1.28 秒。小样本只用于排查链路，正式比较应使用固定的完整测试清单。
3. 在 OmniResponse 根目录，使用下方环境和命令执行评测；每轮使用一个新的输出目录。
4. 先检查 `summary.json` 的 `status`、`coverage`，再读 `metrics`；有失败时到 `pairs.jsonl` 查看具体样本原因。
5. 比较不同模型／训练配置时，固定 GT、样本集合、裁框、帧率、采样参数和权重版本。

**这八项指标只读取视频画面。**不需要 speaker audio、listener WAV、ASR、事件文本或训练 latent 缓存，也不会评测音频干净程度、语音内容或音画同步。音频评测仍使用上级目录现有的 [音质入口](../../audio/audio_quality/README.md) 和 [音画同步入口](../../audio/audio_sync/README.md)。

对于当前 Seamless listen pair，应使用最终确认的 listener 裁框对应视频作 GT，生成端提供同一 listener 的输出视频。整个 pair 处于倾听角色时使用 `--whole-clip-listening`；不要把官方词／句／VAD 的发声区间当作“倾听区间”。

## 输入清单

JSON 数组或 JSONL；相对视频路径相对于清单所在目录。每行一个已知身份，`id` 必须唯一。双人视频应分别提供正确的身份 ROI，不自动按“左脸／右脸”猜身份。

| 字段 | 是否必填 | 含义 |
|---|---|---|
| `id` | 是 | 本轮唯一的样本／身份 ID；用于结果追溯，不用于猜测视频文件名 |
| `gt` | 二选一 | 对应真实视频的本地路径，与 `gt_reference` 互斥 |
| `gt_reference` | 二选一 | seamless 适配器生成的冻结来源引用，按需解码 GT；必须同时设置 `dataset: seamless`，不可再设 `gt_roi` |
| `dataset` | 否 | 数据集名称；seamless 接入明确使用 `seamless`，汇总输出 `datasets` |
| `generated` | 是 | 对应生成视频的本地路径 |
| `identity` | 否 | 身份名称，默认 `listener`；它是记录字段，不会启动身份识别 |
| `listening_intervals` | 条件必填 | 有序、互不重叠的 `[start,end]` 秒区间；只影响 ΔResp，右端点不包含 |
| `gt_roi` | 否 | GT 固定裁框 `[x,y,width,height]` |
| `generated_roi` | 否 | 生成视频固定裁框 `[x,y,width,height]` |

seamless 的 `gt_reference` 由其绑定入口生成，不能自行填一个 MP4 路径替代；读取需 PyAV（当前隔离安装 17.0.1）及相邻 audio-driven-v5 源码。GT 不经过中间有损编码，`gt_sha256_kind=decoded_full_gt_rgb_sequence` 表示完整解码 RGB 序列哈希，另有来源行及读取代码哈希；普通 `gt` 文件路径仍记录文件字节哈希。

普通对话示例（实际文件路径和区间需替换，区间不能超出视频）：

```json
[
  {
    "id": "sample-001-listener",
    "identity": "listener",
    "gt": "/path/to/gt_listener.mp4",
    "generated": "/path/to/generated_listener.mp4",
    "listening_intervals": [[0.0, 2.3], [3.0, 5.0]]
  },
  {
    "id": "sample-002-listener",
    "identity": "listener",
    "gt": "/path/to/gt_listener_002.mp4",
    "generated": "/path/to/generated_listener_002.mp4",
    "listening_intervals": [[0.0, 4.0]]
  }
]
```

可选 `gt_roi`、`generated_roi`：原图中的 `[x,y,width,height]` 整数框，必须在画面内。两侧裁剪后的尺寸必须相同。GT/生成必须从同一时间起点开始、恒定且相同帧率。按帧索引对齐，较长视频的尾部裁去并报告数量；不自动做时间伸缩或身份对齐。OpenCV 能提供时间戳时检查均匀性，否则明确报告未检查，调用方需保证 CFR。清单中的倾听区间以 GT 起点为 0，不能重叠或超出 GT；超出共同配对尾部的部分裁去并保留 original_intervals。

**倾听阶段由任务定义，不能用“listener 静音”替代。**本项目 listen-only pair 全段处于倾听角色，即使存在 “yeah” 回应也仍是倾听阶段；这种清单可以使用 `--whole-clip-listening`。一般对话视频必须显式提供区间；缺少区间只会令 ΔResp 不可测。

使用全段倾听开关时，可以省略 `listening_intervals`。若清单已写此字段，则优先使用清单区间；显式空列表 `[]` 不会被开关覆盖，会导致该样本没有可测倾听阶段。

## 运行

当前机器已验证的环境：`/private/software/conda/envs/AnyTalker/bin/python`。LPIPS/pytorch-fid 以 `--no-deps --target` 安装在独立目录，没有升级已有训练环境。

下面命令用于**全段倾听片段**。`/path/to/...` 是占位路径，需要替换；这不是自动读取某个训练目录的命令。

```bash
cd /private/liqi_workspace/project_code/OmniResponse
PYTHONPATH=/private/liqi_workspace/.cache/paired_video_metrics/python \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/private/software/conda/envs/AnyTalker/bin/python \
  -m evaluation.video.paired_metrics.run \
  --manifest /path/to/pairs.jsonl \
  --assets evaluation/video/paired_metrics/assets.local.json \
  --output /path/to/new_evaluation_directory \
  --whole-clip-listening --device cpu --threads 2 --batch-size 16
```

需要加速可把 `--device cpu` 改成指定空闲 GPU，例如 `cuda:0`，仅影响 LPIPS/FID/FVD；人脸检测和表情模型仍使用受控 CPU 线程。首次模型加载后逐段处理，内存保留当前一对视频与抽样分布特征，面向几秒级评测视频。

若只开放物理 GPU 3，可在同一命令前加 `CUDA_VISIBLE_DEVICES=3`，并使用 `--device cuda:0`；此时进程内编号 0 对应物理卡 3。当前入口是单进程、单 Torch 设备，不会自动调用所有 GPU。`--batch-size` 只调节 LPIPS/FID 帧批量，FVD 仍逐片段，表情分类批量固定为 32。

普通对话清单已提供 `listening_intervals` 时，使用同一命令并去掉 `--whole-clip-listening`。

仅测像素指标无需模型权重：

```bash
python -m evaluation.video.paired_metrics.run \
  --manifest pairs.jsonl --output pixel_results --metrics psnr,ssim
```

`--metrics` 支持：`psnr,ssim,fid,fvd,lpips,emo_acc,emo_div,delta_resp`。新环境依赖见 [requirements.txt](requirements.txt)；请在独立环境安装，PyTorch/torchvision 应选择相互匹配的版本。当前实测版本写在每次输出的 contract.json，ONNX Runtime 可使用 CPU 包或 GPU 包的 CPU provider，不要同时安装两个包。

### 参数速查

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--manifest` | 必填 | JSON/JSONL 配对清单 |
| `--output` | 必填 | 新结果目录；已有评测文件时拒绝覆盖 |
| `--assets` | 无 | 权重路径 JSON；完整八项指标使用 [assets.local.json](assets.local.json) |
| `--metrics` | 全部八项 | 小写逗号分隔，不加空格 |
| `--device` | `cpu` | Torch 模型设备，例如 `cuda:0` |
| `--threads` | `2` | 相关计算库的线程配置 |
| `--batch-size` | `16` | LPIPS/FID 的帧批量；显存不足时减小 |
| `--face-padding` | `0` | 检脸／关键点前每边固定补黑边像素数，随后还原原图坐标；对 GT 和生成同样处理。ResponseNet 接入配置为 64 |
| `--fid-frames` | `32` | 每段 FID 等距采样的最多帧数 |
| `--fvd-frames` | `32` | 每段 FVD 的采样帧数，至少 16 |
| `--fvd-stride` | `1` | FVD 帧间隔；实际要求长度 `(frames−1)×stride+1` |
| `--whole-clip-listening` | 关闭 | 对缺少区间字段的样本声明全段倾听 |
| `--allow-partial` | 关闭 | 允许部分指标失败时退出码仍为 0；不会修复错误或把状态改为 complete |

当前没有断点续跑开关。中断后保留原目录作为记录，用新目录重跑所需清单。不要把分片 FID/FVD 的标量直接平均成全量分数；它们需要全量特征分布重新汇总。

### 换机器或新环境

在独立评测环境安装相互匹配的 PyTorch/torchvision，再安装本目录 `requirements.txt`；不要覆盖训练环境。迁移模型文件并修改 `assets.local.json` 中的绝对路径。评测入口不会自动下载权重。

本机无需重新安装。已有 `AnyTalker` 环境必须带上命令中的 `PYTHONPATH`，才能找到独立安装的 `lpips` 与 `pytorch_fid`。查看命令帮助：

```bash
/private/software/conda/envs/AnyTalker/bin/python \
  -m evaluation.video.paired_metrics.run --help
```

## 权重与可复现性

[assets.local.json](assets.local.json) 记录这台机器的现有权重路径，换机器时修改路径。评测过程不联网下载，也不以缺权重的随机网络计算结果。新增下载的权重位于 `/private/liqi_workspace/.cache/paired_video_metrics/assets`，未加入仓库。

权重来源：

- FID：[pytorch-fid 官方权重](https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth)，不是 torchvision 普通 ImageNet Inception。
- LPIPS：[官方实现](https://github.com/richzhang/PerceptualSimilarity)，0.1.4 包含线性距离权重；AlexNet 为 torchvision 官方 `alexnet-owt-7be5be79.pth`。
- FVD：沿用现有 `evaluation/video/fvd/frechet_video_distance.py` 使用的 `i3d_torchscript.pt`。
- Emotion：[DeepFace 官方表情权重](https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5)。
- InsightFace：[buffalo_l 官方模型包](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip)，只需要 `det_10g.onnx` 和 `2d106det.onnx`，放在 `insightface_root/models/buffalo_l/`。

每次输出：

- `contract.json`：代码、输入清单、权重 SHA256，运行参数、包版本与评测口径。
- `pairs.jsonl`：每段结果、GT/生成路径与 SHA256、帧数、检测覆盖、倾听阶段幅度及错误。
- `summary.json`：数据集汇总、有效样本数／帧数／阶段时长、完整或部分状态。

请求的指标有失败时默认退出码 2，成功指标照常保存；`--allow-partial` 只改变退出码，仍保留 partial 状态。结果目录不可覆盖。不同指标可能有不同有效子集，比较系统时应先对齐有效清单、检查覆盖率，不把漏检产生的高分当成优势。

## 如何读取结果

`summary.json` 的指标字段对应关系：

| 展示名 | `metrics` 字段 | 读取方式 |
|---|---|---|
| PSNR ↑ | `psnr` | dB；可能为字符串 `"inf"` |
| SSIM ↑ | `ssim` | 直接报告，越接近 1 越相似 |
| FID ↓ | `fid` | 数据集分布距离 |
| FVD ↓ | `fvd` | 数据集视频特征分布距离，必须同时记录采样协议 |
| LPIPS ↓ | `lpips` | 平均感知距离 |
| Emo-Acc ↑ | `emo_acc` | 0–1；展示百分比时乘 100 |
| Emo-Div | `emo_div_generated`、`emo_div_gt` | 两个值同时展示，例如“生成 / GT”；不只展示生成值 |
| ΔResp ↓ | `delta_resp` | 倾听阶段的时长加权运动幅度差 |

快速读取（替换结果路径）：

```bash
python - /path/to/new_evaluation_directory/summary.json <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    result = json.load(f)
print("status:", result["status"])
print("requested_pairs:", result["requested_pairs"])
print("pairs_with_errors:", result["pairs_with_errors"])
print("metrics:", json.dumps(result["metrics"], ensure_ascii=False, indent=2))
print("coverage:", json.dumps(result["coverage"], ensure_ascii=False, indent=2))
print("errors:", result["errors"])
PY
```

先看这些覆盖字段：`coverage.fvd.pairs` 是成功提取 FVD 特征的视频数；`coverage.emotion.common_valid_frames / aligned_frames` 是表情共同有效帧比例；`coverage.delta_resp.valid_steps / possible_steps` 是可用运动步数比例，另有可测阶段时长。不存在的字段可能是该指标未被请求；`null` 表示请求后不可测，不能当成 0。

`complete` 表示所请求指标具备可汇总结果，不表示每一帧都检出了人脸。漏脸较多的实验仍需检查覆盖率。逐样本诊断看 `pairs.jsonl` 中的 `errors`、`media.trimmed_gt/trimmed_generated`、`face_detection`、`emotion`、`response.phases`。输入原路径和哈希也保存在每条记录中。

## 常见问题

| 现象 | 原因与处理 |
|---|---|
| `No module named lpips / pytorch_fid` | 本机命令漏了 `PYTHONPATH`，或新环境尚未安装相应依赖 |
| 找不到权重、资产键缺失 | 检查 `--assets` 及各绝对路径；不能用普通 torchvision Inception 权重代替 FID 权重 |
| `FPS mismatch`／时间戳不均匀 | 两侧不是相同恒定帧率；先核对生成、裁剪和转码阶段，不能靠评测时盲目改速纠正 |
| `Spatial sizes differ`／ROI 越界 | 使用同语义、同尺寸的 listener 视频，或修正固定 ROI；本入口不自动拉伸两侧来掩盖尺寸错误 |
| `Short clip` | 共同长度不足 FVD 要求；如果必须修改采样协议，所有对比组统一修改并重新评测 |
| `Fréchet distance requires at least two samples` | 特征样本不足；FVD 至少两段成功片段。排查时可先去掉 fid/fvd |
| `Supply listening_intervals` | 普通对话清单补倾听阶段；明确为 listen-only 才使用全段倾听开关 |
| 表情／ΔResp 为 null | 看共同有效人脸与阶段步数；多脸、漏脸、单帧阶段都可能不可测，不能填 neutral 或 0 |
| 进程退出码 2 | 部分指标不可测；先读 summary 与逐段 errors。`--allow-partial` 只用于显式接受部分结果的流程 |
| `FileExistsError` | 输出目录已有评测文件；换新目录，保留历史版本与哈希 |
| FVD 与旧表格不同 | 检查帧数、stride、样本集合、FPS、空间预处理和权重；不能仅凭都叫 FVD 就直接比较 |

## 计算口径

| 指标 | 实现及汇总方式 | 解释 |
|---|---|---|
| PSNR ↑ | RGB uint8，峰值 255；逐帧 PSNR 后对所有有效帧平均 | 像素重建误差；自对比为 `inf`，JSON 保存字符串 `"inf"`，不伪造 100 分。另存每段 MSE、完全相同帧数 |
| SSIM ↑ | RGB 通道，11×11 高斯窗、σ=1.5、population covariance、data_range=255；逐帧平均 | 结构相似性；图像两边至少 11 像素 |
| LPIPS ↓ | 官方 LPIPS 0.1、AlexNet，预训练 backbone＋学习的距离权重；RGB [-1,1]，逐帧平均 | 感知距离；不会以随机 backbone 替代缺失权重 |
| FID ↓ | pytorch-fid 的 TF-compatible Inception pool3，2048 维；样本协方差 N−1 | 默认每段等距最多 32 帧，在整个数据集聚合；不计算“每条视频的 FID” |
| FVD ↓ | 沿用现有 StyleGAN-V/TFHub I3D TorchScript 的输入/特征接口，400 维，population covariance N | 默认每段中间 32 帧、stride=1、RGB uint8、224×224，一段一个特征；数据集层面统计 |
| Emo-Acc ↑ | DeepFace 七类 argmax，在 GT 和生成视频共同检出人脸的帧上做全局 micro 一致率 | `[0,1]`；不是人工情绪真值的准确率，也不是 backchannel 检测率 |
| Emo-Div | 同一共同有效帧上，每个“视频＋身份”的不同表情类别数；再对视频身份平均 | **同时报告 generated 和 GT，不设越大越好**；范围 1–7，零有效帧为不可测 |
| ΔResp ↓ | InsightFace 106 点中的 22 个眼／眉点；逐倾听阶段计算平均相邻帧位移，再算生成与 GT 的绝对差，最后按阶段时长加权 | 单位为归一化 224 坐标中的像素／帧；只衡量局部运动幅度差，不等价于点头次数、语义或轨迹匹配 |

FID/FVD 至少各需两个特征样本。小样本虽能计算，但分数估计不稳定；不要把两段自检结果当模型排行榜。所有比较组必须使用同一清单、同一帧数／stride／FPS、同一裁框与权重。新入口默认短视频 FVD 协议**不同于旧脚本的 256 帧／stride=3 协议**；不能混用分数。可显式指定 `--fvd-frames 256 --fvd-stride 3`，短片不足时记录失败，不重复帧补齐。

## AnyTalker 来源与适配边界

依据本地论文 `audio-driven-v5/related/11577_AnyTalker_Data_Efficient(1).pdf`：正文 §4.2 式 (4) 定义 ΔResp，附录 E 式 (8)–(9) 定义 Emo-Acc/Div。参考 [AnyTalker benchmark](https://github.com/HKUST-C4G/AnyTalker/tree/main/benchmark)，本地仓库版本 `e5490b60dc6add439ff5ecec5bd7559606a78d53`。

官方 `benchmark/calculate_interactivity.py` 提供单段、单身份的眼部运动幅度计算，并非完整的成对 ΔResp；仓库没有公开完整 DeepFace 表情评测入口。本实现补齐论文定义的配对与数据集汇总，**不声称完全复现 Table 1 数值**。

眼眉索引（0-based）：`43,48,49,50,51,101,102,103,104,105,35,36,37,39,41,42,89,90,91,93,95,96`。

空间归一化参考其实现：GT 与生成视频共同取所有有效人脸框的最大边长，至少 224 px；固定正方形边长，框中心 EMA α=0.7，再映射到 224×224。保留其移动裁框思想，**不复制“过大位移替换成静止点”处理**；漏脸后重置平滑，运动不跨缺失帧连接。适用于已裁好的单 listener 视频，多身份要分别提供固定 ROI 后独立评测。

对倾听阶段 i：

```text
motion_i = 22 个眼眉点在有效相邻帧间的欧氏位移平均
gap_i = abs(motion_generated_i - motion_GT_i)
DeltaResp = sum(duration_i * gap_i) / sum(duration_i)
```

先逐阶段取绝对差；不能先平均 motion 再求差，否则运动过强与过弱会抵消。两边只使用共同有效的相邻帧，绝不跨阶段或跨漏脸连接；单帧阶段不可测。输出每阶段 valid_steps / possible_steps 与可测时长。部分漏脸仍可测，但必须连覆盖率一起报告；完全不可测不填 0。

表情使用 InsightFace 检脸后的 BGR 人脸裁图，先除以 255 归一化到 [0,1]，再经 DeepFace 官方浮点缩放与等比补边到 224，EmotionClient 再灰度缩至 48×48。未启用旋转对齐。七类为 angry / disgust / fear / happy / sad / surprise / neutral。只接受单脸检测，漏脸或多脸帧不强行补 neutral。Emo-Acc/Div 默认计算全部配对帧，不只计算倾听区间；ΔResp 只计算指定倾听区间。

2026-11-02 接入 ResponseNet 时新增可选的 `--face-padding`，以缓解紧裁剪大脸的检测漏检。默认 0 保持原有口径；非零值固定应用于两侧所有帧，仅影响检测与关键点输入，坐标会还原，表情仍裁原图。此参数计入 contract；不同配置不能混合比较。该处理不保证所有帧都能检脸，覆盖率仍需检查。

## 验收

```bash
python -m unittest evaluation.video.paired_metrics.test_core -v
# 含官方 pytorch-fid 数值对照的扩展测试：
python -m unittest evaluation.video.paired_metrics.test_core evaluation.video.paired_metrics.test_reference -v
```

测试覆盖解析误差、相同／扰动画面、Fréchet 协方差口径、无效样本、共同人脸支持、表情多样性、阶段加权、不跨漏脸连接、ROI/帧率及严格 JSON。真实模型自检记录见 [validation/README.md](validation/README.md)。
