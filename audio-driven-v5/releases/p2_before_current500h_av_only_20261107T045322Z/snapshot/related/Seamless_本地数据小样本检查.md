# Seamless 本地数据小样本检查

日期：2026-10-21。数据根目录：`/public/0_DATA/2_Avatar/seamless_interaction/`。

本次遵照“各抽几条、不看完整数据集”的要求，只查看顶层/少量下层目录、README、三个根目录 CSV 的表头及前三条记录，以及 improvised、naturalistic 各一个归档开头的少量样本。没有全库遍历、全量解包、全量 ASR 或统计覆盖率。

## 1. 抽样范围

归档为 `improvised/dev/0000/0000.tar` 与 `naturalistic/dev/0000/0000.tar`。各查看开头四个样本的 JSON，各对前两个样本进一步检查 MP4、WAV、NPZ；另仅查看各归档前 40 个成员名，以了解文件组织。成员顺序并不保证相邻两人是一个 pair。

| 子集 | 详细检查的样本 ID | 视频/独立 WAV 时长 | 视频帧数 |
|---|---|---|---|
| improvised | V00_S0809_I00000309_P0947 | 166 秒 | 4980 |
| improvised | V00_S1061_I00000168_P0731 | 228 秒 | 6840 |
| naturalistic | V00_S0627_I00000381_P0779 | 114 秒 | 3420 |
| naturalistic | V00_S0691_I00000480_P0840 | 258 秒 | 7740 |

视频已用 ffprobe 检查，并查看各样本 10 秒处的一帧；未完成双方配对后的听音核验、口型同步或串音质量验证。以下结论不外推为全库保证。

## 2. 已确认的数据形式

本地样本以 WebDataset TAR 保存。同一人物样本有同名 `.json`、`.mp4`、`.npz`、`.wav`。

- 四条视频均为 1080×1920、30 FPS、单人全身竖屏画面，包含 AAC 双声道音轨。不是等待从左右拼接画面拆开的双人视频。
- 独立 WAV 均为 48 kHz、单声道、32 位浮点格式。README 的一般描述是 16-bit，但本次实际文件为 FLOAT，应以文件探测为准。
- 四条样本的独立 WAV 时长与视频时长一致，NPZ 的逐帧特征长度与视频帧数一致。这只验证长度，不证明精确同步或不存在串音。
- README 将独立 audio 描述为 speaker-bleed denoised audio；应优先评估独立 WAV，不默认 MP4 内音轨与其等价，也不在验证前对所有数据重复做音源分离。

样本 ID 的结构为 `V<vendor>_S<session>_I<interaction>_P<participant>`。后续应按相同 V/S/I、不同 P 关联双方，实际检查人数、文件完整性和时间轴。此次未为寻找另一方而扫描其他全部归档，因此尚未核验具体完整 pair。

## 3. 已经带有转写和 VAD

检查的 JSON 包含：

```text
id
metadata:transcript
  └─ start, end, transcript
     words[]: word, start, end, score
metadata:vad
  └─ start, end
```

转写有句段级及词级时间戳；本次实际 VAD 为起止时间区间列表，不能直接按 README 的“100 Hz”描述当作稠密帧概率数组读取。

已看到 `Yeah.`、`Okay.`、`Exactly.`、`Wow.`、`Uh-huh.` 等短词句，但这些只能作为候选，不是已确认的 backchannel。要结合另一方发言和话轮判断，也要验证音频确实存在对应声音。

现有转写与 VAD 存在局部不一致。例如：

| 样本 | 转写区间（秒） | 转写 | 与已有 VAD 的交集 |
|---|---|---|---|
| V00_S0691_I00000480_P0840 | 0.760–1.200 | Wow. | 无 |
| V00_S0691_I00000480_P0840 | 183.929–184.049 | Uh-huh. | 无 |
| V00_S0809_I00000309_P0947 | 19.078–19.258 | Yeah. | 无 |
| V00_S1061_I00000168_P0731 | 116.218–116.619 | Uh-huh. | 无 |

这是两份标注不一致的证据，不是听音确认的 VAD 漏检证据：可能涉及 VAD 漏检、转写误报或时间误差。候选阶段不宜只取两者交集；可合并候选区间，并通过音频复核处理不一致处。

因此不建议把“对所有数据重新 ASR”作为第一步。先利用并评估现有转写/VAD，对短反馈、缺失、冲突区间做针对性补识别，另补笑声等非语言发声检测。

## 4. 情绪特征与人工标注需要区分

四条详细样本的 NPZ 均带有：

| 字段 | 形状 |
|---|---|
| `boxes_and_keypoints:box` | `[T,4]` |
| `boxes_and_keypoints:is_valid_box` | `[T]` |
| `boxes_and_keypoints:keypoints` | `[T,133,3]` |
| `movement:emotion_scores` | `[T,8]` |
| `movement:emotion_valence` | `[T,1]` |
| `movement:emotion_arousal` | `[T,1]` |
| `movement:is_valid` | `[T,1]` |

另有 FAU、头部姿态、expression、gaze、SMPL-H 等字段。原始 box 的范围及视频画面说明不能直接把它当成目标脸部裁剪框；可以评估已有关键点对脸部定位的帮助，但需要核对关键点索引定义。

README 将 movement 特征描述为 Imitator 模型输出，因此这些 emotion 字段属于自动估计。实读 `emotion_scores` 有负值和大于 1 的值，不能直接当概率；8 个类别的顺序和分数语义仍需核对，也不能直接替换 GLARE FER 的输入。

用户已确认，其所指的情绪标注就是上述 NPZ 中的 `emotion_scores/valence/arousal`。这次在 improvised 与 naturalistic 的详细抽样中均发现这些字段；没有检查它们的全库覆盖率。

README 还列出人工标注类型：`1P-IS`、`1P-R`、`3P-IS`、`3P-R`、`3P-V`，分别涉及本人/第三方内部状态、理由和视觉行为描述。这不是用户此次指定的情绪特征，本次不继续查找；当前抽样未定位到这些人工标注的实际样本，不能据此推断本地没有。

## 5. 对后续处理方案的修订

原有的两个大步骤仍成立，但可利用已有数据减少重复工作：

1. **构建倾听 pair：** 依据 V/S/I/P 建立双方映射，使用逐人视频和独立 WAV；验证同步，确定主话轮和倾听区间，裁剪稳定头部画面。
2. **构建 listener 声音标签：** 先以现有转写和 VAD 提议候选，补查短发声及不一致区间，补充笑声等非语言事件识别，再结合双方上下文区分 backchannel 与正式回应。

保留倾听阶段的静音、短反馈和合理重叠发声，不按 listener VAD 为零来定义倾听。已有自动视觉特征可先评估复用；如需 GLARE 同口径反应标签，仍需其对应检测流程或经过验证的特征适配。
