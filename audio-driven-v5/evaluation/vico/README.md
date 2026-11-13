# ViCo：论文测试集

**最新Var口径：用户指定第二种定义。** GT及两版输出统一采用“各系数跨全部评分帧计算方差，再对系数平均”，`ddof=0`；43条／16,238个共同有效帧，其他五项指标保持。GT Exp／Pose为0.904523／0.012038，SFT2752为0.854602／0.010620。[新结果与原公式对照](../../experiments/p2_vico_dim_var_per_coefficient_20261108_v2/README.md) · [固定配置](../../configs/p2_vico_dim_variance_per_coefficient_v1.json)。这是明确选择的定义，尚非作者公式确认；原封存结果不覆盖。

**配置推断方向（2026-11-08）：** 用户无法联系作者，后续由论文／公开代码及GT诊断重建设置；不将作者回复作为前置条件。[推荐配置及Var／SID实测诊断](../../experiments/p2_vico_dim_inferred_protocol_20261108/README.md)。现有已封存成绩保持，推断不冒称作者确认。

两版×三套配置的敏感性评分已完成，推荐保留正式test43与DIM公开指标为主；另报时间偏移、范围投影及Var／SID定义差异。[实测结果](../../experiments/p2_vico_dim_inferred_protocol_20261108/sensitivity_v1/RESULTS.md) · [固定配置](../../configs/p2_vico_dim_inferred_v1.json)。现有单次输出不标成10次择优结果。

**10次择优补充已排队：** SFT2752固定137–146十种子，新增387条，后台仅node11 GPU7等待余量后自动生成／提取／评分；CPU预检和4项测试通过，最终成绩尚未产生。[实时进度与规则](../../experiments/p2_vico_dim_best10_20261108/README.md)。

**2026-11-08最新状态：DIM/EMOCA工程已完成并复验。** 固定43条，联合8000/base-Qwen与SFT2752两版均已用指定EMOCA及DIM公开函数评分，172组特征、548个数值复算通过。[完整实测](../../experiments/p2_vico_literature_protocol_20261108/RESULTS.md) · [当前验收](../../experiments/p2_vico_dim_handoff_20261108/README.md) · [需要double-check的清单](../../experiments/p2_vico_dim_handoff_20261108/DOUBLE_CHECK.md)。GT统计尚未复现论文值，严格原表等价性仍未确认；以下既有BFM运动表与本次EMOCA表分别保留，不能按同名FD混排。

**三种方法统一页：** [联合模型8000＋DyStream同步推理视频，Learning2Listen四套动作评分／曲线](http://localhost:8876/listen_benchmark_review/p2_vico43_compare_20261106/)。同一43条，四路同步拖动、原生512／评分256切换；L2L无写实renderer，未制造视频或图像分数。[统一指标CSV](../../experiments/p2_compare_vico43_20261106/three_method_review/all_methods_metrics.csv)。

**compare参考模型：** [DyStream43条视频＋Learning2Listen四模型动作曲线](http://localhost:8876/listen_benchmark_review/p2_vico43_compare_20261106/)；[完整指标与六项资源缺项说明](../../experiments/p2_compare_vico43_20261106/RESULTS.md)。

**全部43条生成可视化：** [speaker／GT／联合8000步同步播放](http://localhost:8876/listen_benchmark_review/p2_vico43_combined8000_20261106/)，支持原生512和评分256、音轨切换、逐条运动指标与规划文本；旧原画幅页增加了同一入口。

**运动指标补测已完成（2026-11-06）：** 同一联合8000步test43，129路／48,864帧3DMM提取及独立验收通过；FD、FID_delta_fm、SND、V-D覆盖43条，RTLCC42条、RWTLCC41条，短片段N/A。GPU7已释放。用户确认原始系数、±2秒lag、4秒窗口／1秒步长、逐视频等权；属于公开定义复现。[结果表](../../experiments/p2_vico_motion8000_20261106/RESULTS.md) · [完整协议](../../experiments/p2_vico_motion8000_20261106/README.md) · [最终验收](../../experiments/p2_vico_motion8000_20261106/FINAL_AUDIT.json)。

这是论文发布的 ViCo，不是挑战赛那 30 条只有输入、没有听者真值的提交包。标注表里的 `data_split` 分成训练集、测试集和域外测试。听者真值是裁剪后的听者视频。说话人视频和音频是条件。

| 划分 | 条数 | 用途 |
|---|---:|---|
| test | 43 | 论文里的域内测试，身份在训练集中出现过 |
| ood | 42 | 域外测试，身份不在训练集中 |
| train | 398 | 训练用，不拿来报测试分数 |

同一段时间里偶尔有两个听者。`uuid` 用听者文件名，所以 `yTInFHb_o1Q_000506_000516.0` 和 `yTInFHb_o1Q_000506_000516.2` 是两条，不会并成一条。

算分读的是 256×256 裁剪。原画幅在压缩包 `videos/original/` 里，955 个都在，审核页用那一份看画面，测评不读。

## 2026-11-06：联合模型 8000 步画质与身份五项评测已完成

**已完成：43/43条生成、本轮画质与身份五项官方脚本评分和独立最终验收通过。** 评分视频独立解码共16,288帧；GPU7已释放。生成／导出阶段平均GPU利用率93.30%，中位数100%。

| PSNR ↑ | SSIM ↑ | FID ↓ | CSIM ↑ | CPBD ↑ |
|---:|---:|---:|---:|---:|
| 19.6683 | 0.66245 | 30.1163 | 0.59684 | 0.13913 |

[完整结果](../../experiments/p2_vico256_combined8000_20261106/RESULTS.json) · [CSV](../../experiments/p2_vico256_combined8000_20261106/summary.csv) · [最终验收](../../experiments/p2_vico256_combined8000_20261106/FINAL_AUDIT.json) · [GPU资源记录](../../experiments/p2_vico256_combined8000_20261106/GPU_USAGE.json)

**评测范围说明：** 上述五项仅覆盖画质和身份，不代表完成ViCo论文及后续工作的全部运动指标。ViCo原论文还报告3DMM系数的FD（L1 Feature Distance）。CustomListener（CVPR2024）表1使用FD、RTLCC、RWTLCC、FID_Δfm、SND和V-D；这份五项画质结果不含上述运动／同步指标；独立运动补测已完成，见本页最上方。图像FID不能替代FID_Δfm。严格对比需匹配3DMM提取器、系数基与归一化、帧率／lag／窗口及汇总口径。原始结果和验收摘要保持不变；[范围记录](../../experiments/p2_vico256_combined8000_20261106/SCOPE_CLARIFICATION.json)。

使用 `AV＋0.05 Event＋0.025 video Feature＋0.025 audio Feature＋speaker Whisper` 第8000步，原始未SFT Qwen规划器。正式 `test` 43条，不含ood或train；只使用speaker音频与listener首帧，无未来listener画面、目标音轨或GT描述输入。

- 用户确认：原生 **512×512、25fps** 生成；float RGB带抗混叠bicubic缩小到 **256×256**，按最近时间戳重采样为 **30fps**，与原始ViCo GT评分。保留512/25原始生成、256/30评分副本和逐帧映射，不改变播放速度。
- node8 `172.29.104.237:22`，仅物理GPU7。43条共542.933秒；原speaker WAV尾端最多裁齐/补零0.0491秒，完整记录。
- 本轮画质与身份五项：PSNR、SSIM、图像全局FID、CPBD、ArcFace CSIM。原官方评分脚本保持原样；CPU CPBD与生成重叠，全部结果按官方帧权重汇总。
- [运行状态](../../experiments/p2_vico256_combined8000_20261106/status.json) · [冻结协议](../../experiments/p2_vico256_combined8000_20261106/protocol.json) · [输入预检](../../experiments/p2_vico256_combined8000_20261106/INPUT_PREFLIGHT.json)。控制器已写 `RESULTS.json`、`COMPLETE.json`；独立`FINAL_AUDIT.json`通过，无当前`FAILURE.json`。
- 入口：`scripts/p2_vico256_combined_eval.py`；CPU评分兼容包装：`scripts/p2_vico_official_metrics.py`。三条31–35秒样本使用独立Observation时长适配；共享训练代码和权重不改。

## 读取

机器可读清单：[index.json](index.json)。每条给出听者视频、说话人视频和音频的绝对路径。

| 内容 | 位置 |
|---|---|
| 总表 | `/public/0_DATA/2_Avatar/ViCo/dataset/metadata.csv` |
| 测试集标注 | `/public/0_DATA/2_Avatar/ViCo/dataset/by_split/test/test.csv` |
| 域外测试标注 | `/public/0_DATA/2_Avatar/ViCo/dataset/by_split/ood/ood.csv` |
| 测试集画面 | `/public/0_DATA/2_Avatar/ViCo/dataset/by_split/test/videos/` |
| 域外测试画面 | `/public/0_DATA/2_Avatar/ViCo/dataset/by_split/ood/videos/` |

画面文件名是 `{uuid}.listener.mp4` 和 `{uuid}.speaker.mp4`。音频是同目录上一级 `audios/{uuid}.wav`。

## 权重

| 用途 | 文件 |
|---|---|
| 身份相似度，ArcFace R100 | `/public/0_DATA/2_Avatar/ViCo/ms1mv3_arcface_r100_fp16/backbone.pth` |
| 三维重建 | `/public/0_DATA/2_Avatar/ViCo/epoch_20.pth` |
| 表情基 | `/public/0_DATA/2_Avatar/ViCo/Exp_Pca.bin` |
| Basel Face Model 2009 | `/public/0_DATA/2_Avatar/ViCo/BaselFaceModel.tgz`，里面是 `PublicMM1/01_MorphableModel.mat` |
| PIRender | `/public/0_DATA/2_Avatar/ViCo/face.zip` |

唇形同步权重没有下。那一项只评说话人轨。

## 算分

脚本在 [vico_challenge_baseline/evaluations](../vico_challenge_baseline/evaluations)。生成结果目录里要有同名的 `{uuid}.listener.mp4`。本轮43条生成、逐条验收和画质／身份五项评分均已完成；结果见上方记录。

在 `evaluation/vico_challenge_baseline/evaluations` 下，把测试集换成域外测试时，路径里的 `test` 改成 `ood`：

```bash
python compute_base_metrics.py \
  --gt_video_folder /public/0_DATA/2_Avatar/ViCo/dataset/by_split/test/videos \
  --pd_video_folder <生成结果目录> \
  --anno_file /public/0_DATA/2_Avatar/ViCo/dataset/by_split/test/test.csv \
  --task listener

python compute_fid.py \
  --gt_video_folder /public/0_DATA/2_Avatar/ViCo/dataset/by_split/test/videos \
  --pd_video_folder <生成结果目录> \
  --anno_file /public/0_DATA/2_Avatar/ViCo/dataset/by_split/test/test.csv \
  --task listener

python compute_cpbd.py \
  --pd_video_folder <生成结果目录> \
  --anno_file /public/0_DATA/2_Avatar/ViCo/dataset/by_split/test/test.csv \
  --task listener

python compute_csim.py \
  --gt_video_folder /public/0_DATA/2_Avatar/ViCo/dataset/by_split/test/videos \
  --pd_video_folder <生成结果目录> \
  --anno_file /public/0_DATA/2_Avatar/ViCo/dataset/by_split/test/test.csv \
  --task listener \
  --weight /public/0_DATA/2_Avatar/ViCo/ms1mv3_arcface_r100_fp16/backbone.pth
```

`compute_csim.py` 在 `evaluations/arcface_torch/`。
