# GLARE 四项反应评测数据接口

完整命令与口径见 [实现与运行](../../related/P2_GLARE四项反应评测实现与运行.md)。示例路径是占位符，正式数据由用户提供后再填写。

- 已有生成视频：使用 `pairs.example.json`。路径相对清单所在目录解析；每个 seed 必须包含完全相同的 sample ID 集合，同一 ID 的 GT 视频必须相同。
- 从训练检查点生成：使用原生 [P2 observations](../p2_listen/README.md)，加上 `targets.example.json`。也接受原生 P2 targets 的 `target_video` 字段。observations 只能含 speaker 音频、listener 首帧和时间/几何等输入元数据；必须是与训练身份隔离的 dev/test。
- 所有视频均为零起点、25 FPS、等尺寸、严格对齐的完整目标窗口；不在评测时自动裁剪、缩放或修正时间轴。
- 自动检测时无需 CSV。已有独立视觉检测结果可给每行添加 `gt_csv`、`generated_csv`，并在清单顶层填写非空 `annotation_provenance`。训练 `probe.pt` 或 Event head 输出不可作为生成视频检测结果。
- CSV 列：`frame_index,smiling,laughing,frowning,surprised,nodding,head_shaking`。逐帧从 0 连续，每帧最多一列非零，分数有限且在 [0,1]，已经过检测器后处理。
- 生成失败必须保留条目，省略 `generated_video`，添加非空 `generation_error`；不能删除失败样本后报告原定集合的指标。

单独一份 E1 结果只描述该模型的生成行为。判断 Event loss 是否带来改善，仍需同一输入、同一生成 seed、同一检测协议的 E0/E1 对照。
