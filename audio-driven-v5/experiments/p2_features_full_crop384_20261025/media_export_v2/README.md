# 媒体导出提速与等价验证（2026-10-25）

本目录是当前全量特征流水线的独立版本媒体导出器；`src` 链接原工程。原 `code_manifest.json` 的全部文件保持原哈希，56 个常驻 GPU 进程不用重启。`config.json.project` 只由 `feature_pipeline.export_media()` 用来定位导出脚本；其他代码仍由原路径导入。配置另存 `media_export_source_project`、`media_export_sha256`，导出前核验脚本 SHA-256；`config.before.json` 保留原配置。

只转换最终 25 Hz 取帧真正使用的源帧；仍记录所有源帧时间，取帧索引与原 bisect 规则完全一致。对于偶数尺寸、逐行 yuv420p，先复制带 16 像素色度对齐边界的 YUV 区域，再转 RGB，按原裁框和 PIL bicubic 得到 512×512。其他格式、奇数尺寸、交错帧保留原转换。每个片段每种图像布局首次转换都与原转换逐像素比较；若不一致，该片段该布局全部退回原实现。裁框、帧率、时间、音轨和标签没有改动。

原 4.79 秒的 4K 片段导出用时 52.579 秒；优化含首帧检查用时 12.651 秒。视频、首帧、speaker/listener WAV 的 SHA-256 全部相同，所有源时间与输出取帧索引相同。这是单片段实测，整体吞吐以运行状态为准。

- `pixel_validation_final.json`：96 组随机像素、尺寸、边界、格式与色彩元数据检查全部相同；28 组帧率与相位取帧检查。
- `real_clip_validation_final.json`：真实 120 帧片段四文件哈希一致。
- `code_manifest.json`、`deployment.json`：版本、部署时间与检查信息。
- `baseline_status.json`：部署时完整运行快照。
- `validate.py`：测试入口；其中整段输出目录须为空，不覆盖验证产物。

实验阶段的 `pixel_validation.json` 包含奇数尺寸的失败用例，已用于添加回退，未部署那个版本。原型测试产物保留用于追溯。

恢复原导出路径时仅将运行配置的 `project` 改回 `media_export_source_project`；无需改原代码或已有媒体。不要直接编辑已部署脚本，否则哈希保护会阻止继续导出。
