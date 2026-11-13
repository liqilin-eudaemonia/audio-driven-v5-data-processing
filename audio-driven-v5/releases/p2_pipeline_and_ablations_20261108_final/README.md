# 当前pipeline与消融代码备份（2026-11-08）

本地独立副本，逐文件SHA256与字节数核验通过；详见 [MANIFEST.json](MANIFEST.json)。修改前快照 `../p2_before_stage2_joint_20261108` 原样保留。

| 流程 | 原代码包（src/） | 原配置（experiments/ 下） |
|---|---|---|
| 完整 AV＋Whisper＋Feature＋LTX整段Event | p2_ltx_events 及清单内依赖 | p2_ltx_clip_train31_cadence_20261107/config.json |
| 仅AV | p2_current_av_only | p2_current500h_av_only31_20261107/config.json |
| AV＋Whisper | p2_current_av_whisper | p2_current500h_av_whisper31_20261107/config.json |
| AV＋Feature | p2_current_av_feature | p2_current500h_av_feature31_20261107/config.json |
| AV＋Event | p2_current_av_event_only | p2_current_component_ablations31_20261108/av_event/config.json |
| 去Whisper／去Feature／去Event | p2_current_component_ablation | p2_current_component_ablations31_20261108/{no_whisper,no_feature,no_event}/config.json |
| 第二阶段联合微调（未开训） | p2_stage2_joint | p2_stage2_joint_candidate_full6000_20261108/config.json（候选，未选定最终模型） |

以上代码的脚本入口、外部源码依赖、实验配置、数据索引约定、cluster、已有启动命令、测试及验收记录均包含在副本中。第二阶段另包含固定64条验证清单和规划监督适用清单。后续更新的两份双模型评测协调脚本也保存了当前版本。

## 文件映射与恢复

`MANIFEST.json` 的 `files` 以原始绝对路径为键，每项记录副本相对路径 `path`、`sha256` 和 `bytes`。副本在 `workspace/project_code/...`，保留 audio-driven-v4、audio-driven-v5 以及其他依赖项目之间的相对关系。

恢复时先在独立目录按照映射还原需要的源码和配置，逐文件核对SHA；确认实验输出、数据／模型路径和配置对应后，再使用该组原启动入口。不要直接覆盖运行中的代码，也不要因恢复而重新启用已经取消的训练队列。

这里是代码／配置／索引清单备份。模型权重、Adam/RNG检查点、原始媒体、AV／Whisper／teacher等大缓存保留原路径，通过配置及已有完整检查点标记定位，没有复制这些大文件。因此它不是整个训练环境与数据的离线镜像。

第二阶段代码验收包括27项测试、32条真实缓存回读、64条独立验证排除审计；尚未执行该组完整规模GPU训练或端到端生成评测。
