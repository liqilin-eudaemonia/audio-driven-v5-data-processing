# AV-only消融前源码备份

当前项目src/scripts/configs/tests/external_sources/docs、根文件、related说明以及当前/父训练配置和恢复元数据均已复制。当前正式运行绑定的270份源文件（包括外部v4/LTX依赖）逐个与run.json核验一致，快照及压缩包中每个文件SHA再次验证。

[完整清单](MANIFEST.json) · [备份验收](VERIFIED.json) · [压缩包](code_snapshot.tar.gz)。snapshot中保留原项目相对路径；外部依赖按external_absolute/private/...映射到原绝对路径。恢复时先停相关训练，再按MANIFEST.original恢复；不要向正在训练的共享源码覆盖。权重和大规模数据/缓存未复制，其完整检查点回执、配置和来源路径已保留。
