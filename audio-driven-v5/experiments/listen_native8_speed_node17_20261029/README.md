# node17 原生视频8fps加速测试

实测推荐和命令见 [交接报告](../../related/P2_原生视频8fps打标加速实测_20261029.md)。

已完成8组各50条测试，当前最快为 `shared_large/b256_t16384_c50_m45`。完整状态见 `RESULT_STATUS.json`；输入、输出和性能对比见 `shared_large/comparison.json`。

`code_tested/` 保存与运行协议哈希一致的实际测速核心代码；`code_current/` 保存当前代码（额外提供尚未完成GPU测速的CPU并行解码选项），`code_manifest.json` 记录差异范围与哈希。没有替换全量标注或启动训练。
