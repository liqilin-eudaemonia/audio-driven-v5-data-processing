# P2 当前训练代码备份（保留 feature decoder）

当前包含七类弱背景监督、确定性计算与 FP64 梯度累加，feature decoder 与 feature loss 保持启用；尚未实施 AV-only 简化。

`files/` 按项目相对路径保存 v5 P2 源码、复用的 v4 源码、配置、CLI、测试、文档及选定验收记录；`manifest.json` 记录逐文件 SHA256，复制后再次核对原文件与备份完全一致。

恢复时按 manifest 的 path 将所需文件恢复到 `/private/liqi_workspace/project_code/` 下对应项目；恢复前另存届时工作版本，并检查与运行任务的兼容性。

这是共享工作区中的独立源码快照；预训练权重、数据、特征缓存及训练 checkpoint 沿用原目录，没有复制进本备份。
