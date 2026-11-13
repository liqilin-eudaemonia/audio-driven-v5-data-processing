# 两个项目目录的 GitHub 备份

快照时间（UTC）：2026-11-13T15:08:56.211907+00:00

按用户确认的范围，保存代码、配置、文档及不超过 100 MB 的数据清单。两个原目录分别保存在同名子目录下。共备份 **143,775 个文件**，未压缩原始内容 **42,558,354,814 字节**。

这是上述范围的快照，不是包含全部训练数据的整盘备份。音视频、模型权重、训练产物、虚拟环境、缓存、逐样本中间结果和超过 100 MB 的清单保留在原机器。原目录没有被修改。原有仓库的 main 分支保留。

## 浏览与恢复

普通源码、配置和文档可以直接在两个项目目录中浏览。较大的清单采用 gzip 压缩，存放在 `backup_metadata/packed/`，原路径和有序分片在 `backup_metadata/included.jsonl.gz` 中。压缩文件不是 Git LFS 指针，数据实际存储在本分支中。

克隆本分支后，用 Python 3 校验全部已备份内容：

```bash
git clone --single-branch --branch backup-20261113 git@github.com:liqilin-eudaemonia/audio-driven-v5-data-processing.git audio-project-backup
cd audio-project-backup
python3 restore_backup.py --verify-only
```

恢复到一个新的空目录（自动重组分片、解压并逐文件核对 SHA-256，保留执行权限和修改时间）：

```bash
python3 restore_backup.py --destination /path/to/new-empty-directory
```

## 未上传明细

`backup_metadata/excluded.*.jsonl.gz` 逐行记录未上传文件的相对路径、已知大小、类型、修改时间及原因；符号链接另含原目标。`scope: entire_subtree` 表示该目录及全部子文件均未上传，这类数据和缓存目录无需逐文件遍历。`size_known: false` 表示未读取该排除文件的元数据，大小未知，记录中的 0 不能视为实际大小。目录大小不代表整个目录的容量；统计容量是已清点文件的下限，不能作为完整磁盘使用量。

`backup_metadata/summary.json` 汇总范围、数量与排除原因；`selection_policy.py` 保存筛选规则。超过 100 MB 的清单原因是 `oversized_manifest_over_100MB_user_approved`。凭据文件和检测到真实格式密钥的文件也会登记排除。符号链接的外部目标没有复制，恢复脚本不会创建这些外链。

要恢复未上传的数据，请依据明细中的路径，到上述两个原目录或其外部链接目标提取。仅克隆本分支无法恢复这些数据。

已备份文件的原始 SHA-256、大小、权限、修改时间、编码和存储位置见 `included.jsonl.gz`。源目录可能仍有其他任务更新，本备份是各文件复制时的已校验版本，并非停止生产任务后的全局原子快照。
