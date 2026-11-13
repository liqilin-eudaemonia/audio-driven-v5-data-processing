# 本机未完成 GPU 标签转由 node17、node8 补齐

2026-10-30。用户仅恢复另外两机任务，并要求其覆盖剩余完整待打标队列。本机 GPU 已被另一个任务使用，`STOP_SUPERVISION` 保持；本机 CPU 点头与画质辅助任务继续运行。

## 交接结果

本机原 GPU 分片304,045段，其中已提交来源的62,488段保留原位置。剩余6,647个来源包含241,557段；其中262段已有部分来源标签，通过只读复用链接保留，实际还需推理241,295段。

| 接收机器 | 来源数 | 来源内片段总数（含复用项） |
|---|---:|---:|
| node17 | 3,344 | 124,093 |
| node8 | 3,303 | 117,464 |

任务追加到两机现有 journal，现有 GPU worker 自动领取，无需重启模型。两条部分完成来源被放入可恢复领取队列优先验证，见 [canaries.json](canaries.json)。已完成源标签不迁移，部分完成标签仍保存在本机原目录，接收机以符号链接复用；冻结的GPU加速参数不变。

## GPU 与点头分支分别归属

两机现有 journal 同时供视觉、点头分支读取。因此对交接来源，接收机的 `nod/completed` 存放 `status=delegated_branch_not_measurement`、`clips=0` 的**跳过凭证**，指向本机点头工作目录。这不是完成测量或质量合格证据，不计入 completed_clips，避免接收机重复计算点头。

本机 VLM 游标已推进到旧 journal 末尾，释放的GPU旧claims已清空；CPU点头游标、进程和已有结果未修改。不要直接恢复本机旧GPU队列，它已没有待领取任务。

上级 `nodes.json` 及画质根目录的 `annotation_nodes.json` 新增 `branch_ownership_overrides`，为每个交接来源明确指定 GPU标签和点头标签所在工作目录。`branch_scope_clips` 用于分支范围统计：

| 节点 | GPU 标签范围 | 点头标签范围 |
|---|---:|---:|
| node17 | 612,374 | 488,281 |
| node8 | 476,943 | 359,479 |
| node12（GPU已停） | 62,488（已完成） | 304,045 |

两条分支各自总量均为1,151,805段。逻辑来源总数仍60,869；由于同一来源的不同分支在不同机器运行，不能再把各节点 journal 行数直接相加当作来源数。最终汇总由 `scripts/listen_j_quality_parallel.py` 按分支归属读取结果，并在 `retained_annotation_index.jsonl` 保存 `annotation_work_by_branch`。

## 验证与恢复

- [提交记录](committed.json)、[完整范围校验](coverage_audit.json)、[复用标签与原文件哈希](reused_results.json)。
- `plan.json`、`source_vlm_state_before.json`、原 journal 和追加凭证支持本次交接恢复，不能覆盖回已经继续运行的旧游标。
- 操作入口 `scripts/delegate_listen_j_local_vlm.py`；正常完成后无需再次执行。
- 3项交接测试及5项最终汇总测试通过：验证已有标签跳过、CPU点头不重复执行、分支结果正确关联、禁止未声明的重复归属。
- 两机画质 controller 已平滑重载最终汇总代码，测量 worker 和 GPU模型继续运行。

统计进度请使用队列状态和分支归属，**不能直接数 `nod/completed` 文件数**，其中包含明确标注的分支跳过凭证。
