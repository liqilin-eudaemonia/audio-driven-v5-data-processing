# Listener 原始预裁框 ≥384×384 原始打标

用户于 2026-10-24 将本轮范围收紧为原始 listener 预裁框边长 ≥384 像素；以保证质量为前提尽快完成，无固定时刻截止承诺。当前运行中，未完成全量。

## 范围与输入

- 722,706 段 / 1,224.604152 小时：primary 634,847，onset_reserve 87,859。筛选未四舍五入的 `crop_proposal.size >=384` 且裁框有效；不按缩放后的 512 像素筛选。
- 每分支预计 3,919,483 个整段／滑窗请求。保留原切分、音质准入、起始活动标记。仅 train。
- `crop_proposals.listener` 原图 x/y/size 按 half-up 整数裁剪，bicubic 512×512，8 FPS；独立 normalized_audio.listener，不用混音。
- Qwen3-VL-8B 视觉、Qwen3-Omni Instruct 纯音频、Qwen3-ASR＋ForcedAligner。整段＋3 秒窗/1 秒步长。提示词、4096 输出上限与模型不因提速而减少。

## 运行与复用

三机 24 卡：21 视觉＋2 音频＋1 ASR。音频/ASR 结束后由监控交接给视觉。CUDA Graph、显存配额 0.85；视觉 batch64，音频和 ASR batch128。3 卡 batch96 试验尚未显示提升（实测约 1.78 对 1.86 请求/秒/卡，非同样本配对比较），已回到64，避免无依据扩大。记录见 batch_trial.json、batch_measurement_latest.json；配置见 control.json、allocations.json。

[状态](status.json)、[吞吐历史](throughput_history.jsonl)、[资源与进程](nodes/)、[筛选索引](queue/index.json)、[复用清单](reuse_report.json)、[代码快照](code/)、[代码哈希](code_manifest.json)。页面：远端 8876/listen_benchmark_review/full_raw/，本机需转发远端 8876。

旧运行 listen_full_raw_20261024T123500Z 已冻结。按 sample_id 和 source_line_sha256 校验后逐字节复用原始结果：视觉 30,331、音频 83,318、ASR 119,919 个已记录请求，原 run_id 保留；历史失败也保留，并不算有效标注。ASR 历史对齐修复位于独立目录。

发现两个大 MP4 的 FFmpeg subfile 开启失败；同一 TAR 成员字节的有界 seek 读取可以正常解码。已补充备用读取路径，不重编码、不换源、不改 PTS/裁框，旧失败记录不覆盖，恢复时补跑对应请求。reader_fix.json 和 reader_equivalence_check.json 记录验证。

## 原始输出与口径

`raw/{visual,audio,asr}/results/*.jsonl` 追加保存模型原文、prompt、局部起止时间、窗口偏移、输入证据及来源。`raw/*/state` 是可恢复计数，`raw/*/locks` 防止多卡重复写同一分片。保存 raw 后才解析；结构不合格、输入失败、推理失败分别记录。ASR 原文与对齐结果分开，修复另存。

请求数包含滑窗，不是 pair 数；schema_pass 不是人工正确率。暂无事件合并、类别映射或训练准入。汇总最终结果前需单列未成功项目，不能将失败算作有效覆盖。

## 恢复与停止

每节点运行 code/scripts/supervise_listen_full_raw.py --run <本目录绝对路径> --node node17|node12|node4。启动时验证 GPU 空闲；已有本运行进程须使用 --adopt-existing，禁止重复启动。监控入口 code/scripts/monitor_listen_full_raw.py --run <本目录> --preview <项目>/data_previews/listen_benchmark_review/full_raw。

STOP 文件让 worker 完成本 batch 后退出；STOP_MONITOR 停止监控。恢复前确认原进程退出后删除相应停止标记；不要删除原始结果。GPU 归属由 supervisor 校验，仅管理本运行的进程组。

10 项原始打标/子集/恢复/成员读取测试通过；实测大文件解码及普通输入像素和 PTS 等价检查另存。旧范围的时长审计保留来源，当前范围以 queue/index.json 为准。
