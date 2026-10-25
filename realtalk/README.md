# RealTalk listener pair 切分备份

**最新已确认切分规则（2026-10-25）：** [boundary_v2：代码、规则、137条长视频结果与恢复说明](boundary_v2/README.md)。用户已认可完整原视频的切分；保留正式接话前后和短回应，修复几何切点、黑屏尾帧、音频终点及缓存问题。旧版全量129,094条清单未替换；下方为v1备份说明。

版本：`realtalk-pairs-10s-max11-v1`，2026-10-23。此目录独立于仓库中的 V8 音质筛选基线。

## 当前规则与结果

有人说话时提取另一人的倾听候选；镜头、布局、角色变化处断开，边界前后分别继续提取。目标 10 秒、最长 11 秒，优先句末候选或声学停顿，必要时从句中截断。保留短尾段、不设最低时长，不因长区间存在问题而整体排除。

692 条视频，129,094 个候选 pair，共 110.7779 小时。完整源帧账本核验通过；这些是候选，尚未应用音质或分辨率筛选，尚未导出全量裁剪视频。原生人脸框及预裁剪框按时长统计；未确定框、上下布局及框位置不一致分别标记，不直接丢弃 pair。

## 导航

- [完整规则、统计口径及局限](snapshot/README.md)
- [切分及参数](snapshot/code/pipeline.py)、[镜头/人脸几何复核](snapshot/code/audit_geometry.py)
- [全流程入口](snapshot/code/run_all.py)、[全量核验](snapshot/code/validate_all.py)
- [汇总结果](snapshot/reports/summary.json)、[阈值模拟](snapshot/reports/threshold_retention.csv)
- [报告页面源码](snapshot/reports/index.html)、[生成代码](snapshot/code/report.py)
- [已审核 152 条参考 pair](dependencies/approved_reference/review/samples.json)
- [源文件逐字节 SHA256](BACKUP_MANIFEST.json)、[本次备份检查](BACKUP_VALIDATION.json)

## 备份范围

snapshot 保留生产文件原文；dependencies 补齐已审核示例及其 ASR/语义边界证据、预览目录和 HTTP Range 服务代码。environment.json/requirements.txt 记录本机 Python 包版本；运行还需要 ffmpeg、ffprobe（支持 subfile）。浏览器检查需要 Playwright Chromium。字体许可见 dependencies/font-LICENSE。

未提交原始视频、音频、模型、标注缓存、扫描缓存及动态日志。全量 pair 表和 692 个 manifest 仍位于 `/private/liqi_workspace/realtalk_processing/full_20261023`；大产物路径及 SHA256 见 LOCAL_OUTPUTS.json。这是代码与复核证据备份，不是完整数据集镜像。报告下载链接依赖本地全量产物，不能将 GitHub 中的静态页面视为完整在线报告。

## 恢复复跑

原环境可直接使用原生产入口。迁移环境先运行：

```bash
python realtalk/restore.py --workdir /path/to/new_run --data-root /path/to/realtalk
OPENBLAS_NUM_THREADS=1 python /path/to/new_run/code/run_all.py
python /path/to/new_run/code/serve.py
```

恢复脚本要求新目录，复制代码及依赖并仅改写路径；不复制旧 index.json，首次运行从原始 tar/ASR 重建索引。数据目录需要保留原有 tar 分片和 asr 子目录结构。默认 32 个处理进程，需调整时修改恢复目录中的 run_all.py；预览默认端口 8798。全量程序未重新运行 Whisper，源 ASR 时间可能不准，待复核标记必须保留。

## 分辨率抽样审核

新增 [11 档、33 条同步对照抽样](resolution_review/README.md)，用于人工确定阈值，未执行阈值筛选。

全量v2入口：[full_v2](full_v2/README.md)，复现已确认137条pair后扩展到692条源视频，保留分离与优化双评分。
