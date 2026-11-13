# 当前AV-only6000＋SFT2752总结果页更新

**当前500h AV-only6000＋SFT2752双200评测已完成并加入总结果页（2026-11-08）：** 400条最终验收通过；ResponseNet／Seamless各16组、25项差值对照，新增“相同SFT2752条件：当前仅AV→联合6000”，历史AV基线独立保留。总体指标、音频覆盖、第一／第二名配色、B−A箭头及CSV同步更新；700项数值／差值和浏览器验收通过。[总结果页](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=seamless&ablation=joint_whisper#ablation-joint_whisper)。

新增行ID：current500h_av_only_6000_sft2752；新增差值ID：current_av_joint6000。比较A为当前纯AV、B为主干Event＋Feature＋Whisper6000；相同400条SFT2752规划、H和Gemma。未将历史原生Qwen AV6000冒充本轮基线。

直接读取本实验RESULTS.json，未修改评分、生成媒体或历史结果。dashboard.json记录来源SHA，CSV导出全部16组及25项差值；BROWSER_AUDIT.json包含两数据集、前两名配色、手机布局、原视频／音轨、CSV下载和700项数值检查。代码快照与CODE_SHA256.json同目录保留。
