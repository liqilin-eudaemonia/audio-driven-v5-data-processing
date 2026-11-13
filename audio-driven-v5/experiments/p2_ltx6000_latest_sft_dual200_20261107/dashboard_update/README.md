# 最新step6000指标页面更新

**最新step6000＋SFT2752已加入原测评页面（2026-11-07）：** ResponseNet／Seamless各15组，7组消融、24项对照；新增主干Event＋Feature＋Whisper6000＋SFT2752，与历史联合6000、仅AV6000及AV＋Event6000对比。原值、B−A、指标箭头、前两名配色和音频覆盖均保留；注明数据／Event读出／规划器多项变化，不能归因于单一组件。672项数值／差值及浏览器验收通过。[打开新增对照](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=seamless&ablation=joint_whisper#ablation-whisper)。

读取原始RESULTS.json，未重新推理或评分，未添加6500未完成结果。新增系统ID为 `ltx_hidden_event_feature_whisper_6000_sft2752`。两套测试集均使用原清单、seed和采样配置。`dashboard.json`记录结果文件SHA，`BROWSER_AUDIT.json`记录页面／CSV／原有视频音轨与672项数值验证。

构建和验收入口：`scripts/build_p2_metrics_dashboard.py`、`scripts/check_p2_metrics_dashboard.py`；代码快照及SHA在本目录。原始15组CSV使用 `*_all_summary.csv`，历史 `*_all14_summary.csv` URL兼容导出当前全部15组，避免旧视频页下载断链。
