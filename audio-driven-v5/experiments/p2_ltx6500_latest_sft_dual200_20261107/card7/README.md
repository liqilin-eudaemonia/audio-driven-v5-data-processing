# step6500 仅 node11 GPU7 接续评测

按用户最新要求，6500 评测恢复至 node11（172.29.189.119）的物理 GPU7。先前暂停标志归档为 `PREVIOUS_STOP.txt`，仅移除本评测 STOP；其他训练及其停止标志未改动。旧八卡调度已退出，不再使用 GPU0–6。

保留切换前已提交的 163 条 latent，固定 ResponseNet 分层随机200＋Seamless固定200共400条；剩余生成→视频／音频解码→评分→全量指标汇总自动顺序接续。评分仍使用原8份清单分片，在同一张物理GPU7上执行，不意味着使用8张卡。允许与GPU7已有任务叠加，不停止其他任务。

权重是三机24卡目录的 `step_006500.pt`，最新规划SFT2752，复用6000组400条H／Gemma／speaker Whisper条件。seed137、Euler30、视频／音频CFG3／7、GT及指标协议均保持不变。6000组已完成400条并通过 FINAL_AUDIT。

控制入口：`scripts/p2_ltx6500_gpu7_resume_20261107.py run`。只在node11启动；进程锁防止重复。状态为上级 `status.json`；本目录 `render_process.json`、`render_attempt1.log`、`controller.log` 记录当前进程和日志，失败写 `FAILURE.json`。最终以 `../FINAL_AUDIT.json` 和 `../HANDOFF_COMPLETE.json` 为准。保留共享条件目录，勿删除6000组。

没有恢复聊天定时监控或Goal。STOP仍可在本评测目录停止后续领取和阶段。


2026-11-07解码阶段：增加GPU7同卡辅助解码，原1进程＋3辅助，具体入口与实测见[同卡解码提速](decode_assist/README.md)。评分仍由原控制器接续。
