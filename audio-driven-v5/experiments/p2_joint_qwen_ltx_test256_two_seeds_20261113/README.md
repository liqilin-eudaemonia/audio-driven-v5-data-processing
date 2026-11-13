# 四组Seamless256：seed137／238均值选权重

用户授权：当前seed137全部完成后，使用相同四组和相同256条，换一个seed再测，以两轮各项指标的平均值选择权重搭配。仍仅node8（172.29.104.237）和node12（172.29.252.37）GPU0–7，共16卡，node11保持暂停。

Seed137已于北京时间2026-11-13 17:50:45完成；自动接续程序确认1024条视频、四组汇总和原控制器退出后，于17:52:57启动seed238。

- [接续状态](status.json) · [第二轮实时状态](../p2_joint_qwen_ltx_test256_seed238_16gpu_20261113/status.json)
- [Seed137结果](../p2_joint_qwen_ltx_test256_16gpu_20261113/RESULTS.md)
- [Seed238结果](../p2_joint_qwen_ltx_test256_seed238_16gpu_20261113/RESULTS.md)
- [最终双seed平均表](RESULTS.md) · [均值、标准差、逐seed原值与有效覆盖](RESULTS.json)（两轮均完成后自动生成）
- [接续代码](followup.py) · [平均汇总代码](aggregate.py) · [CPU检查](CPU_ACCEPTANCE.json)

## 冻结条件

四组：8256＋6800、7500＋6800、2752＋6000、原生Qwen＋6800。两轮均固定同256条、相同规划文本／hidden／Gemma／speaker Whisper、首帧、裁框、模型权重、30步Euler、视频CFG3、音频CFG7和原生评分配置。

仅LTX音视频生成seed由137改为238。规划使用原来的确定性输出，1024份条件直接复用；第二轮1024条音视频和评分全部重新生成，不能复用seed137的latent、视频或评分。本轮的seed238由现有双seed实验惯例确定，不根据结果挑seed。首次新latent会自动核验四组实际记录的seed238、权重和规划指纹，记录到SEED238_LIVE_ACCEPTANCE.json。

使用原生生成函数，两个torch.manual_seed调用均读取协议seed。发布评分输入时明确核验metadata.seed==238，原历史publish中写死137的断言仅在本轮入口改为协议seed；共享原生库和正在运行的第一轮代码未修改。

## 平均与选择口径

各指标分别等权计算(seed137＋seed238)/2，同时保留两个原值、样本标准差和绝对差。FID／FVD分别在每轮全部256条上合并特征算出后再取平均；不将两个seed池化成512条独立来源。音频指标包含漏回应、静默误声、时间匹配F1等；UTMOS和其他缺失值不补0，只有两个seed都有有效数值才给均值。表情／ΔResp的有效覆盖分seed保存。

最终表每项均值第一名加粗，第二名下划线。以各项平均表现结合已约定的自然回应目标选择搭配，不临时给不同量纲指标加权造一个总分；指标有取舍时如实列出。两个seed的波动用于检查排名稳定性，不宣称统计显著性。

## 调度、停止和恢复

followup.py在node12后台独立运行：等待第一轮完整收据且控制器退出，确认授权16卡空闲后启动第二轮；只允许这两台机器，不停止外部进程。第二轮沿用14路生成和解码／评分接续，以及尾部动态增加每组worker的策略。第二轮控制器意外退出允许一次有界重启，完成产物继续复用。两轮结束自动执行aggregate.py。

接续PID见CONTROLLER.json，日志followup.log；第二轮PID与日志在其CONTROLLER.json和controller.log。任一轮目录或本目录创建STOP会停止自动接续；若第二轮已由该程序启动，也会终止其控制器并保留已完成结果。

恢复前确认原接续进程退出且两机仍获授权，移除STOP，在node12执行：

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -u experiments/p2_joint_qwen_ltx_test256_two_seeds_20261113/followup.py
```

flock防重复；如第二轮控制器仍运行，接续程序只监测并在完成后汇总，不另起一轮。第二轮代码／协议指纹见其PREPARED.json。复核均值可单独运行本目录aggregate.py，但必须两轮完整结束。

## 用户新增资源指令：下一轮3机24卡（待测评范围答复）

用户要求：当前双seed完成后，停止node11的任务，下一轮使用node8、node12、node11各8卡，共24卡。已准备NEXT_CLUSTER24_PENDING.json，记录NEXT_EVALUATION_REQUEST.json；本轮seed238的代码与资源保持原样。

下一轮是Seamless完整test6624，以及由用户先选组合还是自动推荐两组，已向用户询问，尚无答复。当前没有启动下一轮或执行node11停止动作，也未把不明确的下一轮接到自动执行链上。node11只读核查显示8卡有占用；GPU进程列表未返回PID，ps快照超时，实际停止前需重新定位任务与守护进程，不能按旧PID盲杀。
